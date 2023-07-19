# Adapted from: https://github.com/aharley/pips/blob/486124b4236bb228a20750b496f0fa8aa6343157/nets/raftnet.py

import argparse

import torch
import torch.nn as nn

from .raft_core.raft import RAFT
from .raft_core.util import InputPadder


class Raftnet(nn.Module):
    def __init__(self, ckpt_name=None, small=False, alternate_corr=False, mixed_precision=True):
        super(Raftnet, self).__init__()
        args = argparse.Namespace()
        args.small = small
        args.alternate_corr = alternate_corr
        args.mixed_precision = mixed_precision
        self.model = RAFT(args)
        if ckpt_name is not None:
            state_dict = torch.load(ckpt_name)
            state_dict = {  # The checkpoint was saved as wrapped in nn.DataParallel, this removes the wrapper
                k.replace('module.', ''): v
                for k, v in state_dict.items()
                if k != 'module'
            }
            self.model.load_state_dict(state_dict)

    def forward(self, image1, image2, iters=20, test_mode=True):
        # input images are in [-0.5, 0.5]
        # raftnet wants the images to be in [0,255]
        image1 = (image1 + 0.5) * 255.0
        image2 = (image2 + 0.5) * 255.0

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        if test_mode:
            flow_low, flow_up, feat = self.model(image1=image1, image2=image2, iters=iters, test_mode=test_mode)
            flow_up = padder.unpad(flow_up)
            return flow_up, feat
        else:
            flow_predictions = self.model(image1=image1, image2=image2, iters=iters, test_mode=test_mode)
            return flow_predictions
