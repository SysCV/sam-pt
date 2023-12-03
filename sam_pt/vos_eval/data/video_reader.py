# Taken from: https://github.com/hkchengrex/XMem/blob/083698bbb4c5ac0ffe1a8923a6c313de46169983/inference/data/video_reader.py

import os
from os import path

import numpy as np
import torch.nn.functional as F
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class VideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """

    def __init__(self, vid_name, image_dir, mask_dir,
                 shortest_size=-1, longest_size=None,
                 to_save=None, use_all_mask=False, size_dir=None,
                 mask_mode='P', mask_dtype=np.uint8,
                 ):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        assert shortest_size == -1 or longest_size is None, 'One size constraint should be given, not both.'

        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        self.mask_mode = mask_mode
        self.mask_dtype = mask_dtype

        self.frames = sorted(os.listdir(self.image_dir))
        self.palette = Image.open(path.join(mask_dir, sorted(os.listdir(mask_dir))[0])).getpalette()
        self.first_gt_path = path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[0])

        # TODO SegGPT specific
        if shortest_size == "seggpt":
            shortest_size = (448, 448)

        self.shortest_size = shortest_size
        self.longest_size = longest_size

        # TODO: Model specific transforms are hardcoded here
        if self.shortest_size == -1 and self.longest_size is None:
            self.resize_longest_side_transform = None
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.shortest_size != -1:
            self.resize_longest_side_transform = None
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.shortest_size, interpolation=InterpolationMode.BILINEAR),
            ])
        elif self.longest_size is not None:
            self.resize_longest_side_transform = ResizeLongestSide(self.longest_size)
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise RuntimeError('Invalid size constraints.')

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        gt_path = path.join(self.mask_dir, frame[:-4] + '.png')
        if self.resize_longest_side_transform is not None:
            img = np.array(img)
            img = self.resize_longest_side_transform.apply_image(img)

        img = self.im_transform(img)

        load_mask = self.use_all_mask or (gt_path == self.first_gt_path)
        if load_mask and path.exists(gt_path):
            mask = Image.open(gt_path).convert(self.mask_mode)
            mask = np.array(mask, dtype=self.mask_dtype)
            data['mask'] = mask

        info['shape'] = shape
        info['need_resize'] = self.shortest_size != 0 or self.longest_size is not None
        data['rgb'] = img
        data['info'] = info

        # TODO: SegGPT specific
        if self.shortest_size == (448, 448):
            info['shape'] = (448, 448)

        return data

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        old_h, old_w = mask.shape[-2:]
        if self.resize_longest_side_transform is None:
            min_hw = min(old_h, old_w)
            if self.shortest_size == (448, 448):
                # TODO SegGPT specific
                shape = (448, 448)
            else:
                shape = (int(old_h / min_hw * self.shortest_size), int(old_w / min_hw * self.shortest_size))
        else:
            shape = ResizeLongestSide.get_preprocess_shape(old_h, old_w, self.longest_size)
        return F.interpolate(mask, shape, mode='nearest')

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)
