import os
import torch

import sam_pt.point_tracker.utils.improc
import sam_pt.point_tracker.utils.samp
from sam_pt.point_tracker import PointTracker
from .raftnet import Raftnet


class RaftPointTracker(PointTracker):
    """
    Implements a point tracker that uses the RAFT algorithm for optical flow estimation
    from https://arxiv.org/abs/2003.12039. The tracker computes forward and backward flows
    for each frame in a video sequence and uses these to estimate the trajectories of given points.
    """

    def __init__(self, checkpoint_path):
        """
        Args:
            checkpoint_path (str): The path to the trained RAFT model checkpoint.
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is not None and not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Raft checkpoint not found at {self.checkpoint_path}")
        print(f"Loading Raft model from {self.checkpoint_path}")
        self.model = Raftnet(ckpt_name=self.checkpoint_path)

    def forward(self, rgbs, query_points, summary_writer=None):
        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]

        prep_rgbs = sam_pt.point_tracker.utils.improc.preprocess_color(rgbs)

        flows_forward = []
        flows_backward = []
        for t in range(1, n_frames):
            rgb0 = prep_rgbs[:, t - 1]
            rgb1 = prep_rgbs[:, t]
            flows_forward.append(self.model.forward(rgb0, rgb1, iters=32)[0])
            flows_backward.append(self.model.forward(rgb1, rgb0, iters=32)[0])
        flows_forward = torch.stack(flows_forward, dim=1)
        flows_backward = torch.stack(flows_backward, dim=1)
        assert flows_forward.shape == flows_backward.shape == (batch_size, n_frames - 1, 2, height, width)

        coords = []
        for t in range(n_frames):
            if t == 0:
                coord = torch.zeros_like(query_points[:, :, 1:])
            else:
                prev_coord = coords[t - 1]
                delta = sam_pt.point_tracker.utils.samp.bilinear_sample2d(
                    im=flows_forward[:, t - 1],
                    x=prev_coord[:, :, 0],
                    y=prev_coord[:, :, 1],
                ).permute(0, 2, 1)
                assert delta.shape == (batch_size, n_points, 2), "Forward flow at the discrete points"
                coord = prev_coord + delta

            # Set the ground truth query point location if the timestep is correct
            query_point_mask = query_points[:, :, 0] == t
            coord = coord * ~query_point_mask.unsqueeze(-1) + query_points[:, :, 1:] * query_point_mask.unsqueeze(-1)

            coords.append(coord)

        for t in range(n_frames - 2, -1, -1):
            coord = coords[t]
            successor_coord = coords[t + 1]

            delta = sam_pt.point_tracker.utils.samp.bilinear_sample2d(
                im=flows_backward[:, t],
                x=successor_coord[:, :, 0],
                y=successor_coord[:, :, 1],
            ).permute(0, 2, 1)
            assert delta.shape == (batch_size, n_points, 2), "Backward flow at the discrete points"

            # Update only the points that are located prior to the query point
            prior_to_query_point_mask = t < query_points[:, :, 0]
            coord = (coord * ~prior_to_query_point_mask.unsqueeze(-1) +
                     (successor_coord + delta) * prior_to_query_point_mask.unsqueeze(-1))
            coords[t] = coord

        trajectories = torch.stack(coords, dim=1)
        visibilities = (trajectories[:, :, :, 0] >= 0) & \
                       (trajectories[:, :, :, 1] >= 0) & \
                       (trajectories[:, :, :, 0] < width) & \
                       (trajectories[:, :, :, 1] < height)
        return trajectories, visibilities
