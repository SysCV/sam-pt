from collections import defaultdict

import torch

from sam_pt.point_tracker import PointTracker
from sam_pt.point_tracker.pips_plus_plus import PipsPlusPlus
from sam_pt.point_tracker.utils import saverloader


class PipsPlusPlusPointTracker(PointTracker):

    def __init__(self, checkpoint_path, stride=8, max_sequence_length=128, iters=16, image_size=(512, 896)):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.stride = stride
        self.max_sequence_length = max_sequence_length
        self.iters = iters
        self.image_size = tuple(image_size) if image_size is not None else None

        print(f"Loading PIPS++ model from {self.checkpoint_path}")
        self.model = PipsPlusPlus(stride=self.stride)
        self._loaded_checkpoint_step = saverloader.load(self.checkpoint_path, self.model,
                                                        device="cuda" if torch.cuda.is_available() else "cpu")

    def _forward(self, rgbs, query_points):
        """
        Single direction forward pass.
        """
        B, S, C, H, W = rgbs.shape
        assert query_points.ndim == 2
        assert query_points.shape[1] == 2

        # zero-vel init
        trajs_e = query_points[None, None, :, :].repeat(1, rgbs.shape[1], 1, 1)

        cur_frame = 0
        done = False
        feat_init = None
        while not done:
            end_frame = cur_frame + self.max_sequence_length

            if end_frame > S:
                diff = end_frame - S
                end_frame = end_frame - diff
                cur_frame = max(cur_frame - diff, 0)

            traj_seq = trajs_e[:, cur_frame:end_frame]
            rgb_seq = rgbs[:, cur_frame:end_frame]
            S_local = rgb_seq.shape[1]

            if feat_init is not None:
                feat_init = [fi[:, :S_local] for fi in feat_init]

            preds, preds_anim, feat_init, _ = self.model(traj_seq, rgb_seq, iters=self.iters, feat_init=feat_init)

            trajs_e[:, cur_frame:end_frame] = preds[-1][:, :S_local]
            trajs_e[:, end_frame:] = trajs_e[:, end_frame - 1:end_frame]  # update the future with new zero-vel

            if end_frame >= S:
                done = True
            else:
                cur_frame = cur_frame + self.max_sequence_length - 1

        visibilities = torch.ones_like(trajs_e[:, :, :, 0])
        return trajs_e, visibilities

    def forward(self, rgbs, query_points):
        """
        Forward function for the tracker.
        """
        batch_size, num_frames, C, H, W = rgbs.shape
        if self.image_size is not None:
            rgbs = rgbs.reshape(batch_size * num_frames, C, H, W)
            rgbs = rgbs / 255.0
            rgbs = torch.nn.functional.interpolate(rgbs, size=tuple(self.image_size), mode="bilinear")
            rgbs = rgbs * 255.0
            rgbs = rgbs.reshape(batch_size, num_frames, C, *self.image_size)
            query_points[:, :, 1] *= self.image_size[0] / H
            query_points[:, :, 2] *= self.image_size[1] / W

        # Group query points by their time-step
        groups = defaultdict(list)
        assert query_points.shape[0] == batch_size == 1, "Only batch size 1 is supported."
        for idx, point in enumerate(query_points[0]):
            t = int(point[0].item())
            groups[t].append((idx, point[1:].tolist()))

        # Dictionary to store results
        trajectories_dict = {}
        visibilities_dict = {}

        for t, points_with_indices in groups.items():
            points = [x[1] for x in points_with_indices]

            # Left to right
            if t == num_frames - 1:
                left_trajectories = torch.empty((batch_size, 0, len(points), 2), dtype=torch.float32).cuda()
                left_visibilities = torch.empty((batch_size, 0, len(points)), dtype=torch.float32).cuda()
            else:
                left_rgbs = rgbs[:, t:]
                left_query = torch.tensor(points, dtype=torch.float32).cuda()
                left_trajectories, left_visibilities = self._forward(left_rgbs, left_query)

            # Right to left
            if t == 0:
                right_trajectories = torch.empty((batch_size, 0, len(points), 2), dtype=torch.float32).cuda()
                right_visibilities = torch.empty((batch_size, 0, len(points)), dtype=torch.float32).cuda()
            else:
                right_rgbs = rgbs[:, :t + 1].flip(1)
                right_query = torch.tensor(points, dtype=torch.float32).cuda()
                right_trajectories, right_visibilities = self._forward(right_rgbs, right_query)
                right_trajectories = right_trajectories.flip(1)
                right_visibilities = right_visibilities.flip(1)

            # Merge the results
            trajectories = torch.cat([right_trajectories[:, :-1], left_trajectories], dim=1)
            visibilities = torch.cat([right_visibilities[:, :-1], left_visibilities], dim=1)

            # Store in dictionary
            for idx, (idx, _) in enumerate(points_with_indices):
                trajectories_dict[idx] = trajectories[:, :, idx, :]
                visibilities_dict[idx] = visibilities[:, :, idx]

        # Assemble the results back in the order of the input query points
        n_points = query_points.shape[1]
        final_trajectories = torch.stack([trajectories_dict[i] for i in range(n_points)], dim=2)
        final_visibilities = torch.stack([visibilities_dict[i] for i in range(n_points)], dim=2)

        # Rescale trajectories back to the original size
        if self.image_size is not None:
            final_trajectories[:, :, :, 0] *= H / self.image_size[0]
            final_trajectories[:, :, :, 1] *= W / self.image_size[1]

        return final_trajectories, final_visibilities
