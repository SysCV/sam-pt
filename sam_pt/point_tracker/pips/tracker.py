import torch
from tqdm import tqdm

from sam_pt.point_tracker.pips import Pips
from sam_pt.point_tracker.tracker import PointTracker
from sam_pt.point_tracker.utils import saverloader


class PipsPointTracker(PointTracker):
    """
    The PipsPointTracker class implements a Point Tracker using the Persistent Independent Particles (PIPS) model
    from https://arxiv.org/abs/2204.04153. This tracker will run the PIPS model in both left-to-right and right-to-left
    directions to propagate the query points to all frames, merging the outputs to get the final predictions.
    """

    def __init__(self, checkpoint_path, stride, s, initial_next_frame_visibility_threshold=0.9):
        """
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        stride : int
            Stride parameter for the PIPS model.
        s : int
            Window size parameter for PIPS model.
        initial_next_frame_visibility_threshold : float, optional
            Initial threshold value for the next frame visibility. Default is 0.9.
        """

        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.stride = stride
        self.s = s
        self.initial_next_frame_visibility_threshold = initial_next_frame_visibility_threshold

        print(f"Loading PIPS model from {self.checkpoint_path}")
        self.model = Pips(S=s, stride=stride)
        self._loaded_checkpoint_step = saverloader.load(self.checkpoint_path, self.model,
                                                        device="cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model

    def _forward(self, rgbs, query_points):
        """
        Performs forward passes of the PIPS model from left to right
        and returns the predicted trajectories and visibilities.
        """
        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]

        if not batch_size == 1:
            raise NotImplementedError("Batch size > 1 is not supported for PIPS yet")

        # Batched version of the forward pass
        trajectories = torch.zeros((n_frames, n_points, 2), dtype=torch.float32, device=rgbs.device)
        visibilities = torch.zeros((n_frames, n_points), dtype=torch.float32, device=rgbs.device)

        start_frames = query_points[0, :, 0].long()
        visibilities[start_frames, torch.arange(n_points)] = 1.0
        trajectories[start_frames, torch.arange(n_points), :] = query_points[0, :, 1:]

        # Make a forward pass for each frame, performing the trajectory linking (described in the PIPs paper)
        # where each point is linking its trajectory as to follow the trace a high query point visibility
        # The state will therefore not be updated for all points at every frame but only for the points that use
        # the current frame in their trajectory linking
        feat_init = torch.zeros((batch_size, n_points, self.model.latent_dim), dtype=torch.float32, device=rgbs.device)
        current_point_frames = start_frames.clone()
        for current_frame in tqdm(range(n_frames - 1)):
            # Skip the forward pass if none of the points have it as their current frame
            if (current_point_frames == current_frame).sum() == 0:
                continue

            # 1. Prepare the forward pass for the current frame
            rgbs_input = rgbs[:, current_frame:current_frame + self.s, :, :, :]
            n_missing_rgbs = self.s - rgbs_input.shape[1]
            if n_missing_rgbs > 0:
                last_rgb = rgbs_input[:, -1, :, :, :]
                missing_rgbs = last_rgb.unsqueeze(1).repeat(1, self.s - rgbs_input.shape[1], 1, 1, 1)
                rgbs_input = torch.cat([rgbs_input, missing_rgbs], dim=1)

            # 2. Run the first forward pass to initialize the feature vector
            feat_init_forward_pass_points = start_frames == current_frame
            if (feat_init_forward_pass_points).any():
                _, _, _, feat_init_update, _ = self.model.forward(
                    xys=trajectories[None, current_frame, feat_init_forward_pass_points, :],
                    rgbs=rgbs_input,
                    feat_init=None,
                    iters=6,
                    return_feat=True,
                )
                feat_init[:, start_frames == current_frame, :] = feat_init_update[:, :, :]

            # 3. Run the forward pass to update the state
            forward_pass_points = current_point_frames == current_frame
            output_trajectory_per_iteration, _, output_visibility_logits, _, _ = self.model.forward(
                xys=trajectories[None, current_frame, forward_pass_points, :],
                rgbs=rgbs_input,
                feat_init=feat_init[:, forward_pass_points, :],
                iters=6,
                # sw=summary_writer,  # Slow
                return_feat=True,
            )
            output_visibility = torch.sigmoid(output_visibility_logits).float()  # TODO Hack: convert to float32
            output_trajectory = output_trajectory_per_iteration[-1].float()  # TODO Hack: convert to float32

            # 3. Update the state
            output_frame_slice = slice(1, self.s - n_missing_rgbs)
            predicted_frame_slice = slice(1 + current_frame, current_frame + self.s - n_missing_rgbs)
            visibilities[predicted_frame_slice, forward_pass_points] = output_visibility[0, output_frame_slice, :]
            trajectories[predicted_frame_slice, forward_pass_points, :] = output_trajectory[0, output_frame_slice, :, :]

            # 4. Update the current point frames
            next_frame_visibility_thresholds = torch.where(
                current_point_frames == current_frame,
                torch.ones(n_points, device=rgbs.device) * self.initial_next_frame_visibility_threshold,
                torch.zeros(n_points, device=rgbs.device),
            )
            next_frame_earliest_candidates = torch.where(
                current_point_frames == current_frame,
                current_point_frames + 1,
                current_point_frames,
            )
            next_frame_last_candidates = torch.where(
                current_point_frames == current_frame,
                current_point_frames + self.s - n_missing_rgbs - 1,
                current_point_frames,
            )
            next_frames = next_frame_last_candidates
            while (visibilities[next_frames, torch.arange(n_points)] <= next_frame_visibility_thresholds).any():
                next_frames = torch.where(
                    visibilities[next_frames, torch.arange(n_points)] <= next_frame_visibility_thresholds,
                    next_frames - 1,
                    next_frames,
                )
                next_frame_visibility_thresholds = torch.where(
                    next_frames < next_frame_earliest_candidates,
                    next_frame_visibility_thresholds - 0.02,
                    next_frame_visibility_thresholds,
                )
                next_frames = torch.where(
                    next_frames < next_frame_earliest_candidates,
                    next_frame_last_candidates,
                    next_frames,
                )
            current_point_frames = torch.where(
                current_point_frames == current_frame,
                next_frames,
                current_point_frames,
            )

        visibilities = visibilities > 0.5
        visibilities = visibilities.unsqueeze(0)
        trajectories = trajectories.unsqueeze(0)
        return trajectories, visibilities

    def forward(self, rgbs, query_points):
        query_points = query_points.float()

        # From left to right
        trajectories_to_right, visibilities_to_right = self._forward(rgbs, query_points)

        # From right to left
        rgbs_flipped = rgbs.flip(1)
        query_points_flipped = query_points.clone()
        query_points_flipped[:, :, 0] = rgbs.shape[1] - query_points_flipped[:, :, 0] - 1
        trajectories_to_left, visibilities_to_left = self._forward(rgbs_flipped, query_points_flipped)
        trajectories_to_left = trajectories_to_left.flip(1)
        visibilities_to_left = visibilities_to_left.flip(1)

        # Merge
        trajectory_list = []
        visibility_list = []
        n_points = query_points.shape[1]
        for point_idx in range(n_points):
            start_frame = int(query_points[0, point_idx, 0].item())

            trajectory = torch.cat([
                trajectories_to_left[0, :start_frame, point_idx, :],
                trajectories_to_right[0, start_frame:, point_idx, :]
            ])
            visibility = torch.cat([
                visibilities_to_left[0, :start_frame, point_idx],
                visibilities_to_right[0, start_frame:, point_idx],
            ])

            assert trajectory.shape == trajectories_to_right[0, :, point_idx, :].shape
            assert visibility.shape == visibilities_to_right[0, :, point_idx].shape

            assert torch.allclose(trajectories_to_right[0, start_frame, point_idx, :], query_points[0, point_idx, 1:])
            assert torch.allclose(trajectories_to_left[0, start_frame, point_idx, :], query_points[0, point_idx, 1:])
            assert torch.allclose(trajectory[start_frame, :], query_points[0, point_idx, 1:])

            assert visibilities_to_right[0, start_frame, point_idx] == 1.0
            assert visibilities_to_left[0, start_frame, point_idx] == 1.0
            assert visibility[start_frame] == 1.0

            trajectory_list += [trajectory]
            visibility_list += [visibility]

        trajectories = torch.stack(trajectory_list, dim=1).unsqueeze(0)
        visibilities = torch.stack(visibility_list, dim=1).unsqueeze(0)
        return trajectories, visibilities
