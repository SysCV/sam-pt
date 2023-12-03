"""
A module which combines SAM (segment anything) with point tracking to perform video segmentation.
"""
from typing import Optional

import numpy as np
import torch
from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from skimage import color
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from sam_pt.point_tracker import PointTracker, SuperGluePointTracker
from sam_pt.utils.query_points import extract_kmedoid_points, extract_random_mask_points, extract_corner_points, \
    extract_mixed_points
from sam_pt.utils.util import PointVisibilityType


class SamPt(nn.Module):
    """
    The SamPt class is a PyTorch module which combines SAM (segment anything) with point tracking for video
    segmentation tasks. It provides a high-level interface for video segmentation by tracking points in a video
    using point trackers and applying SAM on top of the predicted point trajectories to get segmentation masks.
    """

    def __init__(
            self,
            point_tracker: PointTracker,
            sam_predictor: SamPredictor,
            sam_iou_threshold: float,
            positive_point_selection_method: str,
            negative_point_selection_method: str,
            positive_points_per_mask: int,
            negative_points_per_mask: int,
            add_other_objects_positive_points_as_negative_points: bool,
            max_other_objects_positive_points: Optional[int],
            point_tracker_mask_batch_size: int,
            iterative_refinement_iterations: bool,
            use_patch_matching_filtering: bool,
            patch_size: int,
            patch_similarity_threshold: float,
            use_point_reinit: bool,
            reinit_point_tracker_horizon: int,
            reinit_horizon: int,
            reinit_variant: str,
    ):
        """
        Parameters
        ----------
        point_tracker : PointTracker
            An object of the PointTracker class responsible for tracking the points across frames.
        sam_predictor : SamPredictor
            An object of the SamPredictor class used for the prediction of the masks.
        sam_iou_threshold : float
            The IoU threshold that determines whether a mask will be set to zeros if SAM's IoU prediction is below it.
        positive_point_selection_method : str
            The method to be used for selecting positive points.
        negative_point_selection_method : str
            The method to be used for selecting negative points.
        positive_points_per_mask : int
            The number of positive points per mask.
        negative_points_per_mask : int
            The number of negative points per mask.
        add_other_objects_positive_points_as_negative_points : bool
            If True, the positive points of other objects are added as negative points when using the SamPredictor.
        max_other_objects_positive_points : Optional[int]
            The maximum number of positive points of other objects to be added as negative points.
        point_tracker_mask_batch_size : int
            The batch size for the point tracker mask.
        iterative_refinement_iterations : bool
            If True, enables iterative refinement of the predicted mask.
        use_patch_matching_filtering : bool
            If True, use patch matching to filter points in trajectories that have very different looking patches.
        patch_size : int
            The size of the patch used in patch matching.
        patch_similarity_threshold : float
            The similarity threshold value for patch matching. Above the threshold, points are marked as non-visible.
        use_point_reinit : bool
            If True, enables point reinitialization.
        reinit_point_tracker_horizon : int
            The number of frames to run the point tracker for before performing reinitialization.
        reinit_horizon : int
            Number of frames after which reinitialization takes place. Can be shorter than the point tracker horizon.
        reinit_variant : str
            The reinitialization variant to be used.
        """
        super().__init__()

        self.point_tracker = point_tracker
        self.sam_predictor = sam_predictor
        self.sam_iou_threshold = sam_iou_threshold

        # Make baseline.to(device) work since the predictor is not a nn.Module
        self._sam: Sam = sam_predictor.model

        self.iterative_refinement_iterations = iterative_refinement_iterations

        self.positive_point_selection_method = positive_point_selection_method
        self.negative_point_selection_method = negative_point_selection_method
        self.positive_points_per_mask = positive_points_per_mask
        self.negative_points_per_mask = negative_points_per_mask
        self.add_other_objects_positive_points_as_negative_points = add_other_objects_positive_points_as_negative_points
        self.max_other_objects_positive_points = max_other_objects_positive_points

        self.point_tracker_mask_batch_size = point_tracker_mask_batch_size

        self.use_patch_matching_filtering = use_patch_matching_filtering
        self.patch_size = patch_size
        self.patch_similarity_threshold = patch_similarity_threshold

        self.use_point_reinit = use_point_reinit
        self.reinit_point_tracker_horizon = reinit_point_tracker_horizon
        self.reinit_horizon = reinit_horizon
        self.reinit_variant = reinit_variant

    @property
    def device(self):
        return self._sam.device

    def forward(self, video):
        """
        Evaluates SAM-PT on the video and outputs the predictions.

        Parameters
        ----------
        video : dict
            Dictionary with video data. It includes the following keys:

            - 'video_name' (str): The name of the video.
            - 'video_id' (int): The ID of the video.
            - 'image' (List[torch.Tensor]): The frames of the video as uint8 tensors of shape (channels, height, width).
            - 'info' (List[dict]): Information for each frame with keys such as 'frame', 'save', 'shape', 'need_resize'.
            - 'target_hw' (Tuple[int, int]): The target height and width for the predicted masks.

            Optional keys (at least one of 'query_points' or 'query_masks' must be provided):
            - 'query_points' (torch.Tensor): A float32 tensor of shape (num_masks, n_points_per_mask, 3) representing
                                              the query points defining the objects to be tracked. This tensor contains
                                              the (t, x, y) values of the query points, where t is the timestep defining
                                              the query video frame and x, y are the coordinates of the query point.
                                              Mutually exclusive with 'query_masks'.
            - 'query_masks' (torch.Tensor): Query masks as binary float32 tensor of shape (num_masks, height, width).
                                            Mutually exclusive with 'query_points'.
            - 'query_point_timestep' (torch.Tensor): The query point timesteps as float32 tensor of shape (num_masks,).
                                                     Required if 'query_masks' is provided.

        Returns
        -------
        dict
            Dictionary with predictions. It includes the following keys:

            - 'logits' (List[torch.Tensor]): The logits as float32 tensors of shape (num_frames, height, width).
            - 'trajectories' (torch.Tensor): The trajectories as float32 tensor
                                             of shape (num_frames, n_masks, n_points_per_mask, 2).
            - 'visibilities' (torch.Tensor): The visibilities as float32 tensor
                                             of shape (num_frames, n_masks, n_points_per_mask).
            - 'scores' (List[float]): The scores as list of 'num_masks' floats.
            - 'scores_per_frame' (List[List[float]]): Scores per frame per mask.
        """

        if self.training:
            raise NotImplementedError(f"{self._get_name()} does not support training...")

        # Unpack images
        images = torch.stack(video["image"], dim=0)
        n_frames, channels, height, width = images.shape
        assert images[0].dtype == torch.uint8, "Input images must be in uint8 format (0-255)"

        # Prepare queries
        if video.get("query_masks") is not None:  # E.g., when evaluating on the VOS task
            assert video.get("query_points") is None
            print("SAM-PT: Using query masks")
            query_masks = video["query_masks"].float()
            query_points_timestep = video["query_point_timestep"]
            query_points = self.extract_query_points(images, query_masks, query_points_timestep)
            query_scores = None
        elif video.get("query_points") is not None:  # E.g., when evaluating on the railway demo
            print("SAM-PT: Using query points")
            query_points = video["query_points"]
            query_masks = self.extract_query_masks(images, query_points)
            query_scores = None
        else:
            raise ValueError("No query points or masks provided")
        n_masks, n_points_per_mask, _ = query_points.shape
        assert query_masks.shape == (n_masks, height, width)

        # The SuperGlue point tracker performs keypoint matching and requires the query mask to be set
        if isinstance(self.point_tracker, SuperGluePointTracker):
            assert self.point_tracker_mask_batch_size >= n_masks
            self.point_tracker.set_masks(query_masks)

        # Run tracking
        self.frame_annotations = [[] for _ in range(n_frames)]
        if not self.use_point_reinit:
            trajectories, visibilities, logits, scores, scores_per_frame = self._forward(images, query_points)
        else:
            trajectories, visibilities, logits, scores, scores_per_frame = self._forward_w_reinit(images, query_points)

        # Post-processing
        target_hw = video["target_hw"]
        resize_factor = torch.tensor(target_hw) / torch.tensor(logits.shape[-2:])
        assert (resize_factor[0] - resize_factor[1]).abs().item() < 0.01, "The resizing should have been isotropic"

        if logits.shape[-2:] != target_hw:
            logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        trajectories = trajectories * resize_factor

        # masks = logits > .0

        if query_scores is not None:
            assert query_scores.shape == scores.shape
            final_scores = query_scores ** 4 * scores ** 0.4
            print("[Baseline]")
            print(f"Scores per frame: {scores_per_frame.tolist()}")
            print(f"Scores:       {scores.tolist()}")
            print(f"Query Scores: {query_scores.tolist()}")
            print(f"Final Scores: {final_scores.tolist()}")
        else:
            final_scores = scores

        assert logits.shape == (n_masks, n_frames, target_hw[0], target_hw[1])
        assert scores.shape == (n_masks,)
        assert scores_per_frame.shape == (n_frames, n_masks)
        assert trajectories.shape == (n_frames, n_masks, n_points_per_mask, 2)
        assert visibilities.shape == (n_frames, n_masks, n_points_per_mask)

        results_dict = {
            "logits": [m for m in logits],
            "scores": final_scores.tolist(),
            "scores_per_frame": scores_per_frame.tolist(),
            "trajectories": trajectories,
            "visibilities": visibilities,
        }

        return results_dict

    def extract_query_points(self, images, query_masks, query_points_timestep):
        """
        Extracts query points from the given images based on the provided masks and points' timesteps.

        Parameters
        ----------
        images : torch.Tensor
            The frames of the video from which to extract the query points,
            as a uint8 tensor of shape (num_frames, channels, height, width).
        query_masks : torch.Tensor
            The query masks as a binary float32 tensor of shape (num_masks, height, width) with values in {0, 1}.
        query_points_timestep : torch.Tensor
            The query point timesteps as float32 tensor of shape (num_masks,).

        Returns
        -------
        torch.Tensor
            A float32 tensor representing the query points extracted from the images. The tensor has shape
            (num_masks, n_points_per_mask, 3) and contains the (t, x, y) values of the query points, where
            t is the timestep defining the query video frame and x, y are the coordinates of the query point.

        Notes
        -----
        This function extracts both positive and negative points based on the provided masks.
        The number of points extracted for each category is determined by the class attributes:
        `positive_points_per_mask` and `negative_points_per_mask`.
        """
        query_masks = query_masks.cpu()
        query_points_timestep = query_points_timestep.cpu()

        query_points_xy = SamPt._extract_query_points_xy(
            images=images,
            query_masks=query_masks,
            query_points_timestep=query_points_timestep,
            point_selection_method=self.positive_point_selection_method,
            points_per_mask=self.positive_points_per_mask,
        )
        if self.negative_points_per_mask > 0:
            negative_query_masks = [1 - qm for qm in query_masks]
            negative_query_points_xy = SamPt._extract_query_points_xy(
                images=images,
                query_masks=negative_query_masks,
                query_points_timestep=query_points_timestep,
                point_selection_method=self.negative_point_selection_method,
                points_per_mask=self.negative_points_per_mask,
            )
            query_points_xy = [torch.cat(x, dim=0) for x in zip(query_points_xy, negative_query_points_xy)]
        query_points_xy = torch.stack(query_points_xy, dim=0)
        query_points_timestep = query_points_timestep[:, None, None].repeat(1, query_points_xy.shape[1], 1)
        query_points = torch.concat([query_points_timestep, query_points_xy], dim=2)
        return query_points

    @staticmethod
    def _extract_query_points_xy(images, query_masks, query_points_timestep, point_selection_method, points_per_mask):
        if point_selection_method == "kmedoids":
            query_points_xy = [extract_kmedoid_points(qm, points_per_mask) for qm in query_masks]
        elif point_selection_method == "shi-tomasi":
            query_points_xy = [
                extract_corner_points(images[int(t.item()), :, :, :], query_mask, points_per_mask)
                for query_mask, t in zip(query_masks, query_points_timestep)
            ]
        elif point_selection_method == "random":
            query_points_xy = [extract_random_mask_points(qm, points_per_mask) for qm in query_masks]
        elif point_selection_method == "mixed":
            query_points_xy = extract_mixed_points(query_masks, query_points_timestep, images,
                                                   points_per_mask)
        else:
            raise NotImplementedError(f"Point selection method {point_selection_method} not implemented")
        return query_points_xy

    def extract_query_masks(self, images, query_points):
        """
        Extracts query masks from the given images based on the provided query points.
        It does so by applying SAM based on the query points to extract the query masks.

        Parameters
        ----------
        images : torch.Tensor
            The frames of the video from which to extract the query masks,
            as a uint8 tensor of shape (num_frames, channels, height, width).
        query_points : torch.Tensor
            A float32 tensor representing the query points that define the query masks. The tensor has shape
            (num_masks, n_points_per_mask, 3) and contains the (t, x, y) values of the query points, where
            t is the timestep defining the query video frame and x, y are the coordinates of the query point.

        Returns
        -------
        torch.Tensor
            A tensor representing the extracted query masks. The tensor has shape (num_masks, height, width)
            and is a binary float32 tensor with values in {0, 1}.
        """
        _, query_masks_logits, _ = self._apply_sam_to_trajectories(
            images=torch.stack([images[int(t.item()), :, :, :] for t in query_points[:, 0, 0]], dim=0),
            trajectories=query_points[:, None, :, 1:],
            visibilities=torch.ones_like(query_points[:, None, :, 0]),
        )
        query_masks = query_masks_logits > self.sam_predictor.model.mask_threshold
        return query_masks[0]

    def _forward(self, images, query_points):
        """
        Forward pass without point re-initialization.

        Parameters
        ----------
        images : torch.Tensor
            The frames of the video as a uint8 tensor of shape (num_frames, channels, height, width).
        query_points : torch.Tensor
            A float32 tensor representing the query points that define the query masks. The tensor has shape
            (num_masks, n_points_per_mask, 3) and contains the (t, x, y) values of the query points, where
            t is the timestep defining the query video frame and x, y are the coordinates of the query point.
        """
        trajectories, visibilities = self._track_points(images, query_points)
        _, logits, scores_per_frame = self._apply_sam_to_trajectories(images, trajectories, visibilities)
        scores = scores_per_frame.mean(dim=0)
        return trajectories, visibilities, logits, scores, scores_per_frame

    def _forward_w_reinit(self, images, query_points):
        """
        Forward pass with point re-initialization using Sam's predicted masks.

        Parameters
        ----------
        images : torch.Tensor
            The frames of the video as a uint8 tensor of shape (num_frames, channels, height, width).
        query_points : torch.Tensor
            A float32 tensor representing the query points that define the query masks. The tensor has shape
            (num_masks, n_points_per_mask, 3) and contains the (t, x, y) values of the query points, where
            t is the timestep defining the query video frame and x, y are the coordinates of the query point.
        """
        n_frames, channels, height, width = images.shape

        # 1. Right-to-left
        (
            trajectories_right_to_left, visibilities_right_to_left,
            logits_right_to_left, _, scores_per_frame_right_to_left,
        ) = self._forward_w_reinit_inner(images, query_points)

        # 2. Left-to-right
        # Flip the images and the query points to track the points from right to left
        images_flipped = images.flip(0)
        query_points_flipped = query_points.clone()
        query_points_flipped[:, :, 0] = n_frames - query_points[:, :, 0] - 1
        (
            trajectories_left_to_right, visibilities_left_to_right,
            logits_left_to_right, _, scores_per_frame_left_to_right,
        ) = self._forward_w_reinit_inner(images_flipped, query_points_flipped)
        # Flip the trajectories back to the original order
        trajectories_left_to_right = trajectories_left_to_right.flip(0)
        visibilities_left_to_right = visibilities_left_to_right.flip(0)
        logits_left_to_right = logits_left_to_right.flip(1)

        # 3. Putting left-to-right and right-to-left together
        query_points_timestep = query_points[:, 0, 0].int()
        trajectories = torch.full_like(trajectories_right_to_left, torch.nan)
        visibilities = torch.full_like(visibilities_right_to_left, False)
        logits = torch.full_like(logits_right_to_left, torch.nan)
        scores_per_frame = torch.full_like(scores_per_frame_right_to_left, torch.nan)
        for mask_idx, timestep in enumerate(query_points_timestep):
            trajectories[timestep:, mask_idx] = trajectories_right_to_left[timestep:, mask_idx]
            trajectories[:timestep, mask_idx] = trajectories_left_to_right[:timestep, mask_idx]
            visibilities[timestep:, mask_idx] = visibilities_right_to_left[timestep:, mask_idx]
            visibilities[:timestep, mask_idx] = visibilities_left_to_right[:timestep, mask_idx]
            logits[mask_idx, timestep:] = logits_right_to_left[mask_idx, timestep:]
            logits[mask_idx, :timestep] = logits_left_to_right[mask_idx, :timestep]
            scores_per_frame[timestep:, mask_idx] = scores_per_frame_right_to_left[timestep:, mask_idx]
            scores_per_frame[:timestep, mask_idx] = scores_per_frame_left_to_right[:timestep, mask_idx]
        assert not torch.isnan(trajectories).any()
        assert not torch.isnan(logits).any()
        scores = scores_per_frame.nanmean(dim=0)
        return trajectories, visibilities, logits, scores, scores_per_frame

    def _forward_w_reinit_inner(self, images, query_points):
        n_frames, channels, height, width = images.shape
        n_masks, points_per_mask, _ = query_points.shape
        assert self.reinit_point_tracker_horizon >= self.reinit_horizon

        trajectories = torch.full((n_frames, n_masks, points_per_mask, 2), torch.nan, dtype=torch.float32)
        visibilities = torch.full((n_frames, n_masks, points_per_mask), False, dtype=torch.float32)
        scores_per_frame = torch.full((n_frames, n_masks), torch.nan, dtype=torch.float32)
        logits = torch.full((n_masks, n_frames, height, width), torch.nan, dtype=torch.float32)

        query_points_timestep = query_points[:, 0, 0].int()
        current_query_points = query_points.clone()
        for start_frame in range(query_points_timestep.min(), n_frames):
            # Prepare the points to track, if any
            end_frame = min(start_frame + self.reinit_horizon, n_frames)
            end_frame_tracker = min(start_frame + self.reinit_point_tracker_horizon, n_frames)
            current_timesteps = current_query_points[:, 0, 0].int()
            tracked_masks_indices = current_timesteps == start_frame
            if tracked_masks_indices.sum() == 0:
                continue

            # Track points
            query_points_i = current_query_points[tracked_masks_indices].clone()
            query_points_i[:, :, 0] -= start_frame
            assert (query_points_i[:, :, 0] == 0).all()

            # The SuperGlue point tracker performs keypoint matching and requires the query mask to be set
            if isinstance(self.point_tracker, SuperGluePointTracker):
                query_masks = self.extract_query_masks(images[start_frame:end_frame_tracker], query_points_i)
                assert self.point_tracker_mask_batch_size >= n_masks
                self.point_tracker.set_masks(query_masks)

            trajectories_i, visibilities_i = self._track_points(images[start_frame:end_frame_tracker], query_points_i)
            trajectories_i = trajectories_i[:self.reinit_horizon, :, :, :]
            visibilities_i = visibilities_i[:self.reinit_horizon, :, :]

            # Predict masks with Sam
            # TODO: Add positive trajectory points from mask indices that are not currently tracked
            #       so that they can be provided as negative points for other masks.
            _, logits_i, scores_per_frame_i = self._apply_sam_to_trajectories(
                images=images[start_frame:end_frame],
                trajectories=trajectories_i,
                visibilities=visibilities_i,
            )
            logits_i = logits_i.type(torch.float32)
            logits[tracked_masks_indices, start_frame:end_frame] = logits_i
            pred_masks_sam_i = logits_i > 0

            # Update the trajectories and visibilities
            trajectories[start_frame:end_frame, tracked_masks_indices] = trajectories_i
            visibilities[start_frame:end_frame, tracked_masks_indices] = visibilities_i
            scores_per_frame[start_frame:end_frame, tracked_masks_indices] = scores_per_frame_i

            # Update the current query points by extracting the new query points from the predicted Sam masks
            if end_frame == n_frames:
                continue
            height, width = pred_masks_sam_i.shape[2:]
            area_per_frame = pred_masks_sam_i[:, 1:, :, :].sum([2, 3]).float()
            area_per_frame[area_per_frame <= 25] = torch.nan
            if self.reinit_horizon // 4 < area_per_frame.shape[1]:
                area_per_frame[:, :self.reinit_horizon // 4] = torch.nan

            if self.reinit_variant == "reinit-on-horizon-and-sync-masks":
                # Fixed reinit timestep
                next_timestep = self.reinit_horizon - 1 - 1
                # Sync with other masks
                other_timesteps = current_timesteps[current_timesteps > start_frame]
                if len(other_timesteps) > 0:
                    first_other_higher_timestep = other_timesteps.min() - start_frame - 1
                    next_timestep = min(next_timestep, first_other_higher_timestep)
                query_points_timestep = torch.full((pred_masks_sam_i.shape[0],), next_timestep, dtype=torch.int64)
            elif self.reinit_variant == "reinit-at-median-of-area-diff":
                query_points_timestep = area_per_frame.nanmedian(dim=1).indices
            elif self.reinit_variant == "reinit-on-similar-mask-area":
                target_mask_area = pred_masks_sam_i[:, 0, :, :].sum([1, 2])
                area_diff = torch.abs(area_per_frame - target_mask_area[:, None])
                area_diff[area_diff.isnan()] = torch.inf
                query_points_timestep = area_diff.argmin(dim=1)
            elif self.reinit_variant == "reinit-on-similar-mask-area-and-sync-masks":
                # Finds the timestep which is the best for all masks so that mask syncing makes sense
                target_mask_area = pred_masks_sam_i[:, 0, :, :].sum([1, 2])
                area_diff = torch.abs(area_per_frame - target_mask_area[:, None])
                area_diff = area_diff / target_mask_area[:, None]  # normalize to [0, 1]
                area_diff[area_diff.isnan()] = 720  # set nan to a high value, in case one of the masks has only nans
                area_diff_per_frame = area_diff.sum(dim=0)
                # Prefer mask syncing, but not too much
                other_timesteps = current_timesteps[current_timesteps > start_frame]
                if len(other_timesteps) > 0:
                    first_other_higher_timestep = other_timesteps.min().item() - start_frame - 1
                    area_diff_per_frame[first_other_higher_timestep] -= 36
                next_timestep = area_diff_per_frame.argmin(dim=0)
                query_points_timestep = torch.full((pred_masks_sam_i.shape[0],), next_timestep, dtype=torch.int64)
            else:
                raise ValueError(f"Unknown reinit variant: {self.reinit_variant}")

            print(f"Horizon: {self.reinit_horizon}, Tracking horizon: {self.reinit_point_tracker_horizon}, "
                  f"    Next Timesteps: {query_points_timestep} / {self.reinit_horizon - 1 - 1}")

            # Masks that would be reinitialized at a zero mask, a very small mask, or a mask with a low SAM IoU score are invalid
            invalid_masks = area_per_frame[torch.arange(len(query_points_timestep)), query_points_timestep] <= 0

            # Extract new query points for valid masks
            if (~invalid_masks).sum() > 0:
                query_masks = pred_masks_sam_i[:, 1:, :, :][
                    torch.arange(len(query_points_timestep)), query_points_timestep]
                query_masks = query_masks.type(torch.float32)
                query_points_update = self.extract_query_points(
                    images=images[start_frame + 1:end_frame],
                    query_masks=query_masks[~invalid_masks],
                    query_points_timestep=query_points_timestep[~invalid_masks],
                )

                valid_tracked_masks = tracked_masks_indices.clone()
                valid_tracked_masks[tracked_masks_indices] = ~invalid_masks.to(tracked_masks_indices.device)
                current_query_points[valid_tracked_masks] = query_points_update.to(current_query_points.device)
                current_query_points[valid_tracked_masks, :, 0] += start_frame + 1

            # For invalid masks, set all future mask logits to -inf and the query points will be set to end of video
            if invalid_masks.sum() > 0:
                invalid_tracked_masks = tracked_masks_indices.clone()
                invalid_tracked_masks[tracked_masks_indices] = invalid_masks

                # TODO: Consider what to do in case the mask used to reinitialize is invalid.
                #       The mask might be invalid in case all points were invisible, out of the frame, etc.
                #       Currently, we will set all future mask logits to -inf and the query points to the end of video.
                current_query_points[invalid_tracked_masks, :, 0] = n_frames
                current_query_points[invalid_tracked_masks, :, 1:] = 0
                trajectories[end_frame:, invalid_tracked_masks] = -72
                visibilities[end_frame:, tracked_masks_indices] = PointVisibilityType.REINIT_FAILED.value
                logits[invalid_tracked_masks, end_frame:] = -float("inf")

        scores = scores_per_frame.nanmean(dim=1)

        return trajectories, visibilities, logits, scores, scores_per_frame

    def _track_points(self, rgbs, query_points):
        """
        Parameters
        ----------
        rgbs : torch.Tensor
            The frames of the video as a uint8 tensor of shape (num_frames, channels, height, width).
        query_points : torch.Tensor
            A float32 tensor representing the query points that define the query masks. The tensor has shape
            (num_masks, n_points_per_mask, 3) and contains the (t, x, y) values of the query points, where
            t is the timestep defining the query video frame and x, y are the coordinates of the query point.

        Returns
        -------
        trajectories : torch.Tensor
            A float32 tensor of shape (num_frames, num_masks, points_per_mask, 2) containing the predicted trajectories.
        visibilities : torch.Tensor
            A float32 tensor of shape (num_frames, num_masks, points_per_mask) containing the predicted visibilities.
        """
        num_masks, points_per_mask, _ = query_points.shape

        trajectories_pred, visibilities_pred = None, None
        for i in range(0, num_masks, self.point_tracker_mask_batch_size):
            batch_query_points = query_points[i:i + self.point_tracker_mask_batch_size]
            trajectories_pred_i, visibilities_pred_i = self.__track_points_inner(rgbs, batch_query_points)
            if trajectories_pred is None:
                trajectories_pred = trajectories_pred_i
                visibilities_pred = visibilities_pred_i
            else:
                trajectories_pred = torch.cat([trajectories_pred, trajectories_pred_i], dim=1)
                visibilities_pred = torch.cat([visibilities_pred, visibilities_pred_i], dim=1)

        return trajectories_pred, visibilities_pred

    def __track_points_inner(self, rgbs, query_points):
        # Query points should be flattened
        num_masks, points_per_mask, _ = query_points.shape
        query_points = query_points.reshape(num_masks * points_per_mask, 3)

        # Move tensors to the correct device
        rgbs = rgbs.to(self.device)
        query_points = query_points.to(self.device)

        # Add dummy batch dimension
        rgbs = rgbs.unsqueeze(0)
        query_points = query_points.unsqueeze(0)

        self.point_tracker.eval()
        with torch.no_grad():  # TODO: Check if this should be removed
            outputs = self.point_tracker.to(self.device).evaluate_batch(rgbs, query_points)
        trajectories_pred = outputs["trajectories_pred"].squeeze(0)
        visibilities_pred = outputs["visibilities_pred"].squeeze(0)

        def extract_patches_from_points(rgbs_lab: torch.Tensor, points_xy: torch.Tensor, patch_size: int):
            """
            Helper function to extract patches from points.
            """
            n_frames, _, h, w = rgbs_lab.shape
            n_frames, n_points_per_frame, _ = points_xy.shape
            assert rgbs_lab.shape == (n_frames, 3, h, w)
            assert points_xy.shape == (n_frames, n_points_per_frame, 2)

            # Prepare patch coordinates
            patch_template = torch.arange(-(patch_size // 2), patch_size // 2 + 1, device=points_xy.device)
            patch_template = torch.meshgrid(patch_template, patch_template)
            patch_template = torch.stack(patch_template, dim=-1).reshape(-1, 2)
            patches_xy = points_xy[:, :, None, :] + patch_template[None, None, :, :]
            shifted_point_xy = patches_xy + 0.5  # Shift to center of pixel
            normalized_point_xy = (shifted_point_xy / torch.tensor([w, h])[None, None, :]) * 2 - 1  # to [-1,1]

            # Extract patch features
            point_features = F.grid_sample(
                input=rgbs_lab[:, :, :, :],
                grid=normalized_point_xy[:, :, :, :],
                align_corners=False,
                mode="bilinear",
            ).permute(0, 2, 3, 1)

            return point_features

        def compute_patch_similarity(trajectory_patches_1, trajectory_patches_2):
            """
            Helper function to compute the patch similarity between two sets of patches.
            """
            # Merge channel and patch dimensions
            trajectory_patches_1 = trajectory_patches_1.flatten(start_dim=2, end_dim=3)
            trajectory_patches_2 = trajectory_patches_2.flatten(start_dim=2, end_dim=3)

            # Compute patch similarity
            diff = trajectory_patches_2[:, :, :] - trajectory_patches_1[:, :, :]
            patch_similarities = torch.exp(-torch.norm(diff, dim=-1) / (2 * self.patch_size ** 2))

            return patch_similarities

        # Remove dummy batch dimension
        rgbs = rgbs.squeeze(0)
        query_points = query_points.squeeze(0)
        visibilities_pred = visibilities_pred.float()

        # Use patch similarity to detect mistakes
        if self.use_patch_matching_filtering:
            rgbs_lab = color.rgb2lab(rgbs[:, [2, 1, 0], :, :].byte().permute(0, 2, 3, 1).cpu().numpy())
            rgbs_lab = torch.as_tensor(rgbs_lab, dtype=torch.float32).permute(0, 3, 1, 2)
            query_points_timestep = query_points[:, 0].long()
            query_points_xy = query_points[:, 1:].cpu()
            query_rgbs_lab = rgbs_lab[query_points_timestep, :, :, :]
            query_point_patches = extract_patches_from_points(query_rgbs_lab, query_points_xy[:, None, :],
                                                              self.patch_size)
            query_point_patches = query_point_patches.squeeze(1)
            trajectory_patches = extract_patches_from_points(rgbs_lab, trajectories_pred[:, :, :], self.patch_size)
            patch_similarities = compute_patch_similarity(query_point_patches[None, :, :, :], trajectory_patches)
            similar = patch_similarities > self.patch_similarity_threshold
            visibilities_pred[(visibilities_pred == 1) & ~similar] = PointVisibilityType.PATCH_NON_SIMILAR.value

        # Unflatten
        trajectories_pred = trajectories_pred.reshape(-1, num_masks, points_per_mask, 2)
        visibilities_pred = visibilities_pred.reshape(-1, num_masks, points_per_mask)
        query_points = query_points.reshape(num_masks, points_per_mask, 3)

        if self.use_patch_matching_filtering:
            # Post process visibilities to omit all points after a low patch similarity
            n_frames = trajectories_pred.shape[0]
            for mask_idx in range(num_masks):
                for point_idx in range(points_per_mask):
                    query_timestep = int(query_points[mask_idx, point_idx, 0].item())
                    for frame_idx in range(query_timestep + 1, n_frames):
                        if visibilities_pred[
                            frame_idx, mask_idx, point_idx] != PointVisibilityType.PATCH_NON_SIMILAR.value:
                            continue
                        visibilities_pred[frame_idx + 1:, mask_idx,
                        point_idx] = PointVisibilityType.REJECTED_AFTER_PATCH_WAS_NON_SIMILAR.value
                        break
                    for frame_idx in range(query_timestep - 1, -1, -1):
                        if visibilities_pred[
                            frame_idx, mask_idx, point_idx] != PointVisibilityType.PATCH_NON_SIMILAR.value:
                            continue
                        visibilities_pred[:frame_idx:, mask_idx,
                        point_idx] = PointVisibilityType.REJECTED_AFTER_PATCH_WAS_NON_SIMILAR.value
                        break

        # Denote out-of-frame points for visualisation debugging purposes
        # Define "out-of-frame" as points that are very close to or outside the image border
        h, w = rgbs.shape[-2:]
        visibilities_pred[trajectories_pred[:, :, :, 0] / w < 0.01] = PointVisibilityType.OUTSIDE_FRAME.value
        visibilities_pred[trajectories_pred[:, :, :, 1] / h < 0.01] = PointVisibilityType.OUTSIDE_FRAME.value
        visibilities_pred[trajectories_pred[:, :, :, 0] / w > 0.99] = PointVisibilityType.OUTSIDE_FRAME.value
        visibilities_pred[trajectories_pred[:, :, :, 1] / h > 0.99] = PointVisibilityType.OUTSIDE_FRAME.value

        return trajectories_pred, visibilities_pred

    def _apply_sam_to_trajectories(self, images, trajectories, visibilities):
        """
        Applies the Sam model to the predicted trajectories to obtain the final mask predictions.
        Sam is applied to each frame independently, and the predicted masks are then combined.
        Sam is only applied to the points that are visible in the frame, if any. If no points are
        visible, the mask is set to zeros.

        Parameters
        ----------
        images : torch.Tensor
            The frames of the video as a uint8 tensor of shape (num_frames, channels, height, width).
        trajectories : torch.Tensor
            A float32 tensor of shape (num_frames, num_masks, points_per_mask, 2) containing the predicted trajectories.
        visibilities : torch.Tensor
            A float32 tensor of shape (num_frames, num_masks, points_per_mask) containing the predicted visibilities.

        Returns
        -------
        pred_scores : np.ndarray
            An array of shape (num_masks,) containing the mean predicted mask scores for frames with visible points,
            for each mask.
        mask_logits : torch.Tensor
            A float32 tensor of shape (num_masks, num_frames, height, width) containing the mask predictions in the form of logits.
        mask_scores_per_frame_tensor : torch.Tensor
            A float32 tensor of shape (num_frames, num_masks) containing the predicted scores of the predicted masks
            for each frame.
        """
        n_frames, channels, height, width = images.shape
        _, n_masks, points_per_mask, _ = trajectories.shape
        assert trajectories.shape == (n_frames, n_masks, points_per_mask, 2)
        assert visibilities.shape == (n_frames, n_masks, points_per_mask)

        def prepare_points(frame_idx, mask_idx):
            """
            Helper function to prepare the point inputs for Sam.
            """
            point_coords = trajectories[frame_idx, mask_idx, :, :]
            point_labels = np.ones((len(point_coords)), dtype=int)  # assume all are foreground points
            if self.negative_points_per_mask > 0:
                point_labels[self.positive_points_per_mask:] = 0  # tail points are negative points
            visible_point_coords = point_coords[visibilities[frame_idx, mask_idx, :] == 1, :].cpu().numpy()
            visible_point_labels = point_labels[visibilities[frame_idx, mask_idx, :].cpu().numpy() == 1]

            if n_masks > 1 and self.add_other_objects_positive_points_as_negative_points:
                other_objects_positive_point_coords = torch.cat([
                    trajectories[frame_idx, other_mask_idx, :self.positive_points_per_mask, :][
                    visibilities[frame_idx, other_mask_idx, :self.positive_points_per_mask] == 1, :]
                    for other_mask_idx in range(n_masks) if other_mask_idx != mask_idx
                ], dim=0).cpu().numpy()
                if self.max_other_objects_positive_points is not None and len(
                        other_objects_positive_point_coords) > self.max_other_objects_positive_points:
                    n = len(other_objects_positive_point_coords)
                    indices = np.random.choice(n, self.max_other_objects_positive_points, replace=False)
                    other_objects_positive_point_coords = other_objects_positive_point_coords[indices, :]
                other_objects_positive_point_labels = np.zeros((len(other_objects_positive_point_coords)), dtype=int)
                visible_point_coords = np.concatenate(
                    [visible_point_coords, other_objects_positive_point_coords],
                    axis=0,
                )
                visible_point_labels = np.concatenate(
                    [visible_point_labels, other_objects_positive_point_labels],
                    axis=0,
                )

            return visible_point_coords, visible_point_labels

        def predict_mask(visible_point_coords, visible_point_labels):
            """
            Helper function that predicts the mask for a single frame and a single mask.
            """

            # Mask is empty if all points are invisible
            if len(visible_point_coords) == 0:
                return np.full((height, width), -float('inf'), dtype=np.float64), None

            # Prepare for SAM predictor: resize coordinates to the image size and convert to torch tensors
            visible_point_coords = torch.as_tensor(
                self.sam_predictor.transform.apply_coords(visible_point_coords, self.sam_predictor.original_size),
                dtype=torch.float,
                device=self.device,
            )
            visible_point_labels = torch.as_tensor(
                visible_point_labels,
                dtype=torch.int,
                device=self.device,
            )

            # Predict the mask by passing the visible points to Sam
            if self.negative_points_per_mask == 0:
                mask_frame_logits, iou_prediction_scores, low_res_masks = self.sam_predictor.predict_torch(
                    point_coords=visible_point_coords[None, :, :],
                    point_labels=visible_point_labels[None, :],
                    boxes=None,
                    mask_input=None,
                    multimask_output=False,
                    return_logits=True,
                )
            else:
                _, _, low_res_masks = self.sam_predictor.predict_torch(
                    point_coords=visible_point_coords[visible_point_labels == 1][None, :, :],
                    point_labels=visible_point_labels[visible_point_labels == 1][None, :],
                    boxes=None,
                    mask_input=None,
                    multimask_output=False,
                    return_logits=True,
                )
                mask_frame_logits, iou_prediction_scores, low_res_masks = self.sam_predictor.predict_torch(
                    point_coords=visible_point_coords[None, :, :],
                    point_labels=visible_point_labels[None, :],
                    boxes=None,
                    mask_input=low_res_masks,
                    multimask_output=False,
                    return_logits=True,
                )

            if self.iterative_refinement_iterations > 0:
                for refinement_iteration in range(self.iterative_refinement_iterations):
                    m = mask_frame_logits[0, 0, :, :] > 0
                    if m.sum() < 2:
                        break
                    yx = m.nonzero()
                    refinement_box = torch.tensor([
                        yx[:, 1].min(),  # xmin
                        yx[:, 0].min(),  # ymin
                        yx[:, 1].max(),  # xmax
                        yx[:, 0].max(),  # ymax
                    ], dtype=torch.float, device=self.device)
                    mask_frame_logits, iou_prediction_scores, low_res_masks = self.sam_predictor.predict_torch(
                        point_coords=visible_point_coords[None, :, :],
                        point_labels=visible_point_labels[None, :],
                        boxes=refinement_box[None, None, :],
                        mask_input=low_res_masks,
                        multimask_output=False,
                        return_logits=True,
                    )

            mask_frame_logits = mask_frame_logits[0, 0, :, :].cpu().numpy()
            iou_prediction_score = iou_prediction_scores[0, 0].cpu().numpy()

            # Mask is empty if SAM's IoU score is too low
            if iou_prediction_score < self.sam_iou_threshold:
                return np.full((height, width), -float('inf'), dtype=np.float64), iou_prediction_score

            return mask_frame_logits, iou_prediction_score

        # Initialize arrays to hold mask logits and scores
        masks_logits = np.full((n_masks, n_frames, height, width), -float('inf'))
        mask_scores_per_frame = np.full((n_frames, n_masks), -float('inf'))

        # Array to store the sum and count of scores for each mask for calculating the mean
        mask_scores_sum = np.zeros(n_masks)
        mask_scores_count = np.zeros(n_masks)

        # Go through all frames and masks and predict the mask logits and IoU scores
        for frame_idx in tqdm(range(n_frames)):
            self.sam_predictor.set_image(images[frame_idx].permute(1, 2, 0).cpu().numpy())
            for mask_idx in range(n_masks):
                visible_point_coords, visible_point_labels = prepare_points(frame_idx, mask_idx)
                mask_frame_logits, iou_prediction_score = predict_mask(visible_point_coords, visible_point_labels)
                masks_logits[mask_idx, frame_idx] = mask_frame_logits

                if iou_prediction_score is not None:
                    mask_scores_per_frame[frame_idx, mask_idx] = iou_prediction_score
                    mask_scores_sum[mask_idx] += iou_prediction_score
                    mask_scores_count[mask_idx] += 1

        # Compute the mask score as the mean of sam's iou_prediction scores for frames with visible points
        pred_scores = mask_scores_sum / np.where(mask_scores_count != 0, mask_scores_count, 1)

        mask_logits = torch.from_numpy(masks_logits).float()
        mask_scores_per_frame_tensor = torch.from_numpy(mask_scores_per_frame).float()

        return pred_scores, mask_logits, mask_scores_per_frame_tensor
