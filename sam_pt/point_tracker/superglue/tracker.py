import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from typing import Union, Tuple, Dict

from sam_pt.point_tracker import PointTracker
from .models.matching import Matching
from .models.utils import (process_resize)


class SuperGluePointTracker(PointTracker):
    """
    The SuperGluePointTracker class performs point tracking by using the SuperGlue feature matching algorithm from
    https://arxiv.org/abs/1911.11763. Specifically, SuperGlue is applied independently between the first and each
    subsequent video frame so that keypoints in the first frame are matched to the keypoints in each subsequent
    frame. The keypoints in the first frame must be inside a reference mask and can differ from frame to frame. The
    keypoints in each subsequent frame are chosen as the top-k keypoint matches with the highest confidence score.
    Since the matches are computed independently for each frame, the trajectories are not consistent across frames.
    It's important to note that this point tracker uniquely necessitates the setting of a reference mask before
    invoking the forward() function.
    """

    def __init__(self, positive_points_per_mask: int, negative_points_per_mask: int,
                 resize: Union[Tuple[int], Tuple[int, int]], matching_config: Dict):
        """
        Parameters
        ----------
        positive_points_per_mask : int
            The number of positive points per mask.
        negative_points_per_mask : int
            The number of negative points per mask.
        resize : tuple
            A tuple of integers containing the dimensions to which the images will be resized.
        matching_config : dict
            A dictionary containing the configurations for the SuperPoint and SuperGlue models.
        """
        super().__init__()

        self.positive_points_per_mask = positive_points_per_mask
        self.negative_points_per_mask = negative_points_per_mask

        # Prepare resizing
        self.resize = resize
        if len(self.resize) == 2 and self.resize[1] == -1:
            self.resize = self.resize[0:1]
        if len(self.resize) == 2:
            print('SuperGluePointTracker: Will resize to {}x{} (WxH)'.format(
                self.resize[0], self.resize[1]))
        elif len(self.resize) == 1 and self.resize[0] > 0:
            print('SuperGluePointTracker: Will resize max dimension to {}'.format(self.resize[0]))
        elif len(self.resize) == 1:
            print('SuperGluePointTracker: Will not resize images')
        else:
            raise ValueError('SuperGluePointTracker: Cannot specify more than two integers for `resize`')

        # Load the SuperPoint and SuperGlue models.
        self.matching_config = matching_config
        self.matching = Matching(self.matching_config).eval()

        self.masks = None

    def set_masks(self, masks):
        """
        Sets the reference masks used for tracking.

        Parameters
        ----------
        masks : np.array
            The binary reference masks to be used for tracking, provided as a float32 tensor
            of shape (n_masks, height, width) and with values in {0, 1}.
        """
        n_masks, height, width = masks.shape
        self.masks = masks

    def forward(self, rgbs, query_points, summary_writer=None):
        assert self.masks is not None, "Masks must be set before calling forward() for SuperGluePointTracker"
        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]
        n_points_per_mask = self.positive_points_per_mask + self.negative_points_per_mask
        n_masks = self.masks.shape[0]
        assert n_points_per_mask * n_masks == n_points
        if batch_size != 1:
            raise NotImplementedError("Batch size > 1 is not supported for SuperGluePointTracker yet")

        # Convert the torch rgbs images to grayscale
        rgbs = F.rgb_to_grayscale(rgbs)

        # Resize the images if necessary
        new_height, new_width = height, width
        if self.resize[0] > 0:
            new_width, new_height = process_resize(width, height, self.resize)
            rgbs = F.resize(rgbs, (new_width, new_height), interpolation=InterpolationMode.BILINEAR, antialias=True)
            raise NotImplementedError("Resizing not tested yet. Note that interpolation of PIL images and tensors "
                                      "is slightly different, because PIL applies antialiasing. This may lead to "
                                      "significant differences in the performance of a network. Therefore, it is "
                                      "preferable to train and serve a model with the same input types.")

        trajectories = torch.zeros(n_frames, n_masks, n_points_per_mask, 2)
        visibilities = torch.zeros(n_frames, n_masks, n_points_per_mask)

        # Dummy values for the first frame as it is the reference frame
        # We will take different points from the reference frame when matching other frames,
        # depending on what keypoint matches we find
        trajectories[0, :, :, :] = query_points[:, :, 1:].reshape(n_masks, n_points_per_mask, 2)
        # Take the first frame as the reference frame, since we assume to have the ground truth mask passed for it
        reference_image = rgbs[0, 0, 0, :, :] / 255

        # Loop over all other frames, find matching keypoints and update the trajectories
        kpts0, scores0, descriptors0 = None, None, None
        for i in range(1, n_frames):
            target_image = rgbs[0, i, 0, :, :].squeeze(1) / 255

            # Perform the matching
            matching_input_data = {}
            matching_input_data['image0'] = reference_image[None, None, ...]
            matching_input_data['image1'] = target_image[None, None, ...]
            if kpts0 is not None:
                matching_input_data['keypoints0'] = [torch.from_numpy(kpts0).to(rgbs.device)]
                matching_input_data['scores0'] = [torch.from_numpy(scores0).to(rgbs.device)]
                matching_input_data['descriptors0'] = [torch.from_numpy(descriptors0).to(rgbs.device)]
            pred = self.matching(matching_input_data)
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            if kpts0 is None:
                kpts0 = pred['keypoints0']
                scores0 = pred['scores0']
                descriptors0 = pred['descriptors0']
            kpts1 = pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            for mask_idx in range(n_masks):
                mask = self.masks[mask_idx, :, :]
                mask = F.resize(mask[None, None, ...], (height, width), interpolation=InterpolationMode.NEAREST)
                mask = mask.squeeze(0).squeeze(0)
                mask = mask > 0.5
                mask = mask.cpu().numpy()

                # Positive points: Keep only the matched points that are inside the mask
                mkpts0_positive = mkpts0[mask[mkpts0[:, 1].astype(int), mkpts0[:, 0].astype(int)]]
                mkpts1_positive = mkpts1[mask[mkpts1[:, 1].astype(int), mkpts1[:, 0].astype(int)]]
                mconf_positive = mconf[mask[mkpts0[:, 1].astype(int), mkpts0[:, 0].astype(int)]]

                # Negative points: Keep only the matched points that are outside the mask
                mkpts0_negative = mkpts0[~mask[mkpts0[:, 1].astype(int), mkpts0[:, 0].astype(int)]]
                mkpts1_negative = mkpts1[~mask[mkpts1[:, 1].astype(int), mkpts1[:, 0].astype(int)]]
                mconf_negative = mconf[~mask[mkpts0[:, 1].astype(int), mkpts0[:, 0].astype(int)]]

                # Randomly take the required number of points from the positive and negative points
                positive_points_index = np.random.choice(
                    a=len(mkpts1_positive),
                    size=min(len(mkpts1_positive), self.positive_points_per_mask),
                )
                negative_points_index = np.random.choice(
                    a=len(mkpts1_negative),
                    size=min(len(mkpts1_negative), self.negative_points_per_mask),
                )

                positive_points = mkpts1_positive[positive_points_index]
                negative_points = mkpts1_negative[negative_points_index]

                positive_points_visibility = torch.ones(self.positive_points_per_mask)
                negative_points_visibility = torch.ones(self.negative_points_per_mask)

                # If there are not enough points, pad with (-1, -1) points
                if len(positive_points) < self.positive_points_per_mask:
                    positive_points_visibility[len(positive_points):] = 0
                    positive_points = np.concatenate([
                        positive_points,
                        np.ones((self.positive_points_per_mask - len(positive_points), 2)) * -1,
                    ], axis=0)
                if len(negative_points) < self.negative_points_per_mask:
                    negative_points_visibility[len(negative_points):] = 0
                    negative_points = np.concatenate([
                        negative_points,
                        np.ones((self.negative_points_per_mask - len(negative_points), 2)) * -1,
                    ], axis=0)

                trajectories[i, mask_idx, :self.positive_points_per_mask, :] = torch.from_numpy(positive_points)
                trajectories[i, mask_idx, self.positive_points_per_mask:, :] = torch.from_numpy(negative_points)
                visibilities[i, mask_idx, :] = torch.cat([positive_points_visibility, negative_points_visibility])

        # Reset mask since it has been used
        self.masks = None

        # Merge mask and points dimensions
        trajectories = trajectories.reshape(n_frames, n_masks * n_points_per_mask, 2)
        visibilities = visibilities.reshape(n_frames, n_masks * n_points_per_mask)

        # Resize trajectories to the original image size
        trajectories[:, :, 0] = trajectories[:, :, 0] * width / new_width
        trajectories[:, :, 1] = trajectories[:, :, 1] * height / new_height

        # Add the dummy batch dimension
        trajectories = trajectories.unsqueeze(0)
        visibilities = visibilities.unsqueeze(0)

        return trajectories, visibilities
