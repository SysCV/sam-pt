import torch
from abc import ABC, abstractmethod
from torch import nn
from typing import Tuple


class PointTracker(ABC, nn.Module):
    """
    Abstract class for point trackers.

    Methods
    -------
    forward(rgbs, query_points)
        Performs a forward pass through the model and returns the predicted trajectories and visibilities.
    evaluate_batch(rgbs, query_points, trajectories_gt=None, visibilities_gt=None)
        Evaluates a batch of videos and returns the results.
    unpack_results(packed_results, batch_idx)
        Unpacks the results for all point and all videos in the batch.
    """

    @abstractmethod
    def forward(self, rgbs, query_points) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the model and returns the predicted trajectories and visibilities.

        Parameters
        ----------
        rgbs : torch.Tensor
            A tensor of shape (batch_size, n_frames, channels, height, width)
            containing the RGB images in uint8 [0-255] format.
        query_points : torch.Tensor
            A tensor of shape (batch_size, n_points, 3) containing the query points,
            each point being (t, x, y).

        Returns
        -------
        tuple of two torch.Tensor
            Returns a tuple of (trajectories, visibilities).
            - `trajectories`: Predicted point trajectories with shape (batch_size, n_frames, n_points, 2), where each
                              trajectory represents a series of (x, y) coordinates in the video for a specific point.
            - `visibilities`: Predicted point visibilities with shape (batch_size, n_frames, n_points), where each
                              visibility represents the likelihood of a point being visible in the corresponding frame
                              of the video.
        """
        pass

    def evaluate_batch(self, rgbs, query_points, trajectories_gt=None, visibilities_gt=None):
        """
        Evaluates a batch of data and returns the results.

        Parameters
        ----------
        rgbs : torch.Tensor
            A tensor of shape (batch_size, n_frames, channels, height, width)
            containing the RGB images in uint8 [0-255] format.
        query_points : torch.Tensor
            A tensor of shape (batch_size, n_points, 3) containing the query points,
            each point being (t, x, y).
        trajectories_gt : torch.Tensor, optional
            A 4D tensor representing the ground-truth trajectory. Its shape is (batch_size, n_frames, n_points, 2).
        visibilities_gt : torch.Tensor, optional
            A 3D tensor representing the ground-truth visibilities. Its shape is (batch_size, n_frames, n_points).

        Returns
        -------
        dict
            A dictionary containing the results.
        """
        trajectories_pred, visibilities_pred = self.forward(rgbs, query_points)
        batch_size = rgbs.shape[0]
        n_frames = rgbs.shape[1]
        n_points = query_points.shape[1]
        assert trajectories_pred.shape == (batch_size, n_frames, n_points, 2)

        results = {
            "trajectories_pred": trajectories_pred.detach().clone().cpu(),
            "visibilities_pred": visibilities_pred.detach().clone().cpu(),
            "query_points": query_points.detach().clone().cpu(),
            "trajectories_gt": trajectories_gt.detach().clone().cpu() if trajectories_gt is not None else None,
            "visibilities_gt": visibilities_gt.detach().clone().cpu() if visibilities_gt is not None else None,
        }

        return results

    @classmethod
    def unpack_results(cls, packed_results, batch_idx):
        """
        Unpacks the results for all point and all videos in the batch.

        Parameters
        ----------
        packed_results : dict
            The dictionary containing the packed results, for all videos in the batch and all points in the video.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        list
            A list of dictionaries, each containing the unpacked results for a data point.
        """
        unpacked_results_list = []
        for b in range(packed_results["trajectories_pred"].shape[0]):
            for n in range(packed_results["trajectories_pred"].shape[2]):
                result = {
                    "idx": f"{batch_idx}_{b}_{n}",
                    "iter": batch_idx,
                    "video_idx": b,
                    "point_idx_in_video": n,
                    "query_point": packed_results["query_points"][b, n, :],
                    "trajectory_pred": packed_results["trajectories_pred"][b, :, n, :],
                    "visibility_pred": packed_results["visibilities_pred"][b, :, n],
                }
                if packed_results["trajectories_gt"] is not None:
                    result["trajectory_gt"] = packed_results["trajectories_gt"][b, :, n, :]
                    result["visibility_gt"] = packed_results["visibilities_gt"][b, :, n]
                unpacked_results_list += [result]
        return unpacked_results_list
