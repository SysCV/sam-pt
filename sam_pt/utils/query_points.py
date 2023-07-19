"""
This module contains a collection of functions for extracting specific types of points from a binary mask.

The functions are typically used in the context of video object segmentation where a ground truth mask is provided
for the first frame and the goal is to track the object in the subsequent frames. The points extracted from the
ground truth mask are used to track the object in the subsequent frames.

Functions:
    - extract_random_mask_points: Extracts random points from a given mask.
    - extract_kmedoid_points: Extracts K-Medoids from a given mask.
    - extract_corner_points: Extracts corner points from a given mask using the Shi-Tomasi corner detection method.
    - erode_mask_proportional_to_its_furthest_points_distance: Erodes a given mask proportionally to the furthest
        distance between two points in the mask.
    - extract_mixed_points: Extracts a mixture of K-Medoid, Shi-Tomasi, and random points from a list of masks.

Notes:
    The functions in this module typically expect masks to be binary torch tensors with float32 data type
    and values in {0, 1}. The points extracted by these functions are returned as torch tensors with float32
    data type, with each point represented as an (x, y) pair.

"""
import cv2
import numpy as np
import torch
from sklearn_extra.cluster import KMedoids
from typing import List


def extract_random_mask_points(mask: torch.Tensor, n_points_to_select: int) -> torch.Tensor:
    """
    Randomly select a specified number of points from the mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor with shape (height, width) of dtype float32 with values in {0, 1}.
    n_points_to_select : int
        The number of points to select from the mask.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_points_to_select, 2) containing the selected points. The dtype of the
        tensor is float32.
    """
    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((n_points_to_select, 2))

    mask_pixels = mask.nonzero().float()
    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        selected_points = mask_pixels[torch.randperm(len(mask_pixels))[:n_points_to_select]]

    selected_points = selected_points.flip(1)  # Change from (y, x) to (x, y)
    assert selected_points.shape == (n_points_to_select, 2)
    return selected_points


def extract_kmedoid_points(mask, n_points_to_select, subsample_size=1800):
    """
    Randomly select the specified number of points from the mask using K-Medoids.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor with shape (height, width) of dtype float32 with values in {0, 1}.
    n_points_to_select : int
        Number of points to select from the mask.
    subsample_size : int, optional
        Size of subsample to use for K-Medoids, by default 1800.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_points_to_select, 2) containing the selected points. The dtype of the
        tensor is float32.
    """
    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_kmedoid_points")
        return torch.zeros((n_points_to_select, 2))

    mask_pixels = mask.nonzero().float()

    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        # Sample N points from the largest cluster by performing K-Medoids with K=N
        mask_pixels = mask_pixels[torch.randperm(len(mask_pixels))[:subsample_size]]
        selected_points = KMedoids(n_clusters=n_points_to_select).fit(mask_pixels).cluster_centers_
        selected_points = torch.from_numpy(selected_points).type(torch.float32)

    # (y, x) -> (x, y)
    selected_points = selected_points.flip(1)

    assert selected_points.shape == (n_points_to_select, 2)
    return selected_points


def extract_corner_points(image, mask, n_points_to_select, kmedoid_subsample_size=2000):
    """
    Select a specified number of points from the mask using a corner detection algorithm. Erosion
    is applied on the mask at various levels if necessary, before performing corner detection,
    as to avoid selecting points on the edges of the mask.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor of shape (channels, height, width) and in uint8 [0-255] format.
    mask : torch.Tensor
        Binary mask tensor with shape (height, width) of dtype float32 with values in {0, 1}.
    n_points_to_select : int
        Number of points to select from the mask.
    kmedoid_subsample_size : int, optional
        Size of subsample to use for K-Medoids, by default 2000.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_points_to_select, 2) and dtype float32 containing the selected points.
        Points are in (x, y) format.
    """
    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_corner_points")
        return torch.zeros((n_points_to_select, 2))

    image = image.permute(1, 2, 0).cpu().numpy()
    mask_eroded = erode_mask_proportional_to_its_furthest_points_distance(mask, erosion_percentage=0.06)
    if mask_eroded.sum() < 10:
        mask_eroded = erode_mask_proportional_to_its_furthest_points_distance(mask, erosion_percentage=0.02)
    if mask_eroded.sum() < 10:
        mask_eroded = erode_mask_proportional_to_its_furthest_points_distance(mask, erosion_percentage=0.01)
    if mask_eroded.sum() < 10:
        mask_eroded = mask
    mask_pixels = mask_eroded.nonzero().float()
    mask_diameter = torch.norm(mask_pixels.max(0)[0] - mask_pixels.min(0)[0]).item()

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corner_points = cv2.goodFeaturesToTrack(
        image=gray_image,
        maxCorners=n_points_to_select,
        qualityLevel=0.001,
        minDistance=mask_diameter / n_points_to_select,
        mask=mask_eroded.cpu().numpy().astype(np.uint8),
        blockSize=3,
        gradientSize=3,
    )
    if corner_points is None:
        corner_points = np.empty((0, 1, 2))
    corner_points = torch.from_numpy(corner_points).type(torch.float32).squeeze(1)

    if len(corner_points) < n_points_to_select:
        # Replace the missing points with K-medoid points
        n_missing_points = n_points_to_select - corner_points.shape[0]
        kmedoid_points = extract_kmedoid_points(mask, n_missing_points, subsample_size=kmedoid_subsample_size)

        corner_points = torch.cat((corner_points, kmedoid_points), dim=0)

    assert corner_points.shape == (n_points_to_select, 2)
    return corner_points


def erode_mask_proportional_to_its_furthest_points_distance(mask: torch.Tensor,
                                                            erosion_percentage: float) -> torch.Tensor:
    """
    Erode the mask by a percentage of its diameter.

    Erode the mask by the specified percentage of its diameter.
    The diameter is computed as the distance between the two
    points that are the farthest from each other on the mask.
    The erosion is performed using a square kernel.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor with shape (height, width) of dtype float32 with values in {0, 1}.
    erosion_percentage : float
        Percentage of the mask diameter to erode the mask by.

    Returns
    -------
    mask : torch.Tensor
        Eroded mask of shape (height, width).
    """
    mask_pixels = mask.nonzero().float()
    mask_diameter = torch.norm(mask_pixels.max(0)[0] - mask_pixels.min(0)[0]).item()
    erosion_size = int(mask_diameter * erosion_percentage)

    mask_for_cv = mask.cpu().numpy().astype(np.uint8)
    eroded_mask_for_cv = cv2.erode(mask_for_cv, np.ones((erosion_size, erosion_size), np.uint8), iterations=1)
    mask = torch.from_numpy(eroded_mask_for_cv).type(mask.dtype).to(mask.device)
    return mask


def extract_mixed_points(query_masks: List[torch.Tensor], query_points_timestep: torch.Tensor,
                         images: torch.Tensor, n_points: int) -> List[torch.Tensor]:
    """
    Extracts a mixed collection of points (k-medoid, Shi-Tomasi, and random) from a list of query masks.

    Parameters
    ----------
    query_masks : list of torch.Tensor
        List of masks from which points are extracted.
        Masks are float32 tensors of shape (height, width) with values in {0, 1}.
    query_points_timestep : torch.Tensor
        Corresponding timesteps for each query point, of shape (n_masks,) and dtype torch.float32.
    images : torch.Tensor
        A tensor representing all frames of a video, with shape (n_frames, channels, height, width)
        and dtype uint8 in the range [0-255].
    n_points : int
        Total number of points to extract.

    Returns
    -------
    mixed_points_xy : list of torch.Tensor
        List of tensors of extracted points for each mask. Each tensor has shape (n_points, 2) and
        dtype float32. Points are in (x, y) format.
    """
    n_kmedoid, n_shi_tomasi = n_points // 4, n_points // 3
    n_random = n_points - n_kmedoid - n_shi_tomasi
    mixed_points_xy_list = []
    if n_kmedoid > 0:
        mixed_points_xy_list += [[extract_kmedoid_points(qm, n_kmedoid) for qm in query_masks]]
    if n_shi_tomasi > 0:
        mixed_points_xy_list += [[
            extract_corner_points(images[int(t.item()), :, :, :], query_mask, n_shi_tomasi)
            for query_mask, t in zip(query_masks, query_points_timestep)
        ]]
    if n_random > 0:
        mixed_points_xy_list += [[extract_random_mask_points(qm, n_random) for qm in query_masks]]
    if len(mixed_points_xy_list) == 1:
        mixed_points_xy = mixed_points_xy_list[0]
    else:
        mixed_points_xy = [torch.cat(x, dim=0) for x in zip(*mixed_points_xy_list)]
    return mixed_points_xy
