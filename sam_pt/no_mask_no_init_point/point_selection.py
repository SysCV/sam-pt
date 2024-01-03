import torch
import numpy as np
import cv2

def ransac_point_selector(trajectories, visibilities):
    step = 15

    assert trajectories.shape[0] > step, "Insufficient frames for comparison."
    assert trajectories.shape[1] == 1, "Function is designed for one mask only."

    valid_points_mask = (visibilities[0, 0] == 1) & (visibilities[step, 0] == 1)
    trajectories_first_two_frames = trajectories[[0, step], 0, :]

    filtered_points_frame0 = trajectories_first_two_frames[0][valid_points_mask]
    filtered_points_frame1 = trajectories_first_two_frames[1][valid_points_mask]

    X = filtered_points_frame0.numpy()
    y = filtered_points_frame1.numpy()

    estimated_affine_transform, inlier_mask = cv2.estimateAffinePartial2D(X, y, method=cv2.RANSAC)
    inlier_mask = inlier_mask.ravel().astype(bool)

    transformed_points = np.dot(X, estimated_affine_transform[:2, :2].T) + estimated_affine_transform[:, 2]
    residuals = np.linalg.norm(transformed_points - y, axis=1)

    inliers = X[inlier_mask]

    outliers = X[~inlier_mask]

    assert outliers.shape[0] > 0, "no foreground detected"

    outlier_residuals = residuals[~inlier_mask]
    sorted_indices = np.argsort(outlier_residuals)[::-1]
    sorted_outliers = outliers[sorted_indices]

    negative_points = torch.from_numpy(sorted_outliers).to(trajectories.dtype)
    negative_points = torch.cat((torch.zeros(negative_points.shape[0], 1), negative_points), dim=1).reshape(1, -1, 3)

    positive_points = torch.from_numpy(inliers).to(trajectories.dtype)
    positive_points = torch.cat((torch.zeros(positive_points.shape[0], 1), positive_points), dim=1).reshape(1, -1, 3)

    return positive_points, negative_points