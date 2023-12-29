import torch
from sklearn.linear_model import RANSACRegressor


def ransac_point_selector(trajectories, visibilities):
    assert trajectories.shape[0] > 24, "1s fixed to compare."
    assert trajectories.shape[1] == 1, "works only for one mask."

    valid_points_mask = (visibilities[0, 0] == 1) & (visibilities[24, 0] == 1)
    trajectories_first_two_frames = trajectories[[0, 24], 0, :]

    filtered_points_frame0 = trajectories_first_two_frames[0][valid_points_mask]
    filtered_points_frame1 = trajectories_first_two_frames[1][valid_points_mask]

    X = filtered_points_frame0.numpy()
    y = filtered_points_frame1.numpy()

    ransac = RANSACRegressor()
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_

    negative_points = filtered_points_frame0[inlier_mask]
    positive_points = filtered_points_frame0[~inlier_mask]

    # remake them in the good format
    negative_points = torch.cat((torch.zeros(negative_points.shape[0], 1), negative_points), dim=1).reshape(1, -1, 3)
    positive_points = torch.cat((torch.zeros(positive_points.shape[0], 1), positive_points), dim=1).reshape(1, -1, 3)

    return positive_points, negative_points
        