import torch


def random_point_initialization(n_masks, timesteps, n_init_point_per_mask, width, height):
    random_points = torch.empty((n_masks, n_init_point_per_mask, 3))

    for i in range(n_masks): # timesteps of the masks
        random_points[i, :, 0] = timesteps[i]

    # TODO: ensure that the points are not the same
    random_points[:, :, 1] = torch.randint(0, width, (n_masks, n_init_point_per_mask))
    random_points[:, :, 2] = torch.randint(0, height, (n_masks, n_init_point_per_mask))

    return random_points