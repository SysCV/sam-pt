"""
Demo program for TAPNet, to make sure that pytorch+jax has been set up correctly.
The following snippet should run without error, and ideally use GPU/TPU to be fast when benchmarking.

Example usage:
```
python -m sam_pt.point_tracker.tapnet.demo
```
"""
import time

import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from torch.nn import functional as F

from demo.demo import load_demo_data
from . import tapnet_model
from .configs.tapnet_config import get_config

if __name__ == '__main__':
    # 1. Prepare config
    config = get_config()
    checkpoint_dir = "./models/tapnet_ckpts/open_source_ckpt/"
    # Keep TF off the GPU; otherwise it hogs all the memory and leaves none for JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.experimental.set_visible_devices([], 'TPU')

    # 2. Prepare model
    checkpoint = np.load(checkpoint_dir + "checkpoint_wo_optstate.npy", allow_pickle=True).item()
    params, state = checkpoint["params"], checkpoint["state"]
    tapnet_model_kwargs = config.experiment_kwargs.config.shared_modules["tapnet_model_kwargs"]


    def forward(rgbs, query_points):
        tapnet = tapnet_model.TAPNet(**tapnet_model_kwargs)
        outputs = tapnet(
            video=rgbs[None, ...],
            query_points=query_points[None, ...],
            query_chunk_size=16,
            get_query_feats=True,
            is_training=False,
        )
        return outputs


    transform = hk.transform_with_state(forward)


    def f(rgbs_tapnet, query_points_tapnet):
        rng = jax.random.PRNGKey(72)
        outputs, _ = transform.apply(params, state, rng, rgbs_tapnet, query_points_tapnet)
        return outputs


    jitted_f = jax.jit(f)

    # 3. Prepare data
    rgbs, _, query_points = load_demo_data(
        frames_path="data/demo_data/bees",
        query_points_path="data/demo_data/query_points__bees.txt",
    )
    original_hw = rgbs.shape[-2:]
    tapnet_input_hw = (
        config.experiment_kwargs.config.inference.resize_height, config.experiment_kwargs.config.inference.resize_width)
    rescale_factor_hw = torch.tensor(tapnet_input_hw) / torch.tensor(original_hw)
    rgbs_tapnet = F.interpolate(rgbs / 255, tapnet_input_hw, mode="bilinear", align_corners=False, antialias=True)
    rgbs_tapnet = rgbs_tapnet.numpy() * 2 - 1
    rgbs_tapnet = rgbs_tapnet.transpose(0, 2, 3, 1)
    query_points_tapnet = query_points.clone()
    query_points_tapnet[:, :, 1:] *= rescale_factor_hw.flip(0)
    query_points_tapnet = query_points_tapnet.flatten(0, 1)
    query_points_tapnet[:, 1:] = query_points_tapnet[:, 1:].flip(-1)
    query_points_tapnet = query_points_tapnet.numpy()
    query_points_tapnet = query_points_tapnet

    # 4. Run model
    outputs = jitted_f(rgbs_tapnet, query_points_tapnet)

    n_frames = rgbs.shape[0]
    n_masks, n_points_per_mask, _ = query_points.shape

    # 5. Postprocess
    tapnet_visibility_threshold = 0.5

    occlussion_logits = torch.from_numpy(np.asarray(outputs["occlusion"][0]).copy()).permute(1, 0)
    occlussion_logits = occlussion_logits.unflatten(1, (n_masks, n_points_per_mask))
    occlussion_probs = torch.sigmoid(occlussion_logits)
    visibilities_probs = 1 - occlussion_probs
    visibilities = visibilities_probs > tapnet_visibility_threshold

    trajectories = torch.from_numpy(np.asarray(outputs["tracks"][0]).copy()).permute(1, 0, 2)
    trajectories = trajectories.unflatten(1, (n_masks, n_points_per_mask))
    trajectories = trajectories / rescale_factor_hw.flip(-1)

    # 6. Visualize
    for mask_idx in range(n_masks):
        if mask_idx != 2:
            continue
        for frame_idx in range(n_frames):
            h, w = rgbs.shape[2], rgbs.shape[3]
            dpi = 100
            plt.figure(figsize=(w / dpi, h / dpi))
            plt.imshow(rgbs[frame_idx].permute(1, 2, 0).numpy(), interpolation="none")
            plt.scatter(trajectories[frame_idx, mask_idx, :, 0], trajectories[frame_idx, mask_idx, :, 1])
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.show()

    # 7. Benchmark forward pass speed in for loop
    n_loops = 100
    start_time = time.time()
    for _ in range(n_loops):
        outputs = jitted_f(rgbs_tapnet, query_points_tapnet)
    end_time = time.time()
    print(f"Forward pass speed: {(end_time - start_time) / n_loops * 1000} ms")

    print("Done")
