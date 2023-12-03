"""
Demo program for TAPIR, to make sure that pytorch+jax has been set up correctly.
The following snippet should run without error, and ideally use GPU/TPU to be fast when benchmarking.

Example usage:
```
python -m sam_pt.point_tracker.tapir.demo
```
"""
import time

import haiku as hk
import jax
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from torch.nn import functional as F

from demo.demo import load_demo_data
from . import tapir_model
from .configs.tapir_config import get_config

if __name__ == '__main__':
    # 1. Prepare config
    config = get_config()
    checkpoint_dir = "./models/tapir_ckpts/open_source_ckpt/"
    # Keep TF off the GPU; otherwise it hogs all the memory and leaves none for JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.experimental.set_visible_devices([], 'TPU')

    # 2. Prepare model
    checkpoint = np.load(checkpoint_dir + "tapir_checkpoint_panning.npy", allow_pickle=True).item()
    params, state = checkpoint["params"], checkpoint["state"]
    # tapir_model_kwargs = config.experiment_kwargs.config.shared_modules["tapir_model_kwargs"]
    tapir_model_kwargs = {
        "bilinear_interp_with_depthwise_conv": False,
        "pyramid_level": 0,
        "use_causal_conv": False,
    }


    def forward(rgbs, query_points):
        tapir = tapir_model.TAPIR(**tapir_model_kwargs)
        outputs = tapir(
            video=rgbs[None, ...],
            query_points=query_points[None, ...],
            query_chunk_size=64,
            is_training=False,
        )
        return outputs


    transform = hk.transform_with_state(forward)


    def f(rgbs_tapir, query_points_tapir):
        rng = jax.random.PRNGKey(72)
        outputs, _ = transform.apply(params, state, rng, rgbs_tapir, query_points_tapir)
        return outputs


    jitted_f = jax.jit(f)

    # 3. Prepare data
    rgbs, _, query_points = load_demo_data(
        frames_path="data/demo_data/bees",
        query_points_path="data/demo_data/query_points__bees.txt",
    )
    original_hw = rgbs.shape[-2:]
    tapir_input_hw = (
        config.experiment_kwargs.config.inference.resize_height, config.experiment_kwargs.config.inference.resize_width)
    rescale_factor_hw = torch.tensor(tapir_input_hw) / torch.tensor(original_hw)
    rgbs_tapir = F.interpolate(rgbs / 255, tapir_input_hw, mode="bilinear", align_corners=False, antialias=True)
    rgbs_tapir = rgbs_tapir.numpy() * 2 - 1
    rgbs_tapir = rgbs_tapir.transpose(0, 2, 3, 1)

    ## Take the loaded query points
    # query_points = query_points
    ## Or make a 16x16 grid of query points
    query_points = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    query_points[:, :, :, 0] = 1
    query_points[:, :, :, 1] = torch.linspace(1, original_hw[1] - 1, 16)
    query_points[:, :, :, 2] = torch.linspace(1, original_hw[0] - 1, 16).unsqueeze(-1)
    query_points = query_points.reshape(1, -1, 3)

    query_points_tapir = query_points.clone()
    query_points_tapir[:, :, 1:] *= rescale_factor_hw.flip(0)
    query_points_tapir = query_points_tapir.flatten(0, 1)
    query_points_tapir[:, 1:] = query_points_tapir[:, 1:].flip(-1)
    query_points_tapir = query_points_tapir.numpy()

    # 4. Run model
    outputs = jitted_f(rgbs_tapir, query_points_tapir)

    n_frames = rgbs.shape[0]
    n_masks, n_points_per_mask, _ = query_points.shape

    # 5. Postprocess
    tapir_visibility_threshold = 0.5

    expected_dist = torch.from_numpy(np.asarray(outputs["expected_dist"][0]).copy()).permute(1, 0)
    expected_dist = expected_dist.unflatten(1, (n_masks, n_points_per_mask))

    occlussion_logits = torch.from_numpy(np.asarray(outputs["occlusion"][0]).copy()).permute(1, 0)
    occlussion_logits = occlussion_logits.unflatten(1, (n_masks, n_points_per_mask))
    visibilities_probs = (1 - torch.sigmoid(occlussion_logits)) * (1 - torch.sigmoid(expected_dist))
    visibilities = visibilities_probs > tapir_visibility_threshold

    trajectories = torch.from_numpy(np.asarray(outputs["tracks"][0]).copy()).permute(1, 0, 2)
    trajectories = trajectories.unflatten(1, (n_masks, n_points_per_mask))
    trajectories = trajectories / rescale_factor_hw.flip(-1)

    # 6. Visualize
    mask_idx = -1
    for frame_idx in range(n_frames):
        h, w = rgbs.shape[2], rgbs.shape[3]
        dpi = 100
        plt.figure(figsize=(w / dpi, h / dpi))
        plt.imshow(rgbs[frame_idx].permute(1, 2, 0).numpy(), interpolation="none")
        x = trajectories[frame_idx, mask_idx, :, 0]
        y = trajectories[frame_idx, mask_idx, :, 1]
        colors = cm.rainbow(np.linspace(0, 1, len(y)))
        v = visibilities[frame_idx, mask_idx, :]
        # v = (visibilities[frame_idx, mask_idx, :] * 0) == 0
        x = x[v]
        y = y[v]
        colors = colors[v]
        plt.title(f"F{frame_idx:02}-M{mask_idx:02}-V{(visibilities_probs[frame_idx, mask_idx, :5] * 1)}")
        plt.scatter(x, y, color=colors, linewidths=6)
        plt.xlim(trajectories[..., 0].min(), trajectories[..., 0].max())
        plt.ylim(trajectories[..., 1].max(), trajectories[..., 1].min())
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
        time.sleep(0.1)

    # 7. Benchmark forward pass speed in for loop
    n_loops = 100
    start_time = time.time()
    for _ in range(n_loops):
        outputs = jitted_f(rgbs_tapir, query_points_tapir)
    end_time = time.time()
    print(f"Forward pass speed: {(end_time - start_time) / n_loops * 1000} ms")

    print("Done")
