import numpy as np
import tensorflow as tf
import torch
from torch.nn import functional as F

from sam_pt.point_tracker import PointTracker


class TapirPointTracker(PointTracker):
    """
    A point tracker that uses TAPIR from https://arxiv.org/abs/2306.08637 to track points.
    """

    def __init__(self, checkpoint_path, visibility_threshold):
        from .configs.tapir_config import get_config
        super().__init__()

        # Keep TF off the GPU; otherwise it hogs all the memory and leaves none for JAX
        tf.config.experimental.set_visible_devices([], 'GPU')
        tf.config.experimental.set_visible_devices([], 'TPU')

        # # v1: use the last GPU
        # # Hardcode JAX to use the last GPU (the first is reserved for other modules from PyTorch)
        # # The environmental flag `XLA_PYTHON_CLIENT_PREALLOCATE=false` is also required along with this
        # gpus = jax.devices('gpu')
        # device = gpus[-1]
        # jax.jit ... device=device

        # v2: share the gpu with Sam since they are run sequentially
        #     but make jax free up the allocated memory once it is done
        #     by setting the environmental variable `XLA_PYTHON_CLIENT_ALLOCATOR=platform`

        assert checkpoint_path is not None
        self.checkpoint_path = checkpoint_path
        self.config = get_config()
        self.visibility_threshold = visibility_threshold
        self.jitted_forward = self._create_jitted_forward()

    def _create_jitted_forward(self):
        import haiku as hk
        import jax
        from . import tapir_model

        checkpoint = np.load(self.checkpoint_path, allow_pickle=True).item()
        params, state = checkpoint["params"], checkpoint["state"]
        # tapir_model_kwargs = self.config.experiment_kwargs.config.shared_modules["tapir_model_kwargs"]
        tapir_model_kwargs = {
            "bilinear_interp_with_depthwise_conv": False,
            "pyramid_level": 0,
            "use_causal_conv": False,
        }

        def _forward(rgbs, query_points):
            tapir = tapir_model.TAPIR(**tapir_model_kwargs)
            outputs = tapir(
                video=rgbs,
                query_points=query_points,
                query_chunk_size=64,
                is_training=False,
            )
            return outputs

        transform = hk.transform_with_state(_forward)

        def forward(rgbs_tapir, query_points_tapir):
            rng = jax.random.PRNGKey(72)
            outputs, _ = transform.apply(params, state, rng, rgbs_tapir, query_points_tapir)
            return outputs

        return jax.jit(forward)

    def forward(self, rgbs, query_points, summary_writer=None):
        batch_size, n_frames, channels, height, width = rgbs.shape
        n_points = query_points.shape[1]

        # 1. Prepare image resizing
        original_hw = (height, width)
        tapir_input_hw = (
            self.config.experiment_kwargs.config.inference.resize_height,
            self.config.experiment_kwargs.config.inference.resize_width,
        )
        rescale_factor_hw = torch.tensor(tapir_input_hw) / torch.tensor(original_hw)

        # 2. Prepare inputs
        assert rgbs.dtype == torch.uint8
        rgbs_tapir = F.interpolate(rgbs.flatten(0, 1) / 255, tapir_input_hw, mode="bilinear", align_corners=False,
                                   antialias=True)
        rgbs_tapir = rgbs_tapir.unflatten(0, (batch_size, n_frames))
        rgbs_tapir = rgbs_tapir.cpu().numpy() * 2 - 1
        rgbs_tapir = rgbs_tapir.transpose(0, 1, 3, 4, 2)
        query_points_tapir = query_points.cpu().clone()
        query_points_tapir[:, :, 1:] *= rescale_factor_hw.flip(0)
        query_points_tapir[:, :, 1:] = query_points_tapir[:, :, 1:].flip(-1)  # flip x and y
        query_points_tapir = query_points_tapir.numpy()

        # 3. Run model
        self._create_jitted_forward()  # TODO: Cannot the function be compiled only once?
        outputs = self.jitted_forward(rgbs_tapir, query_points_tapir)

        # 4. Postprocess outputs
        expected_dist = torch.from_numpy(np.asarray(outputs["expected_dist"]).copy()).permute(0, 2, 1)
        occlussion_logits = torch.from_numpy(np.asarray(outputs["occlusion"]).copy()).permute(0, 2, 1)
        visibilities_probs = (1 - torch.sigmoid(occlussion_logits)) * (1 - torch.sigmoid(expected_dist))
        visibilities = visibilities_probs > self.visibility_threshold
        trajectories = torch.from_numpy(np.asarray(outputs["tracks"]).copy()).permute(0, 2, 1, 3)
        trajectories = trajectories / rescale_factor_hw.flip(-1)

        return trajectories, visibilities
