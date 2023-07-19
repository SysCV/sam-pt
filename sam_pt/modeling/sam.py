"""
This module contains hydra wrapper classes for different types of Sam models. Each hydra wrapper provides functionality
for loading checkpoints and storing additional parameters that we used for variable interpolation within Hydra.
"""

import torch
from mobile_sam.modeling import Sam as MobileSam
from segment_anything.modeling import Sam
from segment_anything_hq.modeling import Sam as SamHQ


class BaseHydra:
    """
    Base class for hydra wrappers that loads the model checkpoint and stores additional parameters that we used for
    variable interpolation within Hydra.
    """

    def __init__(self, model, checkpoint, prompt_embed_dim, image_size, vit_patch_size, image_embedding_size, **kwargs):
        super().__init__(**kwargs)

        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            model.load_state_dict(self, state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint}.")

        # Store additional parameters used for variable interpolation within Hydra
        self.prompt_embed_dim = prompt_embed_dim
        self.image_size = image_size
        self.vit_patch_size = vit_patch_size
        self.image_embedding_size = image_embedding_size


class SamHydra(BaseHydra, Sam):
    """
    Wrapper for the Sam model that allows for loading a checkpoint
    and setting additional parameters used for variable interpolation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(Sam, *args, **kwargs)


class SamHQHydra(BaseHydra, SamHQ):
    """
    Wrapper for the SamHQ model that allows for loading a checkpoint
    and setting additional parameters used for variable interpolation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(SamHQ, *args, **kwargs)


class MobileSamHydra(BaseHydra, MobileSam):
    """
    Wrapper for the MobileSAM model that allows for loading a checkpoint
    and setting additional parameters used for variable interpolation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(MobileSam, *args, **kwargs)
