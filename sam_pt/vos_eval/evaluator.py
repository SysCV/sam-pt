import torch
from abc import abstractmethod, ABC

from sam_pt.modeling.sam_pt import SamPt


class VOSEvaluator(ABC):
    """
    Abstract class for evaluating a model on the semi-supervised video object segmentation task.
    """

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    @abstractmethod
    def evaluate_video(self, video):
        """
        Evaluates model on a video and returns the predictions.

        Parameters
        ----------
        video : dict
            Dictionary with video data. It includes the following keys:
            'video_name': str - The name of the video.
            'video_id': int - The ID of the video.
            'image': List[torch.Tensor] - The frames of the video as uint8 tensors of shape (channels, height, width)
            'info': List[dict] - Information for each frame, includes keys like 'frame', 'save', 'shape', 'need_resize'.
            'target_hw': Tuple[int, int] - The target height and width for the predicted masks.
            'query_masks': torch.Tensor - The query masks as binary float32 tensor of shape (num_masks, height, width).
            'query_point_timestep': torch.Tensor - The query point timesteps as float32 tensor of shape (num_masks,).

        Returns
        -------
        dict
            Dictionary with predictions. It includes the following keys:
            'logits': List[torch.Tensor] - The logits as float32 tensors of shape (num_frames, height, width).
            'trajectories': torch.Tensor - The trajectories as float32 tensor
                                           of shape (num_frames, n_masks, n_points_per_mask, 2).
            'visibilities': torch.Tensor - The visibilities as float32 tensor
                                           of shape (num_frames, n_masks, n_points_per_mask).
            'scores': List[float] - The scores as list of 'num_masks' floats.
        """
        pass


class SamPtEvaluator(VOSEvaluator):
    def evaluate_video(self, video):
        self.model: SamPt = self.model
        device = self.model.device
        for k, v in video.items():
            if isinstance(v, torch.Tensor):
                video[k] = v.to(device)
        outputs = self.model(video)
        return {
            "logits": outputs["logits"],
            "trajectories": outputs["trajectories"],
            "visibilities": outputs["visibilities"],
            'scores': outputs['scores'],
        }
