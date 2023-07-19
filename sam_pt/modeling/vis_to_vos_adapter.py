"""
This module contains the SamBasedVisToVosAdapter class which wraps a model
that performs Video Object Segmentation (VOS) and prompts it with query masks
generated using SAM's automatic mask proposals.
"""
import torch
import torch.nn.functional as F
from detectron2.utils import comm
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
from torch import nn

from sam_pt.modeling.sam_pt import SamPt
from sam_pt.utils.util import visualize_predictions


class SamBasedVisToVosAdapter(nn.Module):
    """
    This class wraps a model that performs VOS (Video Object Segmentation)
    and prompts it with query masks generated using SAM's automatic mask
    proposals. The adapter provides an interface needed to evaluate the
    approach on the VIS task in the Detectron2-based Mask2Former codebase.
    """

    def __init__(
            self,
            model: SamPt,
            sam_generator: SamAutomaticMaskGenerator,
            max_num_masks: int,
            masks_batch_size: int,
            visualize_results: bool,
            max_videos_to_visualize: int,
    ):
        """
        Parameters:
        -----------
        model : SamPt
            Model for the Video Object Segmentation (VOS).
        sam_generator : SamAutomaticMaskGenerator
            Generator of the automatic mask proposal.
        max_num_masks : int
            Maximum number of mask proposals to be generated.
        masks_batch_size : int
            Batch size for the number of masks.
        visualize_results : bool
            Flag to visualize results.
        max_videos_to_visualize : int
            Maximum number of videos to visualize.
        """
        super().__init__()
        self.model = model
        self.sam_generator = sam_generator
        self.max_num_masks = max_num_masks
        self.masks_batch_size = masks_batch_size
        self.visualize_results = visualize_results and comm.is_main_process()  # TODO: Maybe remove comm.is_main_process()
        self.max_videos_to_visualize = max_videos_to_visualize

        # Make baseline.to(device) work since the predictor is not a nn.Module
        self._sam_generator_model: Sam = self.sam_generator.predictor.model

    @property
    def device(self):
        return self._sam_generator_model.device

    def forward(self, batched_inputs):
        """Forward pass of the model."""
        vid_id, images_list, images_tensor, target_hw, query_masks, query_point_timestep, query_labels \
            = self._process_inputs_and_prepare_query_masks(batched_inputs)

        pred_logits_list, pred_trajectories_list, pred_visibilities_list, pred_scores \
            = self._track_masks_through_video(query_masks, query_point_timestep, images_list, images_tensor, target_hw)

        logits, trajectories, visibilities, scores = \
            self._format_predictions(
                pred_logits_list,
                pred_trajectories_list,
                pred_visibilities_list,
                pred_scores
            )

        if self.visualize_results and vid_id < self.max_videos_to_visualize:
            self._visualize_results(
                images_tensor,
                vid_id,
                query_point_timestep,
                query_masks,
                trajectories,
                visibilities,
                logits,
                target_hw,
            )

        results_dict = {
            "image_size": target_hw,
            "pred_scores": scores.tolist(),
            "pred_labels": query_labels.tolist(),
            "pred_masks": [m for m in logits > 0],
            "pred_logits": [m for m in logits],
            "trajectories": trajectories,
            "visibilities": visibilities,
        }

        return results_dict

    def _process_inputs_and_prepare_query_masks(self, batched_inputs):
        """Preprocess inputs and prepare generate query masks."""
        # TODO: Extend this method to make the model handle multiple videos and non-uint8 images
        assert len(batched_inputs) == 1, "Only single video inputs are supported"
        assert batched_inputs[0]["image"][0].dtype == torch.uint8, "Input images must be in uint8 format (0-255)"
        vid_id = batched_inputs[0]["video_id"]
        images_list = [i for i in batched_inputs[0]["image"]]
        images_tensor = torch.stack(images_list, dim=0)
        output_height, output_width = batched_inputs[0]["height"], batched_inputs[0]["width"]
        target_hw = (output_height, output_width)
        # Get query masks by using the automatic mask proposal generation mode from SAM
        result_records = self.sam_generator.generate(images_tensor[0].permute(1, 2, 0).cpu().numpy())
        print(f"Generated {len(result_records)} masks for video {vid_id}, "
              f"keeping the first {min(self.max_num_masks, len(result_records))}")
        query_masks = [torch.from_numpy(r["segmentation"]) for r in result_records[:self.max_num_masks]]
        query_masks = torch.stack(query_masks, dim=0).to(self.device)
        n_masks = query_masks.shape[0]
        query_point_timestep = torch.zeros(n_masks, dtype=torch.int64, device=self.device)  # We queried SAM for frame 0
        query_labels = torch.zeros(n_masks, dtype=torch.int64)  # Dummy labels, since SAM does not classify masks
        return vid_id, images_list, images_tensor, target_hw, query_masks, query_point_timestep, query_labels

    def _track_masks_through_video(self, query_masks, query_point_timestep, images_list, images_tensor, target_hw):
        """Tracks the query masks throughout the video using the VOS model."""
        n_masks = query_masks.shape[0]
        pred_logits_list = []
        pred_trajectories_list = []
        pred_visibilities_list = []
        pred_scores = []
        for i in range(0, n_masks, self.masks_batch_size):
            video = {
                "image": images_list,
                "target_hw": target_hw,
                "query_masks": query_masks[i:i + self.masks_batch_size],
                "query_point_timestep": query_point_timestep[i:i + self.masks_batch_size],
            }
            outputs = self.model(video)
            pred_logits_list += outputs['logits']
            pred_trajectories_list += outputs['trajectories'].permute(1, 0, 2, 3)
            pred_visibilities_list += outputs['visibilities'].permute(1, 0, 2)
            pred_scores += outputs['scores']

        # Sanity checks
        n_frames, channels, input_height, input_width = images_tensor.shape
        output_height, output_width = target_hw
        assert len(pred_logits_list) == n_masks
        assert pred_logits_list[0].shape == (n_frames, output_height, output_width)

        return pred_logits_list, pred_trajectories_list, pred_visibilities_list, pred_scores

    def _format_predictions(self, pred_logits_list, pred_trajectories_list, pred_visibilities_list, pred_scores):
        """Formats the predictions into the desired shape."""

        logits = torch.stack(pred_logits_list, dim=1)
        logits = logits.permute(1, 0, 2, 3)  # Mask first, then frame

        n_masks, n_frames, output_height, output_width = logits.shape

        if pred_trajectories_list[0] is not None:
            trajectories = torch.stack(pred_trajectories_list, dim=1)
            visibilities = torch.stack(pred_visibilities_list, dim=1)
            scores = torch.tensor(pred_scores)
        else:
            trajectories = torch.zeros((n_frames, n_masks, 1, 2), dtype=torch.float32)
            visibilities = torch.zeros((n_frames, n_masks, 1), dtype=torch.float32)
            scores = torch.zeros(n_masks, dtype=torch.float32)
        return logits, trajectories, visibilities, scores

    def _visualize_results(self, images_tensor, vid_id, query_point_timestep, query_masks, trajectories, visibilities,
                           logits, target_hw):
        """Visualizes the results using wandb."""
        n_frames, n_masks, n_points_per_mask, _ = trajectories.shape
        if hasattr(self.model, 'positive_points_per_mask'):
            positive_points_per_mask = self.model.positive_points_per_mask
        else:
            positive_points_per_mask = n_points_per_mask
        query_points = torch.zeros((n_masks, n_points_per_mask, 3), dtype=torch.float32)
        for i, t in enumerate(query_point_timestep.tolist()):
            query_points[i, :, 0] = t
            query_points[i, :, 1:] = trajectories[t, i, :, :]
        query_scores = -1 * torch.ones(n_masks, dtype=torch.float32)  # Dummy query scores
        visualize_predictions(
            images=F.interpolate(images_tensor.float(), target_hw, mode='bilinear').type(torch.uint8),
            step=vid_id,
            query_points=query_points,
            trajectories=trajectories,
            visibilities=visibilities,
            query_masks=F.interpolate(query_masks[None, :, :, :].float(), target_hw, mode='nearest')[0],
            query_scores=query_scores,
            sam_masks_logits=logits,
            positive_points_per_mask=positive_points_per_mask,
            annot_size=1,
            annot_line_width=1,
            visualize_query_masks=False,
        )
