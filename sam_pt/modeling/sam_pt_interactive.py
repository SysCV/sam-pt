"""
Interactive SAM-PT simulator with ground truth mask given as input.
"""
import json
import os
import pickle
from collections import namedtuple, Counter
from typing import Tuple, List, Optional

import cv2
import imageio
import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from torch.nn import functional as F
from tqdm import tqdm

from sam_pt.modeling.sam_pt import SamPt


class SamPtInteractive(SamPt):
    def __init__(
            self,
            interactions_max=300,
            interactions_max_per_frame=3,
            online_interactive_iou_threshold=0.9,
            disable_point_tracking=False,
            online=False,
            font_path=None,
            visualize_all_interactions_separately=False,
            visualize_all_interactions_as_mp4=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.disable_point_tracking = disable_point_tracking
        self.interactions_max = interactions_max
        self.interactions_max_per_frame = interactions_max_per_frame
        self.online = online
        self.font_path = font_path
        self.visualize_all_interactions_separately = visualize_all_interactions_separately
        self.visualize_all_interactions_as_mp4 = visualize_all_interactions_as_mp4

        self.online_interactive_iou_threshold = online_interactive_iou_threshold
        # TODO: Hardcoded offline thresholds
        self.offline_interactive_iou_thresholds = [
            0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.88, 0.90, 0.92, 0.95,
            # More than 0.95 is too high, the mask is already good enough
        ]

    def forward(self, video, debug=True):
        # TODO: Code duplication with SamPt
        if self.training:
            raise NotImplementedError(f"{self._get_name()} does not support training...")

        # Unpack images
        images = torch.stack(video["image"], dim=0)
        n_frames, channels, height, width = images.shape
        assert images[0].dtype == torch.uint8, "Input images must be in uint8 format (0-255)"

        # Prepare queries
        if video.get("query_masks") is not None:  # E.g., when evaluating on the VOS task
            assert video.get("query_points") is None
            print("SAM-PT: Using query masks")
            query_masks = video["query_masks"].float()
            query_points_timestep = video["query_point_timestep"]
            query_points = self.extract_query_points(images, query_masks, query_points_timestep)
            query_scores = None
        elif video.get("query_points") is not None:  # E.g., when evaluating on the railway demo
            print("SAM-PT: Using query points")
            query_points = video["query_points"]
            query_masks = self.extract_query_masks(images, query_points)
            query_scores = None
        else:
            raise ValueError("No query points or masks provided")
        n_masks, n_points_per_mask, _ = query_points.shape
        assert query_masks.shape == (n_masks, height, width)

        ################################################################################################################
        # > Interactive Point Correction

        # Allowed interactions:
        # 1. Add point
        # 2. Remove point
        # (  Maybe:  )
        # 3. Drag point ( add + remove )
        # 4. Clear points ( remove all points )

        if self.online:
            interactive_iou_thresholds = [self.online_interactive_iou_threshold]
        else:
            interactive_iou_thresholds = self.offline_interactive_iou_thresholds
        interactions_max = self.interactions_max
        interactions_max_per_frame = self.interactions_max_per_frame
        interactions_left = interactions_max

        if self.disable_point_tracking:
            interactive_iou_thresholds = [1.0]
            interactions_max = interactions_max_per_frame * n_frames

        assert n_masks == 1, "Interactive point correction only works with a single mask"
        assert "gt_masks" in video, "Ground truth masks must be provided for interactive point correction"
        gt_masks = torch.stack(video["gt_masks"]).squeeze(1).bool()

        if self.visualize_all_interactions_as_mp4:
            viz_frames = []

        # 1. Cache SAM encoder
        print("Caching SAM encoder...")
        sam_encoder_cache = {}
        for frame_idx in tqdm(range(n_frames)):
            img = images[frame_idx].permute(1, 2, 0).cpu().numpy()
            self.sam_predictor.set_image(img)
            sam_encoder_cache[frame_idx] = {
                "features": self.sam_predictor.features,
            }
            if "original_size" not in sam_encoder_cache:
                sam_encoder_cache["original_size"] = self.sam_predictor.original_size
                sam_encoder_cache["input_size"] = self.sam_predictor.input_size
            assert sam_encoder_cache["original_size"] == self.sam_predictor.original_size
            assert sam_encoder_cache["input_size"] == self.sam_predictor.input_size

        def set_image_from_cache(frame_idx):
            self.sam_predictor.features = sam_encoder_cache[frame_idx]["features"]
            assert sam_encoder_cache["original_size"] == self.sam_predictor.original_size
            assert sam_encoder_cache["input_size"] == self.sam_predictor.input_size

        def predict_mask(frame_idx, point_coords, point_labels):
            if len(point_coords) == 0 or point_labels.sum() == 0:
                return torch.zeros((height, width), dtype=torch.float32), torch.tensor(0, dtype=torch.float32)
            set_image_from_cache(frame_idx)
            point_coords = self.sam_predictor.transform.apply_coords_torch(
                coords=point_coords,
                original_size=self.sam_predictor.original_size,
            )
            pos_point_coords = point_coords[point_labels == 1]
            pos_point_labels = point_labels[point_labels == 1]
            neg_point_coords = point_coords[point_labels == 0]
            neg_point_labels = point_labels[point_labels == 0]

            # 1. First pass: only positive points
            mask_frame_logits, iou_prediction_scores, low_res_masks = self.sam_predictor.predict_torch(
                point_coords=pos_point_coords[None, :, :].to(self.device),
                point_labels=pos_point_labels[None, :].to(self.device),
                boxes=None,
                mask_input=None,
                multimask_output=False,
                return_logits=True,
            )
            # 2. Second pass: positive and negative points, if any negative points
            if len(neg_point_coords) > 0:
                mask_frame_logits, iou_prediction_scores, low_res_masks = self.sam_predictor.predict_torch(
                    point_coords=point_coords[None, :, :].to(self.device),
                    point_labels=point_labels[None, :].to(self.device),
                    boxes=None,
                    mask_input=low_res_masks,
                    multimask_output=False,
                    return_logits=True,
                )

            # 3. Iterative refinement
            if self.iterative_refinement_iterations > 0:
                for refinement_iteration in range(self.iterative_refinement_iterations):
                    m = mask_frame_logits[0, 0, :, :] > 0
                    if m.sum() < 2:
                        break
                    yx = m.nonzero()
                    refinement_box = torch.tensor([
                        yx[:, 1].min(),  # xmin
                        yx[:, 0].min(),  # ymin
                        yx[:, 1].max(),  # xmax
                        yx[:, 0].max(),  # ymax
                    ], dtype=torch.float, device=self.device)
                    mask_frame_logits, iou_prediction_scores, low_res_masks = self.sam_predictor.predict_torch(
                        point_coords=point_coords[None, :, :].to(self.device),
                        point_labels=point_labels[None, :].to(self.device),
                        boxes=refinement_box[None, None, :],
                        mask_input=low_res_masks,
                        multimask_output=False,
                        return_logits=True,
                    )

            return mask_frame_logits[0, 0, :, :].cpu(), iou_prediction_scores[0, 0].cpu()

        def predict_mask_against_gt_mask(frame_idx, trajectories, visibilities, point_labels):
            curr_point_vis = visibilities[frame_idx, 0, :]
            curr_point_coords = trajectories[frame_idx, 0, :, :][curr_point_vis == 1]
            curr_point_labels = point_labels[curr_point_vis == 1]
            logits, sam_scores = predict_mask(frame_idx, curr_point_coords, curr_point_labels)
            m = logits > 0
            gt_m = gt_masks[frame_idx]

            def my_iou():
                intersection = m & gt_m
                union = m | gt_m
                if union.sum() == 0:
                    iou_score = 1.0
                else:
                    iou_score = intersection.sum() / union.sum()
                return iou_score

            # %timeit my_iou()
            # 14.9 ms ± 10 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
            # iou_score = my_iou()

            from davis2017.metrics import db_eval_boundary, db_eval_iou
            # %timeit db_eval_iou(m.numpy(), gt_m.numpy())
            # 658 µs ± 4.66 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
            iou_score = torch.tensor(db_eval_iou(m.numpy(), gt_m.numpy()))

            # %timeit db_eval_boundary(m.numpy(), gt_m.numpy())
            # 8.6 ms ± 1.04 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            boundary_score = torch.tensor(db_eval_boundary(m.numpy(), gt_m.numpy()))

            return m, gt_m, iou_score, boundary_score, logits, sam_scores, curr_point_vis, curr_point_coords, curr_point_labels

        def full_pass(trajectories, visibilities, point_labels):
            logits = torch.zeros((n_masks, n_frames, height, width), dtype=torch.float32)
            scores_per_frame = torch.zeros((n_frames, n_masks), dtype=torch.float32)
            ious = []
            boundary_scores = []
            for frame_idx in range(n_frames):
                (_, _, iou_score, boundary_score, logits_, sam_scores, _, _, _
                 ) = predict_mask_against_gt_mask(frame_idx, trajectories, visibilities, point_labels)
                logits[:, frame_idx] = logits_
                scores_per_frame[frame_idx] = sam_scores
                ious += [iou_score]
                boundary_scores += [boundary_score]
            scores = scores_per_frame.mean(dim=0)
            return logits, scores, scores_per_frame, ious, boundary_scores

        # 2. Run point tracking with query points
        if self.disable_point_tracking:
            trajectories = torch.zeros((n_frames, 1, 1, 2), dtype=torch.float32)
            visibilities = torch.zeros((n_frames, 1, 1), dtype=torch.float32)
            point_labels = torch.ones((1,), dtype=torch.int)
            interactions_left = interactions_max
            print(f"Point tracking is disabled. Interactions left: {interactions_left}")
        else:
            print(f"Running initial point tracking using {n_points_per_mask} query points...")
            trajectories, visibilities = self._track_points(images, query_points)
            point_labels = torch.ones((n_points_per_mask,), dtype=torch.int)
            point_labels[self.positive_points_per_mask:] = 0
            interactions_left -= len(query_points[0])  # Query points count as point interactions
            print(f"Initial point tracking done. Interactions used: {len(query_points[0])} of {interactions_left}")

        # 3. Keep correcting until no more correction budget left
        achieved_iou_thresholds_cache = []
        current_threshold = interactive_iou_thresholds.pop(0)
        HistoryEntry = namedtuple('HistoryEntry',
                                  'action type '
                                  'frame_idx point_idx '
                                  'iou_before iou_after '
                                  'interaction_idx current_iou_threshold '
                                  'overall_iou_before overall_iou_after '
                                  'boundary_score_before boundary_score_after '
                                  'overall_boundary_score_before overall_boundary_score_after '
                                  'jf_score_before jf_score_after')
        interaction_history: List[HistoryEntry] = []
        current_pass_ious = []
        current_pass_boundary_scores = []
        frame_idx = 0
        frame_interactions = 0
        _, _, _, prev_iou, prev_boundary_score = full_pass(trajectories, visibilities, point_labels)
        prev_iou = np.mean(prev_iou)
        prev_boundary_score = np.mean(prev_boundary_score)
        while interactions_left > 0:
            # 4. Go frame by frame and check if the predicted mask is good enough
            #    If not, "interactively" correct the mask using the ground truth (no user input, this is simulation)
            #    If yes, move to the next frame
            if frame_idx == n_frames:
                assert len(current_pass_ious) == n_frames, f"Expected {n_frames} IoUs, got {len(current_pass_ious)}"
                achieved_iou_thresholds_cache += [{
                    "current_threshold": current_threshold,
                    "trajectories": trajectories.clone(),
                    "visibilities": visibilities.clone(),
                    "point_labels": point_labels.clone(),
                    "interaction_history": interaction_history.copy(),
                    "interactions_left": interactions_left,
                    "average_iou": np.mean(current_pass_ious),
                    "average_boundary_score": np.mean(current_pass_boundary_scores),
                    "current_pass_ious": current_pass_ious,
                    "current_pass_boundary_scores": current_pass_boundary_scores,
                }]
                if len(interactive_iou_thresholds) == 0:
                    print(f"No more thresholds left. Interactions left: {interactions_left}. Stopping.")
                    break
                current_threshold = interactive_iou_thresholds.pop(0)
                print(f"New threshold: {current_threshold}")
                frame_idx = 0
                frame_interactions = 0
                current_pass_ious = []
                current_pass_boundary_scores = []

            (m, gt_m, iou_score, boundary_score, _, _, curr_point_vis, curr_point_coords, curr_point_labels
             ) = predict_mask_against_gt_mask(frame_idx, trajectories, visibilities, point_labels)

            # 5. Check if the predicted mask is good enough
            if iou_score >= current_threshold:
                if self.visualize_all_interactions_as_mp4:
                    viz_before_int = visualize_int(
                        rgb=images[frame_idx] / 255,
                        pred_mask=m.float(),
                        gt_mask=gt_m.float(),
                        point_vis=curr_point_vis,
                        point_coords=curr_point_coords,
                        point_labels=curr_point_labels,
                        text=(
                            f"Frame idx: {frame_idx + 1}\n"
                            f"Interaction idx: {interactions_max - interactions_left + 1}\n"
                            f"Frame IoU: {iou_score * 100:.1f}\n"
                            f"Action: None (IoU above {current_threshold * 100:.0f})"
                        ),
                        font_path=self.font_path,
                    )
                    viz_frames += [viz_before_int]
                    print(f"viz_frames: {len(viz_frames)}")
                frame_idx += 1
                frame_interactions = 0
                current_pass_ious += [iou_score]
                current_pass_boundary_scores += [boundary_score]
                continue

            # 6. If not, "interactively" correct the mask using the ground truth
            tp_mask = m & gt_m
            tn_mask = ~m & ~gt_m
            fp_mask = m & ~gt_m
            fn_mask = ~m & gt_m
            if self.visualize_all_interactions_separately:
                viz_1 = visualize_pred_against_gt_mask(images[frame_idx] / 255, m.float(), gt_m.float(),
                                                       curr_point_vis, curr_point_coords, curr_point_labels,
                                                       debug_text=f"Frame: {frame_idx}\n"
                                                                  f"Threshold: {current_threshold:.2f}\n"
                                                                  f"Interactions left: {interactions_left - 1}\n")

            # 6.0. Prepare point categories
            PointCategory = namedtuple('Point', 'visible positive correct tp tn fp fn')
            points: List[PointCategory] = []
            for point_idx in range(trajectories.shape[2]):
                visible = visibilities[frame_idx, 0, point_idx].item() == 1
                if not visible:
                    points += [PointCategory(visible, None, None, None, None, None, None)]
                    continue
                positive = point_labels[point_idx].item() == 1
                x, y = trajectories[frame_idx, 0, point_idx, :].round().int().tolist()
                tp = tp_mask[y, x].item()
                tn = tn_mask[y, x].item()
                fp = fp_mask[y, x].item()
                fn = fn_mask[y, x].item()
                correct = (positive and (tp or fn)) or (not positive and (tn or fp))
                points += [PointCategory(visible, positive, correct, tp, tn, fp, fn)]
            # for p in points:
            #     print(p)

            incorrect_negative_points = [p.visible and not p.positive and not p.correct for p in points]
            incorrect_positive_points = [p.visible and p.positive and not p.correct for p in points]

            # 6.1. Remove incorrect negative points
            if any(incorrect_negative_points):
                first_incorrect_negative_point_idx = incorrect_negative_points.index(True)
                visibilities[frame_idx:, 0, first_incorrect_negative_point_idx] = 0
                action_name = "remove"
                action_type = "negative"
                action_point_idx = first_incorrect_negative_point_idx

            # 6.2. Remove incorrect positive points
            elif any(incorrect_positive_points):
                first_incorrect_positive_point_idx = incorrect_positive_points.index(True)
                visibilities[frame_idx:, 0, first_incorrect_positive_point_idx] = 0
                action_name = "remove"
                action_type = "positive"
                action_point_idx = first_incorrect_positive_point_idx

            else:
                action_name = "add"
                action_point_idx = trajectories.shape[2]
                if fn_mask.sum() > fp_mask.sum():
                    # 6.3. Add new positive points to cover the false negatives
                    mask = fn_mask
                    label = 1
                    action_type = "positive"
                else:
                    # 6.4. Add negative points to cover the false positives
                    mask = fp_mask
                    label = 0
                    action_type = "negative"

                mask_sum = mask.sum().item()
                assert mask_sum > 0
                x, y = extract_largest_cluster_points(mask, n_points_to_select=min(3, mask_sum))[0, :].tolist()
                if self.disable_point_tracking:
                    curr_trajectories = torch.zeros((n_frames, 1, 1, 2), dtype=torch.float32)
                    curr_visibilities = torch.zeros((n_frames, 1, 1), dtype=torch.float32)
                    curr_trajectories[frame_idx, 0, 0, :] = torch.tensor([x, y], dtype=torch.float32)
                    curr_visibilities[frame_idx, 0, 0] = 1
                else:
                    curr_query_points = torch.tensor([0, x, y], dtype=torch.int)[None, None, :]
                    curr_trajectories, curr_visibilities = self._track_points(images[frame_idx:], curr_query_points)
                    curr_trajectories[0, 0, 0, :] = torch.tensor([x, y], dtype=torch.float32)
                    curr_visibilities[0, 0, 0] = 1
                    curr_trajectories = torch.cat(
                        [torch.zeros((frame_idx, 1, 1, 2), dtype=torch.float32), curr_trajectories])
                    curr_visibilities = torch.cat(
                        [torch.zeros((frame_idx, 1, 1), dtype=torch.float32), curr_visibilities])

                trajectories = torch.cat([trajectories, curr_trajectories], dim=2)
                visibilities = torch.cat([visibilities, curr_visibilities], dim=2)
                point_labels = torch.cat([point_labels, torch.tensor([label], dtype=torch.int)], dim=0)

            # 7. Update the history
            # Time it
            (m_after, _, iou_score_after, boundary_score_after, _, _,
             curr_point_vis_after, curr_point_coords_after, curr_point_labels_after,
             ) = predict_mask_against_gt_mask(frame_idx, trajectories, visibilities, point_labels)
            if self.disable_point_tracking:
                next_iou = prev_iou
                next_boundary_score = prev_boundary_score
            else:
                _, _, _, next_iou, next_boundary_score = full_pass(trajectories, visibilities, point_labels)
                next_iou = np.mean(next_iou)
                next_boundary_score = np.mean(next_boundary_score)
            interaction_entry = HistoryEntry(
                action=action_name,
                type=action_type,
                frame_idx=frame_idx,
                point_idx=action_point_idx,
                iou_before=iou_score.item(),
                iou_after=iou_score_after.item(),
                interaction_idx=interactions_left,
                current_iou_threshold=current_threshold,
                overall_iou_before=prev_iou.item(),
                overall_iou_after=next_iou.item(),
                boundary_score_before=boundary_score.item(),
                boundary_score_after=boundary_score_after.item(),
                overall_boundary_score_before=prev_boundary_score.item(),
                overall_boundary_score_after=next_boundary_score.item(),
                jf_score_before=(prev_iou.item() + prev_boundary_score.item()) / 2,
                jf_score_after=(next_iou.item() + next_boundary_score.item()) / 2,
            )
            interaction_history += [interaction_entry]

            if self.visualize_all_interactions_separately:
                viz_2 = visualize_pred_against_gt_mask(images[frame_idx] / 255, m_after.float(), gt_m.float(),
                                                       curr_point_vis_after,
                                                       curr_point_coords_after,
                                                       curr_point_labels_after)
                viz_2 = viz_2[:, viz_2.shape[1] // 2:]
                visualize_all_interactions_separately = np.concatenate([viz_1, viz_2], axis=1)
                interactions_root = f"interactions/{video['video_id']}/"
                if not os.path.exists(interactions_root):
                    os.makedirs(interactions_root)
                img_save_path = (
                    f"{interactions_root}"
                    f"{interactions_max - interactions_left + 1:04d}"
                    f"_{frame_idx:02d}"
                    f"_{action_name[:3]}"
                    f"_{action_type[:3]}"
                    f"_{action_point_idx:02d}"
                    f"_{current_threshold:.3f}"
                    f"_{iou_score:.3f}--{iou_score_after:.3f}"
                    f"_{prev_iou:.3f}--{next_iou:.3f}"
                    f"_{prev_boundary_score:.3f}--{next_boundary_score:.3f}"
                    f".png"
                )
                print(os.path.abspath(img_save_path))
                Image.fromarray(visualize_all_interactions_separately).save(img_save_path)
                # dpi = 120
                # plt.figure(figsize=(visualize_all_interactions_separately.shape[1] / dpi, visualize_all_interactions_separately.shape[0] / dpi), dpi=dpi)
                # plt.imshow(visualize_all_interactions_separately)
                # plt.axis('off')
                # plt.tight_layout(pad=0)
                # plt.show()

            if self.visualize_all_interactions_as_mp4:
                viz_before_int = visualize_int(
                    rgb=images[frame_idx] / 255,
                    pred_mask=m.float(),
                    gt_mask=gt_m.float(),
                    point_vis=curr_point_vis,
                    point_coords=curr_point_coords,
                    point_labels=curr_point_labels,
                    text=(
                        f"Frame idx: {frame_idx + 1}\n"
                        f"Interaction idx: {interactions_max - interactions_left + 1}\n"
                        f"Frame IoU: {iou_score * 100:.1f}\n"
                        f"Action: {action_name.title()} {action_type[:3]}. point\n"
                    ),
                    font_path=self.font_path,
                )
                viz_frames += [viz_before_int] * 4
                viz_after_int = visualize_int(
                    rgb=images[frame_idx] / 255,
                    pred_mask=m_after.float(),
                    gt_mask=gt_m.float(),
                    point_vis=curr_point_vis_after,
                    point_coords=curr_point_coords_after,
                    point_labels=curr_point_labels_after,
                    text=(
                        f"Frame idx: {frame_idx + 1}\n"
                        f"Interaction idx: {interactions_max - interactions_left + 1}\n"
                        f"Frame IoU: {iou_score_after * 100:.1f}\n"
                        f"Action: {action_name.title()} {action_type[:3]}. point\n"
                    ),
                    font_path=self.font_path,
                )
                viz_frames += [viz_after_int] * 7
                print(f"viz_frames: {len(viz_frames)}")

            interactions_left -= 1
            frame_interactions += 1
            prev_iou = next_iou
            prev_boundary_score = next_boundary_score
            if iou_score_after >= current_threshold or frame_interactions >= interactions_max_per_frame:
                frame_idx += 1
                frame_interactions = 0
                current_pass_ious += [iou_score_after]
                current_pass_boundary_scores += [boundary_score_after]
            print(f"Interaction: {interaction_entry}")

        # 8. Re-run SAM for each frame to get the final mask predictions
        logits, scores, scores_per_frame, final_pass_ious, final_pass_boundary_scores = full_pass(trajectories,
                                                                                                  visibilities,
                                                                                                  point_labels)
        final_iou = np.mean(final_pass_ious)
        print(f"Final IoU: {np.mean(final_pass_ious)}")
        print(f"Final IoUs: {final_pass_ious}")
        print(f"Interactions left: {interactions_left}")
        print(f"Interaction history:")
        for entry in interaction_history:
            print(entry)

        # Save the interaction history to a file
        interactions_root = f"interactions/{video['video_id']}/"
        if not os.path.exists(interactions_root):
            os.makedirs(interactions_root)
        with open(f"{interactions_root}history.json", "w") as f:
            json.dump(interaction_history, f, indent=4)

        # Pickle the achieved_iou_thresholds_cache and final results
        with open(f"{interactions_root}achieved_iou_thresholds_cache.pkl", "wb") as f:
            x = achieved_iou_thresholds_cache
            for i in range(len(x)):
                x[i]["interaction_history"] = [he._asdict() for he in x[i]["interaction_history"]]
            pickle.dump(x, f)
        with open(f"{interactions_root}final.pkl", "wb") as f:
            pickle.dump({
                "trajectories": trajectories,
                "visibilities": visibilities,
                "point_labels": point_labels,
                "logits": logits,
                "scores": scores,
                "scores_per_frame": scores_per_frame,
            }, f)

        # Plot the overall IoU history
        iou_threshold_list = [history_entry.current_iou_threshold for history_entry in interaction_history]
        achieved_iou_threshold_list = [
            max([0] + [iou for iou in iou_threshold_list[:i + 1] if iou < iou_threshold_list[i]])
            for i in range(len(iou_threshold_list))
        ]
        overall_iou_before_list = [history_entry.overall_iou_before for history_entry in interaction_history]
        overall_iou_after_list = [history_entry.overall_iou_after for history_entry in interaction_history]
        final_iou_list = [
            max([achieved_iou_threshold_list[i], overall_iou_after_list[i]])
            for i in range(len(iou_threshold_list))
        ]
        with open(f"{interactions_root}overall_iou_history.json", "w") as f:
            json.dump({
                "threshold": iou_threshold_list,
                "achieved_threshold": achieved_iou_threshold_list,
                "before": overall_iou_before_list,
                "after": overall_iou_after_list,
            }, f, indent=4)
        plt.figure(figsize=(10, 5))
        plt.plot(iou_threshold_list, label="Threshold")
        plt.plot(achieved_iou_threshold_list, label="Achieved Threshold (ckpt)")
        plt.plot(overall_iou_before_list, label="Before")
        plt.plot(overall_iou_after_list, label="After")
        plt.plot(final_iou_list, label="Final", linewidth=3)

        plt.xlabel("Interaction")
        plt.ylabel("IoU")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{interactions_root}overall_iou_history.png")
        # plt.show()
        plt.close()

        if self.visualize_all_interactions_as_mp4:
            fps = 15
            imageio.mimsave(f"interactions/{video['video_id']}.mp4", viz_frames, format='mp4', fps=fps)
            wandb_video = wandb.Video(f"interactions/{video['video_id']}.mp4", fps=fps)
            wandb.log({f"interactions/{video['video_id']}.mp4": wandb_video})

        if len(achieved_iou_thresholds_cache) > 0:
            best_iou_idx = np.argmax([x["average_iou"] for x in achieved_iou_thresholds_cache])
            best = achieved_iou_thresholds_cache[best_iou_idx]
            if best["average_iou"] > final_iou:
                print(f"Using IoU threshold from cache: {best['current_threshold']}")
                print(f"IoU: {best['average_iou']}")
                print(f"IoUs: {best['current_pass_ious']}")
                trajectories = best["trajectories"]
                visibilities = best["visibilities"]
                point_labels = best["point_labels"]
                # interaction_history = best["interaction_history"]
                # interactions_left = best["interactions_left"]

                logits, scores, scores_per_frame, final_pass_ious, final_pass_boundary_scores = full_pass(trajectories,
                                                                                                          visibilities,
                                                                                                          point_labels)
                assert np.isclose(np.mean(final_pass_ious), best["average_iou"], atol=0.001)
                assert np.isclose(np.mean(final_pass_boundary_scores), best["average_boundary_score"], atol=0.001)

        n_points_per_mask = trajectories.shape[2]

        ################################################################################################################

        # TODO: Code duplication with SamPt

        # # Run tracking
        # self.frame_annotations = [[] for _ in range(n_frames)]
        # if not self.use_point_reinit:
        #     trajectories, visibilities, logits, scores, scores_per_frame = self._forward(images, query_points)
        # else:
        #     trajectories, visibilities, logits, scores, scores_per_frame = self._forward_w_reinit(images, query_points)

        # Post-processing
        target_hw = video["target_hw"]
        resize_factor = torch.tensor(target_hw) / torch.tensor(logits.shape[-2:])
        assert (resize_factor[0] - resize_factor[1]).abs().item() < 0.01, "The resizing should have been isotropic"

        if logits.shape[-2:] != target_hw:
            logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        trajectories = trajectories * resize_factor

        # masks = logits > .0

        if query_scores is not None:
            assert query_scores.shape == scores.shape
            final_scores = query_scores ** 4 * scores ** 0.4
            print("[Baseline]")
            print(f"Scores per frame: {scores_per_frame.tolist()}")
            print(f"Scores:       {scores.tolist()}")
            print(f"Query Scores: {query_scores.tolist()}")
            print(f"Final Scores: {final_scores.tolist()}")
        else:
            final_scores = scores

        assert logits.shape == (n_masks, n_frames, target_hw[0], target_hw[1])
        assert scores.shape == (n_masks,)
        assert scores_per_frame.shape == (n_frames, n_masks)
        assert trajectories.shape == (n_frames, n_masks, n_points_per_mask, 2)
        assert visibilities.shape == (n_frames, n_masks, n_points_per_mask)

        # results_dict = {
        #     "logits": [m for m in logits],
        #     "scores": final_scores.tolist(),
        #     "scores_per_frame": scores_per_frame.tolist(),
        #     "trajectories": trajectories,
        #     "visibilities": visibilities,
        # }
        results_dict = {
            "logits": [m for m in logits],
            "scores": None,
            "scores_per_frame": None,
            "trajectories": None,
            "visibilities": None,
        }

        return results_dict


def extract_largest_cluster_points(
        mask,
        n_points_to_select,
        dbscan_points=18000,
        db_largest_cluster_min_points=180,
        kmedian_points=720,
):
    """
    Randomly select the specified number of points from the largest cluster in the mask.
    The largest cluster is computed using DBSCAN. Then the points are selected using K-Medoids.
    If the largest cluster has less than `db_largest_cluster_min_points`, the input mask is used directly for K-Medoids.
    DBSCAN uses a random subset of `dbscan_points` points to avoid out of memory errors.
    K-Medoids uses a random subset of `kmedian_points` points to run faster.

    :param mask: binary mask tensor with shape (H, W)
    :param n_points_to_select: number of points to select from the largest cluster
    :param dbscan_points: number of points to use for DBSCAN
    :param db_largest_cluster_min_points: minimum number of points in the largest cluster, otherwise use the mask directly for kmedoids
    :param kmedian_points: number of points to use for K-Median
    :return: tensor of shape (n_points_to_select, 2) containing the selected points
    """
    mask_pixels = mask.nonzero().float()

    # Randomly select at most `dbscan_points` points to avoid out of memory errors
    mask_pixels = mask_pixels[torch.randperm(len(mask_pixels))[:dbscan_points]]
    assert len(mask_pixels) > 0

    # Compute the largest cluster in the mask
    dbscan_eps = 2.4 * (mask.shape[0] * mask.shape[1]) / dbscan_points
    db = DBSCAN(eps=dbscan_eps, min_samples=10).fit(mask_pixels)
    cluster_count = Counter(db.labels_)
    if -1 in cluster_count:
        cluster_count.pop(-1)
    if len(cluster_count) == 0:
        print("WARNING: No clusters found in a mask of mask.sum()={mask.sum()} pixels, using the mask instead")
        largest_cluster_points = mask.nonzero().float()
    else:
        largest_cluster_id = cluster_count.most_common(1)[0][0]
        largest_cluster_points = mask_pixels[db.labels_ == largest_cluster_id]
        if len(largest_cluster_points) < db_largest_cluster_min_points:
            print(f"WARNING: Largest cluster has only {len(largest_cluster_points)} points, using the mask instead")
            largest_cluster_points = mask.nonzero().float()

    # Sample N points from the largest cluster by performing K-medoids with K=N
    largest_cluster_points = largest_cluster_points[torch.randperm(len(largest_cluster_points))[:kmedian_points]]
    selected_points = KMedoids(n_clusters=n_points_to_select).fit(largest_cluster_points).cluster_centers_
    selected_points = torch.from_numpy(selected_points).type(torch.float32)

    # (y, x) -> (x, y)
    selected_points = selected_points.flip(1)

    return selected_points


def visualize_pred_against_gt_mask(
        rgb,
        pred_mask,
        gt_mask,
        point_vis,
        point_coords,
        point_labels,
        debug_text="",
        plot=False,
):
    # Compute the true positives, false positives, and false negatives
    tp = gt_mask * pred_mask
    fp = (1 - gt_mask) * pred_mask
    fn = gt_mask * (1 - pred_mask)

    # Create the mixed mask
    mixed_mask = torch.zeros_like(rgb)
    mixed_mask[0] = fp + fn  # Red and yellow channels (false positives and false negatives)
    mixed_mask[1] = tp + fn  # Green and yellow channels (true positives and false negatives)

    # Convert to numpy for display
    rgb_img = rgb.permute(1, 2, 0).numpy()
    gt_img = np.stack((gt_mask.numpy(),) * 3, axis=-1)
    pred_img = np.stack((pred_mask.numpy(),) * 3, axis=-1)
    mixed_img = mixed_mask.permute(1, 2, 0).numpy()

    # To uint8
    rgb_img = (rgb_img * 255).astype(np.uint8)
    gt_img = (gt_img * 255).astype(np.uint8)
    pred_img = (pred_img * 255).astype(np.uint8)
    mixed_img = (mixed_img * 255).astype(np.uint8)

    # Ensure contiguous layout
    rgb_img = np.ascontiguousarray(rgb_img)
    gt_img = np.ascontiguousarray(gt_img)
    pred_img = np.ascontiguousarray(pred_img)
    mixed_img = np.ascontiguousarray(mixed_img)

    # Put the points on top of all the images
    assert point_coords.shape[0] == point_labels.shape[0]
    point_indices = torch.arange(point_vis.shape[0])[point_vis == 1]
    for (x, y), is_positive, point_idx in zip(
            point_coords.round().int().tolist(),
            point_labels.bool().tolist(),
            point_indices.tolist(),
    ):
        if is_positive:
            color = (0, 255, 0)
            marker = 'o'
        else:
            color = (255, 0, 0)
            marker = 'x'

        for img, color_ in [
            (rgb_img, color),
            (gt_img, color),
            (pred_img, color),
            (mixed_img, (0, 0, 255)),  # Blue as green and red are not visible on top of the mixed mask
        ]:
            annot_size = 8
            annot_line_width = 4
            if marker == 'o':
                cv2.circle(img, (x, y), annot_size, color_, annot_line_width)
            elif marker == 'x':
                line_size = annot_size // 2 + 1
                cv2.line(img, (x - line_size, y - line_size), (x + line_size, y + line_size), color_, annot_line_width)
                cv2.line(img, (x + line_size, y - line_size), (x - line_size, y + line_size), color_, annot_line_width)
            else:
                raise ValueError(f"Unknown marker: {marker}")

        point_name = f"{point_idx:03d}"
        cv2.putText(img, point_name, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255))

    # Write the IoU value in the top left corner
    iou = (tp.sum() / (tp.sum() + fp.sum() + fn.sum())).item()
    txt = f"IoU: {iou:.3f}"
    mixed_img = put_debug_text_onto_image(mixed_img, txt, font_scale=1)

    # Write the debug text in the top left corner
    if debug_text:
        rgb_img = put_debug_text_onto_image(rgb_img, debug_text, font_scale=0.5)

    # Concatenate images
    top_row = np.concatenate((rgb_img, mixed_img), axis=1)
    bottom_row = np.concatenate((gt_img, pred_img), axis=1)
    concatenated_image = np.concatenate((top_row, bottom_row), axis=0)

    # Plot without padding
    if plot:
        dpi = 120
        plt.figure(figsize=(concatenated_image.shape[1] / dpi, concatenated_image.shape[0] / dpi), dpi=dpi)
        plt.imshow(concatenated_image)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

    # Return the concatenated image as an np array
    return concatenated_image


def visualize_int(
        rgb,
        pred_mask,
        gt_mask,
        point_vis,
        point_coords,
        point_labels,
        text,
        annot_size=36,
        annot_line_width=12,
        plot=False,
        font_path=None,
):
    if font_path is not None:
        assert os.path.exists(font_path), f"Font file not found: {font_path}"

    # Convert to numpy for display
    rgb_img = rgb.permute(1, 2, 0).numpy()
    pred_img = np.stack((pred_mask.numpy(),) * 3, axis=-1)

    # To uint8
    rgb_img = (rgb_img * 255).astype(np.uint8)
    pred_img = (pred_img * 255).astype(np.uint8)

    # Ensure contiguous layout
    rgb_img = np.ascontiguousarray(rgb_img)
    pred_img = np.ascontiguousarray(pred_img)

    # Put the points on top of all the images
    assert point_coords.shape[0] == point_labels.shape[0]
    point_indices = torch.arange(point_vis.shape[0])[point_vis == 1]
    for (x, y), is_positive, point_idx in zip(
            point_coords.round().int().tolist(),
            point_labels.bool().tolist(),
            point_indices.tolist(),
    ):
        if is_positive:
            color = (0, 255, 0)
            marker = 'o'
        else:
            color = (255, 0, 0)
            marker = 'x'

        for img, color_ in [
            (rgb_img, color),
            (pred_img, color),
        ]:
            if marker == 'o':
                cv2.circle(img, (x, y), annot_size, color_, annot_line_width)
            elif marker == 'x':
                line_size = annot_size // 2 + 1
                cv2.line(img, (x - line_size, y - line_size), (x + line_size, y + line_size), color_, annot_line_width)
                cv2.line(img, (x + line_size, y - line_size), (x - line_size, y + line_size), color_, annot_line_width)
            else:
                raise ValueError(f"Unknown marker: {marker}")

        point_name = f"{point_idx:03d}"
        rgb_img = put_fancy_text_onto_image(rgb_img, point_name, font_path, 18, x - 15, y)
        pred_img = put_fancy_text_onto_image(pred_img, point_name, font_path, 18, x - 15, y)

    # Put the text on top of the RGB image
    rgb_img = put_fancy_text_onto_image(rgb_img, text.strip(), font_path)

    # Concatenate images
    concatenated_image = np.concatenate((rgb_img, pred_img), axis=1)

    # Plot without padding
    if plot:
        dpi = 120
        plt.figure(figsize=(concatenated_image.shape[1] / dpi, concatenated_image.shape[0] / dpi), dpi=dpi)
        plt.imshow(concatenated_image)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

    # Return the concatenated image as an np array
    return concatenated_image


def put_debug_text_onto_image(img: np.ndarray, text: str, font_scale: float = 0.5, left: int = 5, top: int = 20,
                              font_thickness: int = 1, text_color_bg: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Overlay debug text on the provided image.

    Parameters
    ----------
    img : np.ndarray
        A 3D numpy array representing the input image. The image is expected to have three color channels.
    text : str
        The debug text to overlay on the image. The text can include newline characters ('\n') to create multi-line text.
    font_scale : float, default 0.5
        The scale factor that is multiplied by the font-specific base size.
    left : int, default 5
        The left-most coordinate where the text is to be put.
    top : int, default 20
        The top-most coordinate where the text is to be put.
    font_thickness : int, default 1
        Thickness of the lines used to draw the text.
    text_color_bg : Tuple[int, int, int], default (0, 0, 0)
        The color of the text background in BGR format.

    Returns
    -------
    img : np.ndarray
        A 3D numpy array representing the image with the debug text overlaid.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    font_color = (255, 255, 255)

    # Write each line of text in a new row
    (_, label_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    if text_color_bg is not None:
        for i, line in enumerate(text.split('\n')):
            (line_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            top_i = top + i * label_height
            cv2.rectangle(img, (left, top_i - label_height), (left + line_width, top_i), text_color_bg, -1)
    for i, line in enumerate(text.split('\n')):
        top_i = top + i * label_height
        cv2.putText(img, line, (left, top_i), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img


def put_fancy_text_onto_image(
        img: np.ndarray,
        text: str,
        font_path: Optional[str] = None,
        font_size: int = 63,
        left: int = 5,
        top: int = 20,
        padding: int = 5,
        text_color: tuple = (255, 255, 255),
        text_color_bg: tuple = (0, 0, 0),
        alpha: int = 180,
) -> np.ndarray:
    """
    Overlay fancy text on the provided image using the PIL library with a transparent background.

    Parameters
    ----------
    ... (other parameters are unchanged)
    alpha : int, default 128
        The alpha value for the text background, where 0 is fully transparent and 255 is fully opaque.

    Returns
    -------
    ... (return value is unchanged)
    """
    # Convert the numpy array to a PIL Image
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(image)

    # Load a font file
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Calculate text size and position
    text_size = draw.textsize(text, font=font)
    text_position = (left, top)

    # Create a new image for the transparent rectangle
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Calculate the position and size of the background rectangle
    rectangle_position = (
        left - padding,
        top - padding,
        left + text_size[0] + padding,
        top + text_size[1] + padding,
    )

    # Draw a semi-transparent rectangle on the overlay
    overlay_draw.rectangle(rectangle_position, fill=text_color_bg + (alpha,))

    # Blend the overlay with the original image
    image = Image.alpha_composite(image.convert('RGBA'), overlay)

    # Draw the text onto the image with transparency
    draw = ImageDraw.Draw(image)
    draw.text(text_position, text, font=font, fill=text_color)

    # Convert back to RGB to strip alpha channel and convert to numpy array
    final_img = np.array(image.convert('RGB'))

    return final_img
