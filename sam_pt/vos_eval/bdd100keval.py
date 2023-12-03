"""
Run the main for example as:
```bash
python -m sam_pt.vos_eval.bdd100keval \
  --results_path /srv/beegfs02/scratch/visobt4s/data/3d_point_tracking/sampt_outputs/K9.000--debug--cotracker-0--1-1024/eval_BDD100K_val__dummy \
  --dataset_path /scratch/leikel/frano/03-code/sam-pt/data/bdd100k/vos/val \
  --eval_only_on_the_sequences_present_in_the_results

python -m sam_pt.vos_eval.bdd100keval \
  --results_path /srv/beegfs02/scratch/visobt4s/data/3d_point_tracking/sampt_outputs/K9.003--cotracker-bdd100k-less-other-neg-points/eval_BDD100K_val \
  --dataset_path /scratch/leikel/frano/03-code/sam-pt/data/bdd100k/vos/val

python -m sam_pt.vos_eval.bdd100keval \
  --results_path /srv/beegfs02/scratch/visobt4s/data/3d_point_tracking/sampt_outputs/SegGPT--BDD100K-val--in-sampt-env/eval_BDD100K_val__dummy \
  --dataset_path /scratch/leikel/frano/03-code/sam-pt/data/bdd100k/vos/val \
  --eval_only_on_the_sequences_present_in_the_results

python -m sam_pt.vos_eval.bdd100keval \
  --results_path /srv/beegfs02/scratch/visobt4s/data/3d_point_tracking/sampt_outputs/SegGPT--BDD100K-val--in-sampt-env/overlapping_dummy \
  --dataset_path /scratch/leikel/frano/03-code/sam-pt/data/bdd100k/vos/val \
  --eval_only_on_the_sequences_present_in_the_results --object_overlapping_allowed

```
"""
import argparse
import concurrent
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from time import time
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017.utils import db_statistics
from tqdm import tqdm


class Results(object):
    def __init__(self, root_dir, object_overlapping_allowed=False):
        self.root_dir = root_dir
        self.object_overlapping_allowed = object_overlapping_allowed

    def _read_mask(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f'{frame_id}.png')
            return np.array(Image.open(mask_path))
        except IOError as err:
            frames = os.listdir(os.path.join(self.root_dir, sequence))
            if len(frames) > 0:
                # XMem doesn't create save the masks up until the first object appears
                mask_path = os.path.join(self.root_dir, sequence, frames[0])
                return np.array(Image.open(mask_path)) * 0
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write("The frames have to be indexed PNG files placed inside the corespondent sequence "
                             "folder.\nThe indexes have to match with the initial frame.\n")
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_masks(self, sequence, masks_id, target_hw=None):
        # TODO: Remove the SegGPT hacks for readability

        seggpt_v1_path = os.path.join(self.root_dir, "../overlapping", sequence, f'probs.pt')
        if os.path.exists(seggpt_v1_path):
            probs = torch.load(seggpt_v1_path)
            probs = probs.clamp(min=1e-6, max=1 - 1e-6)

            if probs.shape[-2:] == (448, 448):
                # SegGPT memory saving hack was used. Upsample to original size.
                probs = F.interpolate(probs, size=target_hw, mode="bilinear", align_corners=False)

            if not self.object_overlapping_allowed:
                masks = probs.argmax(dim=1).numpy().astype(np.uint8)
            else:
                masks = probs.numpy() > 0.5
                # masks = masks[:, 1:, :, :]  # Remove background

            return masks

        if not self.object_overlapping_allowed:
            mask_0 = self._read_mask(sequence, masks_id[0])
            masks = np.zeros((len(masks_id), *mask_0.shape))

            for ii, m in enumerate(masks_id):
                masks[ii, ...] = self._read_mask(sequence, m)

            if set(np.unique(masks).tolist()) == {0., 255.}:
                masks = masks / 255.

            return masks

        else:
            seggpt_v2_path = os.path.join(self.root_dir, sequence, f'{sequence.replace("-chunk5", "")}-0000001.pt')
            if os.path.exists(seggpt_v2_path):
                masks = torch.load(seggpt_v2_path).numpy()
                # masks = masks[:, 1:, :, :]  # Remove background
                assert set(np.unique(masks).tolist()) == {False, True}
                return masks

            seggpt_v3_path = os.path.join(self.root_dir, sequence, f'masks.pt')
            if os.path.exists(seggpt_v3_path):
                masks = torch.load(seggpt_v3_path).numpy()
                # masks = masks[:, 1:, :, :]  # Remove background
                return masks

            raise RuntimeError(f"object_overlapping_allowed supported only for our SegGPT output formats.")


class BDD100K:
    def __init__(self, root: str, sequences: Union[str, list] = "all"):
        self.root = root
        self.img_path = os.path.join(self.root, "JPEGImages")
        self.mask_path = os.path.join(self.root, "Annotations")
        print(f"BDD100K root: {os.path.abspath(self.root)}")
        print(f"BDD100K img_path: {os.path.abspath(self.img_path)}")
        print(f"BDD100K mask_path: {os.path.abspath(self.mask_path)}")
        assert os.path.exists(self.root)
        assert os.path.exists(self.img_path)
        assert os.path.exists(self.mask_path)

        self.sequences = defaultdict(dict)
        if sequences == 'all':
            sequences = sorted(os.listdir(self.mask_path))
        for seq in sequences:
            images_path = os.path.join(self.img_path, seq)
            images = np.sort(glob(os.path.join(images_path, '*.jpg'))).tolist()
            if len(images) == 0:
                raise FileNotFoundError(f'Images for sequence {seq} not in {os.path.abspath(images_path)}.')

            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))

            self.sequences[seq]['images'] = images
            self.sequences[seq]['masks'] = masks

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_masks(self, sequence):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        assert (masks != 255).all()
        return masks, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq


class BDD100KEvaluation:
    def __init__(self, dataset_root, sequences: Union[str, list] = "all"):
        self.dataset = BDD100K(root=dataset_root, sequences=sequences)

    @staticmethod
    def compute_metrics_for_id(ii, masks_gt, masks_res, metric):
        os.sched_setaffinity(0, range(os.cpu_count()))

        # Only the frames after the object has appeared are considered
        # The first frame where the object appears is also not considered
        gt_visibility = np.sum(masks_gt, axis=(1, 2)) > 0
        appeared_frame_idx = np.where(gt_visibility)[0][0]
        if appeared_frame_idx == len(masks_gt) - 1:
            return (ii, 1, 1,
                    np.array([1.]), np.array([1.]),
                    np.array([1.]), np.array([1.]),
                    np.array([1.]), np.array([1.]))
        gt_visibility = gt_visibility[appeared_frame_idx + 1:]
        masks_gt = masks_gt[appeared_frame_idx + 1:, ...]
        masks_res = masks_res[appeared_frame_idx + 1:, ...]

        n_frames = len(gt_visibility) + 1
        visible_frames = np.sum(gt_visibility) + 1
        nonvisible_frames = n_frames - visible_frames

        j_metric, f_metric = None, None
        j_metric_vis, f_metric_vis = None, None
        j_metric_nonvis, f_metric_nonvis = None, None
        if 'J' in metric:
            j_metric = db_eval_iou(masks_gt, masks_res, None)
            j_metric_vis = j_metric[gt_visibility]
            j_metric_nonvis = j_metric[~gt_visibility]
        if 'F' in metric:
            f_metric = db_eval_boundary(masks_gt, masks_res, None)
            f_metric_vis = f_metric[gt_visibility]
            f_metric_nonvis = f_metric[~gt_visibility]

        return (ii, n_frames, visible_frames,
                j_metric, f_metric,
                j_metric_vis, f_metric_vis,
                j_metric_nonvis, f_metric_nonvis)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, metric, object_overlapping_allowed, mp_pool=True):
        max_res_id = int(np.max(all_res_masks))
        max_gt_id = int(np.max(all_gt_masks))
        assert max_gt_id > 0, "There are no objects in the ground truth!"
        assert max_res_id <= max_gt_id, "In your PNG files there is an index higher than the number of objects in the sequence!"

        # Initialize dictionaries for metrics
        j_metrics_res, f_metrics_res = {}, {}
        j_metrics_vis_res, f_metrics_vis_res = {}, {}
        j_metrics_nonvis_res, f_metrics_nonvis_res = {}, {}
        frame_count = {}

        if mp_pool:
            # Create a process pool executor to parallelize the loop
            with ProcessPoolExecutor() as executor:
                # Create a generator of tasks
                tasks = [
                    executor.submit(
                        BDD100KEvaluation.compute_metrics_for_id,
                        ii - 1,
                        all_gt_masks == ii,
                        all_res_masks == ii if not object_overlapping_allowed else all_res_masks[:, ii, :, :].copy(),
                        metric,
                    )
                    for ii
                    in range(1, max_gt_id + 1)  # Skip background
                ]

                # Iterate through the results as they complete
                for future in tqdm(concurrent.futures.as_completed(tasks), total=max_gt_id):
                    (ii, n_frames, visible_frames,
                     j_metric, f_metric,
                     j_metric_vis, f_metric_vis,
                     j_metric_nonvis, f_metric_nonvis) = future.result()
                    if j_metric is not None:
                        j_metrics_res[ii] = j_metric
                        j_metrics_vis_res[ii] = j_metric_vis
                        j_metrics_nonvis_res[ii] = j_metric_nonvis
                    if f_metric is not None:
                        f_metrics_res[ii] = f_metric
                        f_metrics_vis_res[ii] = f_metric_vis
                        f_metrics_nonvis_res[ii] = f_metric_nonvis
                    frame_count[ii] = (n_frames, visible_frames, n_frames - visible_frames)
        else:
            for ii in range(1, max_gt_id + 1):
                # Skip background
                (ii, n_frames, visible_frames,
                 j_metric, f_metric,
                 j_metric_vis, f_metric_vis,
                 j_metric_nonvis, f_metric_nonvis) = BDD100KEvaluation.compute_metrics_for_id(
                    ii - 1,
                    all_gt_masks == ii,
                    all_res_masks == ii if not object_overlapping_allowed else all_res_masks[:, ii, :, :],
                    metric,
                )

                if j_metric is not None:
                    j_metrics_res[ii] = j_metric
                    j_metrics_vis_res[ii] = j_metric_vis
                    j_metrics_nonvis_res[ii] = j_metric_nonvis
                if f_metric is not None:
                    f_metrics_res[ii] = f_metric
                    f_metrics_vis_res[ii] = f_metric_vis
                    f_metrics_nonvis_res[ii] = f_metric_nonvis
                frame_count[ii] = (n_frames, visible_frames, n_frames - visible_frames)

        return j_metrics_res, f_metrics_res, j_metrics_vis_res, f_metrics_vis_res, j_metrics_nonvis_res, f_metrics_nonvis_res, frame_count

    def evaluate(self, res_path, object_overlapping_allowed=False, metric=('J', 'F'), debug=False):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res['J_vis'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res['J_nonvis'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res['F_vis'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            metrics_res['F_nonvis'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        metrics_res["frame_count"] = {"n_frames": [], "visible_frames": [], "nonvisible_frames": []}

        # Sweep all sequences
        results = Results(root_dir=res_path, object_overlapping_allowed=object_overlapping_allowed)
        for seq in self.dataset.get_sequences():
            assert os.path.exists(os.path.join(res_path, seq)), f"Sequence {seq} not found in {res_path}."

        # print("Sanity check:")
        # for seq in tqdm(list(self.dataset.get_sequences())):
        #     # if seq != "b1ca2e5d-84cf9134-chunk5":
        #     #     continue
        #     print(f"seq: {seq}")
        #     all_gt_masks, all_masks_id = self.dataset.get_all_masks(seq)
        #     print(f"all_gt_masks.shape: {all_gt_masks.shape}")
        #     maxi = all_gt_masks.max()
        #     print(f"all_gt_masks.max(): {maxi}")
        #     all_res_masks = results.read_masks(seq, all_masks_id, target_hw=all_gt_masks.shape[-2:])
        #     print(f"all_res_masks.shape: {all_res_masks.shape}")
        #     assert all_res_masks.shape[1] == maxi + 1

        for seq in tqdm(list(self.dataset.get_sequences())):
            print(f"seq: {seq}")
            all_gt_masks, all_masks_id = self.dataset.get_all_masks(seq)
            print(f"all_gt_masks.shape: {all_gt_masks.shape}")
            maxi = all_gt_masks.max()
            print(f"all_gt_masks.max(): {maxi}")
            all_res_masks = results.read_masks(seq, all_masks_id, target_hw=all_gt_masks.shape[-2:])
            print(f"all_res_masks.shape: {all_res_masks.shape}")
            if object_overlapping_allowed:
                assert all_res_masks.shape[1] == maxi + 1
            (j_metrics_res, f_metrics_res,
             j_metrics_vis_res, f_metrics_vis_res,
             j_metrics_nonvis_res, f_metrics_nonvis_res,
             frame_count) = BDD100KEvaluation._evaluate_semisupervised(all_gt_masks, all_res_masks,
                                                                       metric, object_overlapping_allowed)
            for ii in range(int(all_gt_masks.max())):
                seq_name = f'{seq}_{ii + 1}'
                if 'J' in metric:
                    for m, res in [
                        ("J", j_metrics_res[ii]),
                        ("J_vis", j_metrics_vis_res[ii]),
                        ("J_nonvis", j_metrics_nonvis_res[ii]),
                    ]:
                        [JM, JR, JD] = db_statistics(res)
                        metrics_res[m]["M"].append(JM)
                        metrics_res[m]["R"].append(JR)
                        metrics_res[m]["D"].append(JD)
                        metrics_res[m]["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    for m, res in [
                        ("F", f_metrics_res[ii]),
                        ("F_vis", f_metrics_vis_res[ii]),
                        ("F_nonvis", f_metrics_nonvis_res[ii]),
                    ]:
                        [FM, FR, FD] = db_statistics(res)
                        metrics_res[m]["M"].append(FM)
                        metrics_res[m]["R"].append(FR)
                        metrics_res[m]["D"].append(FD)
                        metrics_res[m]["M_per_object"][seq_name] = FM
                metrics_res["frame_count"]["n_frames"].append(frame_count[ii][0])
                metrics_res["frame_count"]["visible_frames"].append(frame_count[ii][1])
                metrics_res["frame_count"]["nonvisible_frames"].append(frame_count[ii][2])

            # Show progress
            if debug:
                sys.stdout.write(seq + '\n')
                sys.stdout.flush()
        return metrics_res


class BDD100KEvaluator:
    def __init__(
            self,
            results_path: str,
            dataset_path: str,
            sequences: Union[str, list] = "all",
            object_overlapping_allowed: bool = False,
            short_object_threshold: int = 5,
            long_object_threshold: int = 30,
    ):
        """
        :param results_path: Path to the folder containing the sequences folders.
        :param davis_path: Path to the folder containing the `JPEGImages` and `Annotations` folders.
        :param sequences: List of sequences to evaluate. If "all", evaluate all sequences.
        """
        self.results_path = results_path
        self.dataset_path = dataset_path
        self.sequences = sequences
        self.object_overlapping_allowed = object_overlapping_allowed
        self.sot = short_object_threshold
        self.lot = long_object_threshold

    def evaluate(self):
        time_start = time()
        csv_name_global = f'global_results.csv'
        csv_name_per_sequence = f'per-sequence_results.csv'

        # Check if the method has been evaluated before, if so read the results, otherwise compute the results
        csv_name_global_path = os.path.join(self.results_path, csv_name_global)
        csv_name_per_sequence_path = os.path.join(self.results_path, csv_name_per_sequence)
        if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
            print('Using precomputed results...')
            table_g = pd.read_csv(csv_name_global_path)
            table_seq = pd.read_csv(csv_name_per_sequence_path)
        else:
            print(f'Evaluating BDD100K sequences...')
            # Create dataset and evaluate
            dataset_eval = BDD100KEvaluation(self.dataset_path, sequences=self.sequences)
            metrics_res = dataset_eval.evaluate(self.results_path, self.object_overlapping_allowed)
            J, F = metrics_res['J'], metrics_res['F']
            J_vis, F_vis = metrics_res['J_vis'], metrics_res['F_vis']
            J_nonvis, F_nonvis = metrics_res['J_nonvis'], metrics_res['F_nonvis']
            frame_count = metrics_res['frame_count']

            # Generate dataframe for the general results
            g_dict = {
                # Standard VOS Metrics
                'J&F-Mean': (np.mean(J["M"]) + np.mean(F["M"])) / 2.,
                'J-Mean': np.mean(J["M"]),
                'J-Recall': np.mean(J["R"]),
                'J-Decay': np.mean(J["D"]),
                'F-Mean': np.mean(F["M"]),
                'F-Recall': np.mean(F["R"]),
                'F-Decay': np.mean(F["D"]),

                # VOS Metrics for visible frames
                'J&F-Mean-Vis': (np.nanmean(J_vis["M"]) + np.nanmean(F_vis["M"])) / 2.,
                'J-Mean-Vis': np.nanmean(J_vis["M"]),
                'F-Mean-Vis': np.nanmean(F_vis["M"]),

                # VOS Metrics for non-visible frames
                'J&F-Mean-NonVis': (np.nanmean(J_nonvis["M"]) + np.nanmean(F_nonvis["M"])) / 2.,
                'J-Mean-NonVis': np.nanmean(J_nonvis["M"]),
                'F-Mean-NonVis': np.nanmean(F_nonvis["M"]),

                # VOS Metrics for objects visible for a short time (1 -- self.sot visible frames)
                'J&F-Mean-Short': np.array(J["M"])[np.array(frame_count["visible_frames"]) < self.sot].mean() / 2. +
                                  np.array(F["M"])[np.array(frame_count["visible_frames"]) < self.sot].mean() / 2.,
                'J-Mean-Short': np.array(J["M"])[np.array(frame_count["visible_frames"]) < self.sot].mean(),
                'F-Mean-Short': np.array(F["M"])[np.array(frame_count["visible_frames"]) < self.sot].mean(),

                # VOS Metrics for objects visible for a medium-long time (self.sot+1 -- self.lot visible frames)
                'J&F-Mean-Medium': np.array(J["M"])[
                                       (np.array(frame_count["visible_frames"]) >= self.sot) &
                                       (np.array(frame_count["visible_frames"]) < self.lot)].mean() / 2. +
                                   np.array(F["M"])[
                                       (np.array(frame_count["visible_frames"]) >= self.sot) &
                                       (np.array(frame_count["visible_frames"]) < self.lot)].mean() / 2.,
                'J-Mean-Medium': np.array(J["M"])[
                    (np.array(frame_count["visible_frames"]) >= self.sot) &
                    (np.array(frame_count["visible_frames"]) < self.lot)].mean(),
                'F-Mean-Medium': np.array(F["M"])[
                    (np.array(frame_count["visible_frames"]) >= self.sot) &
                    (np.array(frame_count["visible_frames"]) < self.lot)].mean(),

                # VOS Metrics for objects visible for a long time (>self.lot+1 visible frames)
                'J&F-Mean-Long': np.array(J["M"])[np.array(frame_count["visible_frames"]) >= self.lot].mean() / 2. +
                                 np.array(F["M"])[np.array(frame_count["visible_frames"]) >= self.lot].mean() / 2.,
                'J-Mean-Long': np.array(J["M"])[np.array(frame_count["visible_frames"]) >= self.lot].mean(),
                'F-Mean-Long': np.array(F["M"])[np.array(frame_count["visible_frames"]) >= self.lot].mean(),
            }
            g_dict = {k: [v] for k, v in g_dict.items()}

            table_g = pd.Series(g_dict)
            with open(csv_name_global_path, 'w') as f:
                table_g.to_csv(f, index=True, float_format="%.3f")
            print(f'Global results saved in {csv_name_global_path}')

            # Generate a dataframe for the per sequence results
            seq_dict = {
                'Sequence': list(J['M_per_object'].keys()),
                'J-Mean': list(J['M_per_object'].values()),
                'F-Mean': list(F['M_per_object'].values()),
                'J-Mean-Vis': list(J_vis['M_per_object'].values()),
                'F-Mean-Vis': list(F_vis['M_per_object'].values()),
                'J-Mean-NonVis': list(J_nonvis['M_per_object'].values()),
                'F-Mean-NonVis': list(F_nonvis['M_per_object'].values()),
                'n_frames': frame_count['n_frames'],
                'visible_frames': frame_count['visible_frames'],
                'nonvisible_frames': frame_count['nonvisible_frames'],
                'short-medium-long': ['short' if v < self.sot else 'medium' if v < self.lot else 'long'
                                      for v in frame_count['visible_frames']],
            }
            table_seq = pd.DataFrame(seq_dict)
            with open(csv_name_per_sequence_path, 'w') as f:
                table_seq.to_csv(f, index=False, float_format="%.3f")
            print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

        # Print the results
        sys.stdout.write(f"\n---------- Per sequence results ----------\n")
        print(table_seq.to_string(index=False))
        sys.stdout.write(f"--------------------------- Global results ---------------------------\n")
        print(table_g.to_string(index=True))
        total_time = time() - time_start
        sys.stdout.write('\nTotal time:' + str(total_time))

        return table_g, table_seq


if __name__ == '__main__':
    # multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description='Evaluate a method on the BDD100K dataset')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to the folder containing the sequences folders.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the BDD100K folder containing the `JPEGImages` and `Annotations` folders.')
    parser.add_argument('--eval_only_on_the_sequences_present_in_the_results', action='store_true',
                        help='If True, evaluate only on the sequences present in the results folder.')
    parser.add_argument('--object_overlapping_allowed', action='store_true',
                        help='If True, evaluate each mask separately and allow overlap.')
    args = parser.parse_args()

    sequences = 'all'
    if args.eval_only_on_the_sequences_present_in_the_results:
        assert os.path.exists(args.results_path)
        sequences = sorted(os.listdir(args.results_path))
        sequences = [s for s in sequences if s != "overlapping" and "." not in s]
        print(f"Evaluating only on the sequences present in the results folder: {sequences}")

    evaluator = BDD100KEvaluator(
        results_path=args.results_path,
        dataset_path=args.dataset_path,
        sequences=sequences,
        object_overlapping_allowed=args.object_overlapping_allowed,
    )
    evaluator.evaluate()
