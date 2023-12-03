# Adapted from: https://github.com/hkchengrex/XMem/blob/083698bbb4c5ac0ffe1a8923a6c313de46169983/eval.py

import os
import shutil
import time
from os import path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam_pt.modeling.sam_pt import SamPt
from sam_pt.modeling.sam_pt_interactive import SamPtInteractive
from sam_pt.point_tracker.cotracker import CoTrackerPointTracker
from sam_pt.utils.query_points import extract_kmedoid_points
from sam_pt.utils.util import visualize_predictions, seed_all
from sam_pt.vos_eval.bdd100keval import BDD100KEvaluator
from sam_pt.vos_eval.data.mask_mapper import MaskMapper
from sam_pt.vos_eval.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset, \
    MOSETestDataset, BDD100KTestDataset
from sam_pt.vos_eval.davis2017eval import Davis2017Evaluator
from sam_pt.vos_eval.evaluator import VOSEvaluator


def evaluate(cfg):
    print(OmegaConf.to_yaml(cfg))

    seed_all(cfg.seed)

    wandb.init(
        entity=cfg.logging.wandb.entity,
        project=cfg.logging.wandb.project,
        name=cfg.logging.exp_id_verbose,
        group=cfg.logging.exp_id_verbose,
        config={
            "cfg": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            "work_dir": os.getcwd(),
            "hydra_cfg": HydraConfig.get(),
        },
    )
    wandb.run.log_code(cfg.logging.wandb.log_code_path)
    wandb.run.summary["work_dir"] = os.path.abspath(os.getcwd())

    """
    Data preparation
    """
    is_youtube = cfg.dataset.startswith('Y')
    is_davis = cfg.dataset.startswith('D')
    is_lv = cfg.dataset.startswith('LV')

    if is_youtube:
        if cfg.dataset == 'Y18':
            yv_path = cfg.y18_path
        elif cfg.dataset == 'Y19':
            yv_path = cfg.y19_path

        if cfg.split == 'val':
            cfg.split = 'valid'
            meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='valid', size=cfg.size,
                                                 longest_size=cfg.longest_size)
        elif cfg.split == 'test':
            meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='test', size=cfg.size,
                                                 longest_size=cfg.longest_size)
        else:
            raise NotImplementedError

    elif is_davis:
        if cfg.dataset == 'D16':
            if cfg.split == 'val':
                # Set up Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
                meta_dataset = DAVISTestDataset(cfg.d16_path, imset='../../2017/trainval/ImageSets/2016/val.txt',
                                                size=cfg.size, longest_size=cfg.longest_size)
            else:
                raise NotImplementedError
            palette = None
        elif cfg.dataset == 'D17':
            if cfg.split == 'val':
                meta_dataset = DAVISTestDataset(path.join(cfg.d17_path, 'trainval'), imset='2017/val.txt',
                                                size=cfg.size, longest_size=cfg.longest_size,
                                                return_all_gt_masks=cfg.simulate_interactive_point_correction)
            elif cfg.split == 'test':
                meta_dataset = DAVISTestDataset(path.join(cfg.d17_path, 'test-dev'), imset='2017/test-dev.txt',
                                                size=cfg.size, longest_size=cfg.longest_size)
            else:
                raise NotImplementedError

    elif is_lv:
        if cfg.dataset == 'LV1':
            meta_dataset = LongTestDataset(path.join(cfg.lv_path, 'long_video'), longest_size=cfg.longest_size)
        elif cfg.dataset == 'LV3':
            meta_dataset = LongTestDataset(path.join(cfg.lv_path, 'long_video_x3'), longest_size=cfg.longest_size)
        else:
            raise NotImplementedError
    elif cfg.dataset == 'G':
        meta_dataset = LongTestDataset(path.join(cfg.generic_path), size=cfg.size, longest_size=cfg.longest_size)
        if not cfg.save_all:
            cfg.save_all = True
            print('save_all is forced to be true in generic evaluation mode.')

    elif cfg.dataset == 'MOSE':
        meta_dataset = MOSETestDataset(
            data_root=cfg.mose_path,
            split=cfg.split,
            shortest_size=cfg.size,
            longest_size=cfg.longest_size,
        )

    elif cfg.dataset == 'BDD100K':
        meta_dataset = BDD100KTestDataset(
            data_root=cfg.bdd100k_path,
            split=cfg.split,
            shortest_size=cfg.size,
            longest_size=cfg.longest_size,
        )

    else:
        raise NotImplementedError

    if is_youtube or cfg.save_scores:
        out_path = path.join(cfg.output, 'Annotations')
    else:
        out_path = cfg.output

    torch.autograd.set_grad_enabled(False)

    # Set up loader
    meta_loader = meta_dataset.get_datasets()

    # Load our checkpoint
    model = instantiate(cfg.model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    # If CoTracker is used, the seed needs to be set again since building the model changed the seed
    if isinstance(model, SamPt) and isinstance(model.point_tracker, CoTrackerPointTracker):
        print('CoTracker is used, setting seed again.')
        seed_all(cfg.seed)

    vos_evaluator: VOSEvaluator = instantiate(cfg.evaluator, cfg=cfg, model=model)

    total_process_time = 0
    total_frames = 0

    # Start eval
    for vid_id, vid_reader in enumerate(tqdm(meta_loader, total=len(meta_dataset))):
        if cfg.vid_ids is not None:
            if vid_id not in cfg.vid_ids:
                continue

        if cfg.max_videos is not None and vid_id >= cfg.max_videos:
            print(f"Reached maximum number of videos to process: {cfg.max_videos}")
            break

        vid_name = vid_reader.vid_name
        print(f'Processing {vid_name}... [{vid_id + 1}/{len(meta_dataset)}]')
        if os.path.exists(out_path) and vid_name in os.listdir(out_path):
            print(f'Already processed {vid_name}, skipping...')
            continue

        vid_length = len(vid_reader) if cfg.max_frames is None else min(len(vid_reader), cfg.max_frames)
        mapper = MaskMapper()

        # Load all video frames
        rgbs = []
        infos = []
        all_gt_masks = []
        gt_ti_list = []
        gt_mask_list = []
        gt_labels_list = []
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=0)
        for ti, data in enumerate(loader):
            if cfg.max_frames is not None and ti >= cfg.max_frames:
                print(f"Reached maximum number of frames to process: {cfg.max_frames}")
                break
            rgb = data['rgb']
            msk = data.get('mask')
            info = data['info']
            need_resize = info['need_resize'][0]

            if cfg.flip:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            if cfg.dataset == "BDD100K":
                # BDD100K labels  have annotations for all visible objects at all frames,
                # not only for the query frame where the object first appears.
                # Thus, remove the other objects after their appearance from subsequent frames.
                label_has_been_seen = (msk[:, :, :, None] == torch.tensor(mapper.labels)[None, None, None, :]).any(-1)
                msk[label_has_been_seen] = 0
                if msk.sum() == 0:
                    msk = None

            if msk is not None:
                assert msk.shape[0] == 1, "The mask should be in index representation, each integer being a class"
                msk, new_mapped_labels = mapper.convert_mask(
                    mask=msk[0].numpy(),
                    old_labels_allowed=cfg.simulate_interactive_point_correction,
                )
                # msk, labels = mapper.convert_mask(msk[0].numpy(), dtype=np.uint8 if cfg.dataset != 'BDD100K' else np.int16)
                msk = torch.Tensor(msk)
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                all_gt_masks += [msk]
                for l_remapped in new_mapped_labels:
                    remapping = {v: k for k, v in mapper.remappings.items()}
                    l_original = remapping[l_remapped]
                    if l_original not in gt_labels_list:
                        m = msk[l_remapped - 1]
                        assert m.sum() > 0, "This mask should not be a dummy mask since the label has not been added yet"
                        gt_mask_list += [m]
                        gt_ti_list += [ti]
                        gt_labels_list += [l_original]
                    else:
                        zero_mask = msk[l_remapped - 1].sum() == 0
                        matches_existing = (gt_mask_list[gt_labels_list.index(l_original)] == msk[l_remapped - 1]).all()
                        assert zero_mask or matches_existing, "The mask should be the same as the existing one"

            assert rgb.shape[0] == 1, "The RGB should be a single image"
            rgb = rgb[0]
            rgb = (rgb * 255).type(torch.uint8)

            rgbs += [rgb]
            infos += [info]

        # Prepare model inputs
        assert all([m.sum().item() > 0 for m in gt_mask_list])
        height, width = infos[0]["shape"]
        print(f"height: {height}, width: {width}, rgbs[0].shape: {rgbs[0].shape}")

        query_point_timestep = torch.tensor(gt_ti_list, dtype=torch.float32)

        if cfg.input_only_one_gt_mask_point:
            query_masks = []
            for mask_idx in range(len(gt_mask_list)):
                point_coords = extract_kmedoid_points(gt_mask_list[mask_idx], n_points_to_select=1).numpy()
                timestep = gt_ti_list[mask_idx]
                model.sam_predictor.set_image(rgbs[timestep].permute(1, 2, 0).cpu().numpy())
                mask_frame_logits, iou_prediction_scores, low_res_masks = model.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=np.ones(len(point_coords)),
                    mask_input=None,
                    multimask_output=False,
                    return_logits=True,
                )
                print(f"[One GT Point Only] "
                      f"Video: {vid_id: 3d}, "
                      f"Mask: {mask_idx:1d}, "
                      f"Timestep: {timestep}, "
                      f"IoU: {iou_prediction_scores.item() * 100: 6.2f}")
                query_masks += [torch.from_numpy(mask_frame_logits > 0).float()[0]]
            query_masks = torch.stack(query_masks, dim=0)
        else:
            query_masks = torch.stack(gt_mask_list, dim=0)

        # The forward pass, wrapped in timing
        if torch.cuda.is_available():
            # Use CUDA events for timing if CUDA is available
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            # Use time.time() if CUDA is not available
            start = time.time()

        pred_logits_list = []
        pred_trajectories_list = []
        pred_visibilities_list = []
        n_masks = query_masks.shape[0]
        target_hw = (height.item(), width.item())
        pred_scores = []
        for i in range(0, n_masks, cfg.masks_batch_size):
            video = {
                "video_name": vid_name,
                "video_id": f"{vid_id:03d}--{vid_name}--mask-{i}",
                "image": rgbs,
                "info": infos,
                "target_hw": target_hw,
                "query_masks": query_masks[i:i + cfg.masks_batch_size],
                "query_point_timestep": query_point_timestep[i:i + cfg.masks_batch_size],
            }
            if isinstance(model, SamPtInteractive):
                assert len(all_gt_masks) == len(rgbs)
                video["gt_masks"] = [m[i:i + 1, :, :] for m in all_gt_masks]
            # outputs = {
            #     "logits": [
            #         torch.zeros((n_frames, height, width))
            #         for _ in range(len(query_masks[i:i + cfg.masks_batch_size]))
            #     ],
            #     "trajectories": None,
            #     "visibilities": None,
            #     "scores": [0] * cfg.masks_batch_size,
            # }
            outputs = vos_evaluator.evaluate_video(video)
            pred_logits_list += outputs['logits']
            if outputs['trajectories'] is not None:
                pred_trajectories_list += outputs['trajectories'].permute(1, 0, 2, 3)
                pred_visibilities_list += outputs['visibilities'].permute(1, 0, 2)
                pred_scores += outputs['scores']
        logits = torch.stack([torch.zeros_like(pred_logits_list[0])] + pred_logits_list, dim=1)
        del pred_logits_list

        assert torch.all(logits[:, 0] == 0), "The first mask should always be the background with zero logits"
        n_frames = logits.shape[0]
        if len(pred_trajectories_list) > 0:
            trajectories = torch.stack(pred_trajectories_list, dim=1)
            visibilities = torch.stack(pred_visibilities_list, dim=1)
            scores = torch.tensor(pred_scores)
        else:
            trajectories = torch.zeros((n_frames, n_masks, 1, 2), dtype=torch.float32)
            visibilities = torch.zeros((n_frames, n_masks, 1), dtype=torch.float32)
            scores = torch.zeros(n_masks, dtype=torch.float32)
        del pred_trajectories_list, pred_visibilities_list, pred_scores

        # Post process the predictions to set masks to zero for all frames before the query frame
        for i, gt_ti in enumerate(gt_ti_list):
            logits[:gt_ti, i + 1] = -1e8
        # Overwrite the predictions with the ground truth masks at corresponding timesteps
        for i, (gt_ti, gt_mask) in enumerate(zip(gt_ti_list, gt_mask_list)):
            gt_mask_resized = F.interpolate(gt_mask[None, None, :, :], target_hw, mode='nearest')[0, 0]
            logits[gt_ti, i + 1] = torch.where(gt_mask_resized.bool(), 1e8, -1e8)
        probs = F.softmax(logits, dim=1)
        if cfg.dataset == "BDD100K" and not cfg.visualize_results:
            del logits

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end) / 1000)
        else:
            end = time.time()
            total_process_time += end - start
        total_frames += len(rgbs)

        # Process the results
        for ti in range(len(rgbs)):
            prob = probs[ti]
            info = infos[ti]
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['need_resize'][0]

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]

            if cfg.flip:
                prob = torch.flip(prob, dims=[-1])

            # Probability mask -> index mask
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
            # out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8 if cfg.dataset != 'BDD100K' else np.int16)

            if cfg.save_scores:
                prob = (prob.detach().cpu().numpy() * 255).astype(np.uint8)

            # Save the mask
            if cfg.save_all or info['save'][0]:
                this_out_path = path.join(out_path, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4] + '.png'))

            if cfg.save_scores:
                import hickle as hkl
                np_path = path.join(cfg.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti == len(loader) - 1:
                    hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                if cfg.save_all or info['save'][0]:
                    hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')

        # Save the mask
        if cfg.save_all or info['save'][0]:
            if cfg.save_overlapping_masks:
                np_path = path.join(cfg.output, "../overlapping", vid_name)
                os.makedirs(np_path, exist_ok=True)
                torch.save(logits, os.path.join(np_path, f'logits.pt'))

        # Visualize results using wandb
        if cfg.visualize_results and vid_id < cfg.max_videos_to_visualize and (cfg.vid_ids_to_visualize is None or
                                                                               vid_id in cfg.vid_ids_to_visualize):
            n_frames, n_masks, n_points_per_mask, _ = trajectories.shape
            if hasattr(model, 'positive_points_per_mask'):
                positive_points_per_mask = model.positive_points_per_mask
            else:
                positive_points_per_mask = n_points_per_mask
            query_points = torch.zeros((n_masks, n_points_per_mask, 3), dtype=torch.float32)
            for i, gt_ti in enumerate(gt_ti_list):
                query_points[i, :, 0] = gt_ti
                query_points[i, :, 1:] = trajectories[gt_ti, i, :, :]
            query_scores = -1 * torch.ones(n_masks, dtype=torch.float32)  # Dummy query scores
            visualize_predictions(
                images=F.interpolate(
                    torch.stack(rgbs, dim=0).float(),
                    target_hw,
                    mode='bilinear'
                ).type(torch.uint8),
                # additional_log_images=additional_log_images,
                step=vid_id,
                query_points=query_points,
                trajectories=trajectories,
                visibilities=visibilities,
                query_masks=F.interpolate(query_masks[None, :, :, :], target_hw, mode='nearest')[0],
                query_scores=query_scores,
                sam_masks_logits=logits[:, 1:, :, :].permute(1, 0, 2, 3),
                positive_points_per_mask=positive_points_per_mask,
                verbose=cfg.verbose_visualisations,
                log_fmt=cfg.log_fmt,
            )

    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {total_frames}')
    if total_process_time > 0:
        print(f'FPS: {total_frames / total_process_time}')
    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2 ** 20)}')

    wandb.run.summary["total_frames"] = total_frames
    wandb.run.summary["total_process_time"] = total_process_time
    wandb.run.summary["fps"] = total_frames / total_process_time if total_process_time > 0 else 0

    if not cfg.save_scores:
        print('Making zip...')
        if is_youtube:
            shutil.make_archive(path.join(cfg.output, path.basename(cfg.output)), 'zip', cfg.output, 'Annotations')
        else:
            shutil.make_archive(cfg.output, 'zip', cfg.output)
        wandb.run.summary["work_dir"] = os.getcwd()
        wandb.run.summary["output_output"] = os.path.abspath(cfg.output)

    # For D16/D17, val split, get the evaluation results automatically
    if cfg.dataset in ["D16", "D17"] and cfg.split == 'val':
        print(os.path.abspath(cfg.output))
        print(os.path.abspath(cfg.d17_path))
        sequences = 'all'
        if cfg.vid_ids is not None:
            sequences = sorted(os.listdir(cfg.output))
            sequences = [s for s in sequences if s != "overlapping" and "." not in s]
            print(f"Evaluating only on the sequences present in the results folder: {sequences}")

        df_global, df_per_seq = Davis2017Evaluator(
            results_path=cfg.output,
            davis_path=os.path.join(cfg.d17_path, "trainval"),
            set="val",
            task="semi-supervised",
            year="2017" if cfg.dataset == "D17" else "2016",
            sequences=sequences,
        ).evaluate()

        wandb.log({"df_global": wandb.Table(dataframe=df_global)})
        wandb.log({"df_per_seq": wandb.Table(dataframe=df_per_seq)})

        wandb.run.summary["score"] = df_global["J&F-Mean"].item()

    if cfg.dataset == "BDD100K" and cfg.split == "val":
        print(os.path.abspath(cfg.output))
        print(os.path.abspath(cfg.bdd100k_path))

        sequences = os.listdir(cfg.output)
        print(f"Sequences to evaluate: {sequences}")
        df_global, df_per_seq = BDD100KEvaluator(
            results_path=cfg.output,
            dataset_path=os.path.join(cfg.bdd100k_path, cfg.split),
            sequences=sequences,
        ).evaluate()

        wandb.log({"df_global": wandb.Table(dataframe=df_global)})
        wandb.log({"df_per_seq": wandb.Table(dataframe=df_per_seq)})

        wandb.run.summary["n_sequences"] = len(sequences)

    print(f'Done. Find the results in {os.path.abspath(cfg.output)}')


@hydra.main(config_path="../../configs", config_name="vos_eval_root", version_base="1.1")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == '__main__':
    main()
