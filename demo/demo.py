"""
This module is used to run a demo of the SAM-PT model. It loads a video and a set of query points, runs inference on
the video, and visualizes the results.

To run the interactive demo (where query points are specified using mouse clicks on a pop-up window), run the following:
```bash
python -m demo.demo frames_path=/path/to/video/frames query_points_path=null
```

To run the demo using a set of pre-specified query points, run the following:
```bash
python -m demo.demo frames_path=/path/to/video/frames query_points_path=/path/to/query/points
```
"""

import cv2
import glob
import hydra
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import wandb
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from time import sleep

from sam_pt.utils.util import visualize_predictions


@hydra.main(config_path="../configs", config_name="demo", version_base="1.1")
def main(cfg):
    # Set seed
    print(f"Setting seed to {cfg.seed}")
    set_seed(cfg.seed)

    # Set up logging
    setup_logging(cfg)

    # Load data
    rgbs, positive_points_per_mask, query_points = load_data(cfg)

    # Load our checkpoint
    n_masks, n_points_per_mask, _ = query_points.shape
    negative_points_per_mask = n_points_per_mask - positive_points_per_mask
    model = load_model(cfg, positive_points_per_mask, negative_points_per_mask)

    # Run inference
    height, width = rgbs.shape[2:]
    target_hw = (height, width)
    logits, trajectories, visibilities, scores = run_inference(model, rgbs, query_points, target_hw)

    # Visualize predictions and save them to wandb
    visualize_and_save_predictions(rgbs, query_points, target_hw, positive_points_per_mask,
                                   logits, trajectories, visibilities, scores, cfg.annot_size, cfg.annot_line_width)


def set_seed(seed):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(cfg):
    cfg.logging.exp_id = f"{cfg.logging.exp_id}" \
                         f"_{cfg.seed}" \
                         f"_{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}"
    print(OmegaConf.to_yaml(cfg))
    wandb.init(
        entity=cfg.logging.wandb.entity,
        project=cfg.logging.wandb.project,
        name=cfg.logging.exp_id,
        group=cfg.logging.exp_id,
        config={
            "cfg": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            "work_dir": os.getcwd(),
            "hydra_cfg": HydraConfig.get(),
        },
    )
    wandb.run.log_code(cfg.logging.wandb.log_code_path)
    wandb.run.summary["work_dir"] = os.path.abspath(os.getcwd())


def load_data(cfg):
    if cfg.query_points_path is not None:
        data_loader = load_demo_data
    else:
        data_loader = load_demo_data_interactive

    return data_loader(
        frames_path=cfg.frames_path,
        query_points_path=cfg.query_points_path,
        frame_stride=cfg.frame_stride,
        longest_side_length=cfg.longest_side_length,
        annot_size=cfg.annot_size,
        annot_line_width=cfg.annot_line_width,
        max_frames=cfg.max_frames,
    )


def load_model(cfg, positive_points_per_mask, negative_points_per_mask):
    cfg.model.positive_points_per_mask = positive_points_per_mask
    cfg.model.negative_points_per_mask = negative_points_per_mask
    model = instantiate(cfg.model)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def run_inference(model, rgbs, query_points, target_hw):
    n_masks, n_points_per_mask, _ = query_points.shape
    n_frames, _, _, _ = rgbs.shape

    # Prepare inputs
    video = {
        "video_id": 0,
        "image": [rgb for rgb in rgbs],
        "target_hw": target_hw,
        "query_points": query_points,
    }
    device = model.device
    for k, v in video.items():
        if isinstance(v, torch.Tensor):
            video[k] = v.to(device)

    # Forward pass
    outputs = model(video)

    # Unpack outputs
    logits_list = outputs['logits']
    trajectories = outputs['trajectories']
    visibilities = outputs['visibilities']
    scores = outputs['scores']

    # Post-process outputs
    logits = torch.stack([torch.zeros_like(logits_list[0])] + logits_list, dim=1)
    assert torch.all(logits[:, 0] == 0), "We always set the background mask to zero logits"

    assert logits.shape[0] == n_frames
    if trajectories is None:
        trajectories = torch.zeros((n_frames, n_masks, n_points_per_mask, 2), dtype=torch.float32)
        visibilities = torch.zeros((n_frames, n_masks, n_points_per_mask), dtype=torch.float32)
        scores = torch.zeros(n_masks, dtype=torch.float32)
    assert trajectories.shape == (n_frames, n_masks, n_points_per_mask, 2)

    # Post process the predictions to set masks to zero for all frames before the query frame
    for i, timestep in enumerate(query_points[:, 0, 0].tolist()):
        timestep = int(timestep)
        logits[:timestep, i + 1] = -1e8

    return logits, trajectories, visibilities, scores


def visualize_and_save_predictions(rgbs, query_points, target_hw, positive_points_per_mask,
                                   logits, trajectories, visibilities, scores, annot_size, annot_line_width):
    predictions_with_trajectories = visualize_predictions(
        images=F.interpolate(
            rgbs.float(),
            target_hw,
            mode='bilinear'
        ).type(torch.uint8),
        step=0,
        query_points=query_points,
        trajectories=trajectories,
        visibilities=visibilities,
        query_masks=None,
        query_scores=torch.ones(len(query_points), dtype=torch.float32),
        sam_masks_logits=logits[:, 1:, :, :].permute(1, 0, 2, 3),
        positive_points_per_mask=positive_points_per_mask,
        annot_size=annot_size,
        annot_line_width=annot_line_width,
    )
    i = 0
    while True:
        img = predictions_with_trajectories[i % len(predictions_with_trajectories)]
        img = img[:, :, ::-1]
        cv2.imshow('sam-pt demo', img)
        sleep(0.2)
        k = cv2.waitKey(1)
        # If user presses 'esc' exit
        if k == 27:
            break
        i += 1
    cv2.destroyAllWindows()


def load_demo_data(frames_path, query_points_path, frame_stride=1, longest_side_length=None,
                   annot_size=8, annot_line_width=4, max_frames=None):
    assert query_points_path is not None

    # Load frames
    frames = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
    frames += sorted(glob.glob(os.path.join(frames_path, '*.png')))
    assert len(frames) > 0, f"No frames found in {frames_path}"

    frames = frames[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]

    rgbs = []
    for frame in frames:
        img = cv2.imread(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)

        if longest_side_length is not None:
            resize_factor = longest_side_length / max(img.shape[1], img.shape[2])
            img = torch.nn.functional.interpolate(img[None], scale_factor=resize_factor)[0]
        else:
            resize_factor = 1.0

        rgbs += [img]
    rgbs = torch.stack(rgbs)

    # Load query points from a file or select them interactively from the first frame
    query_points, num_positive_points = load_query_points(query_points_path, frame_stride, resize_factor)

    return rgbs, num_positive_points, query_points


def load_query_points(query_points_path, frame_stride, resize_factor):
    query_point_timestep_list = []
    query_points_list = []
    with open(query_points_path, 'r') as f:
        lines = f.readlines()
        num_positive_points = int(lines[0].strip())
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            timestep, queries_xy = line.split(';')
            queries_xy = [x.split(',') for x in queries_xy.split()]
            queries_xy = [[float(x), float(y)] for x, y in queries_xy]
            queries_xy = torch.tensor(queries_xy)
            queries_xy = queries_xy * resize_factor

            timestep = int(timestep)
            assert timestep % frame_stride == 0
            timestep = timestep // frame_stride

            query_point_timestep_list += [timestep]
            query_points_list += [queries_xy]

    query_points_xy = torch.stack(query_points_list)
    query_points_timestep = torch.tensor(query_point_timestep_list, dtype=torch.float32)[:, None, None]
    query_points = torch.cat([query_points_timestep.repeat(1, query_points_xy.shape[1], 1), query_points_xy], dim=2)

    return query_points, num_positive_points


def load_demo_data_interactive(frames_path, query_points_path, frame_stride, longest_side_length,
                               annot_size=8, annot_line_width=4, max_frames=None):
    assert query_points_path is None, "Interactive mode does not support loading query points from a file"

    frames = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
    frames += sorted(glob.glob(os.path.join(frames_path, '*.png')))
    assert len(frames) > 0, f"No frames found in {frames_path}"
    frames = frames[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    rgbs = []
    for frame in frames:
        img = cv2.imread(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)

        if longest_side_length is not None:
            resize_factor = longest_side_length / max(img.shape[1], img.shape[2])
            img = torch.nn.functional.interpolate(img[None], scale_factor=resize_factor)[0]
        else:
            resize_factor = 1.0

        rgbs += [img]
    rgbs = torch.stack(rgbs)
    first_frame_img = rgbs[0].cpu().permute(1, 2, 0).flip(-1).numpy()
    img = first_frame_img.copy()
    positive_points_per_mask = None
    negative_points_per_mask = None
    frame_timestep = 0
    while positive_points_per_mask is None:
        x = input('How many positive points? ')
        if x.isdigit():
            positive_points_per_mask = int(x)
    while negative_points_per_mask is None:
        x = input('How many negative points? ')
        if x.isdigit():
            negative_points_per_mask = int(x)
    # Mouse callback function
    positions = []
    timesteps = []
    query_points_xy = []

    def callback(event, x, y, flags, param):
        if event == 1:
            positions.append((x, y))
            point_type = 'positive' if len(positions) <= positive_points_per_mask else 'negative'
            print(f'Added {point_type} point {len(positions)} at ({x}, {y})')

            cmap = plt.get_cmap('tab10')
            cmap_colors = cmap(list(range(10)))
            color = cmap_colors[len(query_points_xy)]
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            color = color[::-1]

            if point_type == 'positive':
                cv2.circle(img, (x, y), annot_size, color, annot_line_width)
            else:
                line_size = annot_size // 2 + 1
                cv2.line(
                    img,
                    (x - line_size, y - line_size),
                    (x + line_size, y + line_size),
                    color, annot_line_width,
                )
                cv2.line(
                    img,
                    (x + line_size, y - line_size),
                    (x - line_size, y + line_size),
                    color, annot_line_width,
                )

            if len(positions) == positive_points_per_mask + negative_points_per_mask:
                timesteps.append(frame_timestep)
                query_points_xy.append(positions.copy())
                positions.clear()
                print(f'Added query points for mask {len(timesteps)}, '
                      f'you can now select points for the next mask '
                      f'or press esc to exit.')
                print()

    cv2.namedWindow('sam-pt demo', flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('sam-pt demo', callback)
    # Mainloop - show the image and collect the data
    while True:
        cv2.imshow('sam-pt demo', img)
        # Wait, and allow the user to quit with the 'esc' key
        k = cv2.waitKey(1)
        # If user presses 'esc' exit
        if k == 27:
            break

    # Write data to a format that can be used by the point tracking script
    str = ''
    str += f'{positive_points_per_mask}\n'
    for i in range(len(timesteps)):
        str += f'{timesteps[i]};'
        str += " ".join([f'{x / resize_factor},{y / resize_factor}' for x, y in query_points_xy[i]])
        str += '\n'

    print("Below is the query point data that you can copy and paste into a file. "
          "You can then use the point tracking script to track the predefined points "
          "by passing the file as the query points path "
          "as `query_points_path=/some/path/to/my_query_points.txt`, if providing a absolute path, "
          "or as `query_points_path='${hydra:runtime.cwd}/my_query_points.txt'`, if providing a relative path.")
    print("------------------")
    print(str)
    print("------------------")
    print()

    query_points_xy = torch.tensor(query_points_xy)
    query_points_timestep = torch.tensor(timesteps, dtype=torch.float32)[:, None, None]
    query_points_timestep = query_points_timestep.repeat(1, query_points_xy.shape[1], 1)
    query_points = torch.cat([query_points_timestep, query_points_xy], dim=2)
    return rgbs, positive_points_per_mask, query_points


if __name__ == '__main__':
    main()
