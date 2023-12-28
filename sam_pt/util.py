import os
import pathlib
import random
import time
from datetime import datetime
from enum import IntEnum
from typing import List, Any, Optional, Tuple

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


class Object(object):
    """
    Empty class for use as a default placeholder object.
    """
    pass


def get_str_formatted_time() -> str:
    """
    Returns the current time in the format of '%Y.%m.%d_%H.%M.%S'.
    Returns
    -------
    str
        The current time
    """
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')


HORSE = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;poi;'. ;   |          )   \\ | ;;;;;;
          ,;nts;\\/   |.        /   /` | ';;;*;
           ;;;**;/    |/       /   /__/   ';;;
           '*;;;/     |       /    |      ;*;
                `""""`        `""""`     ;'"""


def nice_print(msg, last=False):
    """
    Print a message in a nice format.
    Parameters
    ----------
    msg : str
        The message to be printed
    last : bool, optional
        Whether to print a blank line at the end, by default False
    Returns
    -------
    None
    """
    print()
    print("\033[0;35m" + msg + "\033[0m")
    if last:
        print()


class AttrDict(dict):
    """
    A dictionary class that can be accessed with attributes.
    Note that the dictionary keys must be strings and
    follow attribute naming rules to be accessible as attributes,
    e.g., the key "123xyz" will give a syntax error.
    Usage:
    ```
    x = AttrDict()
    x["jure"] = "mate"
    print(x.jure)
    # "mate"
    x[123] = "abc"
    x.123
    # SyntaxError: invalid syntax
    ```
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def ensure_dir(dirname):
    """
    Ensure that a directory exists. If it doesn't, create it.
    Parameters
    ----------
    dirname : str or pathlib.Path
        The directory, the existence of which will be ensured
    Returns
    -------
    None
    """
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def batchify_list(datapoints: List[Any], batch_size: int, batch_post_processing_fn=None) -> List[Any]:
    """
    Batchify a list of anything into batches of given batch size,
    apply an optional post-processing function on top of each batch.

    Parameters
    ----------
    datapoints: List[Any]
        The list of datapoints.
    batch_size: int
        The size of batches.
    batch_post_processing_fn: function, optional
        An optional post-processing function that will be applied on top of the batches.

    Returns
    -------
    A list of batches. The batch might be anything after post-processing, thus List[Any] is returned.
    """
    assert batch_size > 0
    num_batches = math.ceil(len(datapoints) / batch_size)
    batches = [datapoints[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    assert len(datapoints) == sum([len(batch) for batch in batches])
    if batch_post_processing_fn is not None:
        batches = [batch_post_processing_fn(batch) for batch in batches]
    return batches


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Source:
    https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/optimization.py#L75

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer for which to schedule the learning rate.
    num_warmup_steps: int
        The number of steps for the warmup phase.
    num_training_steps: int
        The total number of training steps.
    last_epoch: int, optional (default=-1)
        The index of the last epoch when resuming training.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LambdaLR
        A learning rate scheduler with the appropriate schedule.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def zip_strict(*lists):
    """
    Zip lists, with strict equality of length.

    Given an arbitrary number of lists, zip_strict ensures that all lists have the same length before returning the
    zipped object.

    Parameters
    ----------
    *lists : list of lists
        An arbitrary number of lists.

    Returns
    -------
    zipped_lists : zip object
        An object representing a sequence of tuples, where the i-th tuple contains the i-th element from each of the
        input lists.

    Raises
    ------
    AssertionError
        If not all input lists have the same length.

    Examples
    --------
    >>> list(zip_strict([1, 2, 3], [4, 5, 6]))
    [(1, 4), (2, 5), (3, 6)]

    >>> list(zip_strict([1, 2, 3], [4, 5, 6], [7, 8, 9]))
    [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

    >>> list(zip_strict([1, 2, 3], [4, 5]))
    AssertionError:
    """
    lengths = [len(_list) for _list in lists]
    assert all([length == lengths[0] for length in lengths]), "All input lists must have the same length."
    return zip(*lists)


def seed_all(seed):
    """
    Seed all random number generators.

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_video_to_wandb(log_key: str, frames: List[np.ndarray],
                       step: Optional[int] = None, commit: Optional[bool] = None,
                       fmt: str = "gif", fps: int = 4) -> None:
    """
    Log a video to Wandb.

    Parameters
    ----------
    log_key : str
        The key under which the video will be logged.
    frames : List[np.ndarray]
        The frames of the video to be logged. Each frame should be a 3D numpy array.
    step : int, optional
        The step at which to log the video. If None, the step is automatically determined by wandb.
    commit : bool, optional
        If true, increments the logging step and saves the data to the wandb server.
    fmt : str, default 'gif'
        The format of the video to be logged.
    fps : int, default 4
        The frames per second of the video to be logged.

    Returns
    -------
    None
    """
    frames_4d = np.stack(frames, axis=0)
    frames_4d = frames_4d.transpose((0, 3, 1, 2))
    wandb.log({log_key: wandb.Video(frames_4d, format=fmt, fps=fps)}, step=step, commit=commit)


class PointVisibilityType(IntEnum):
    """
    Enum to distinguish different point (in)visibility types, for visualization debugging purposes.
    VISIBILITY: point is visible
    INVISIBILITY: point is not visible
    REINIT_FAILED: point is not visible because the point reinitialization failed, e.g. because all points were occluded
    OUTSIDE_FRAME: point is outside the frame
    PATCH_NON_SIMILAR: point is not visible because its patch is not similar enough to its previous patch
    REJECTED_AFTER_PATCH_WAS_NON_SIMILAR: point is rejected in after it its patch was not similar enough to its previous
    """
    VISIBLE = 1
    INVISIBLE = 0
    REINIT_FAILED = -1
    OUTSIDE_FRAME = -2
    PATCH_NON_SIMILAR = -3
    REJECTED_AFTER_PATCH_WAS_NON_SIMILAR = -4


VISIBILITY_TO_COLOR = {
    PointVisibilityType.VISIBLE.value: None,
    PointVisibilityType.INVISIBLE.value: (255, 0, 0),
    PointVisibilityType.REJECTED_AFTER_PATCH_WAS_NON_SIMILAR.value: (255, 255, 0),
    PointVisibilityType.OUTSIDE_FRAME.value: (236, 240, 241),
    PointVisibilityType.PATCH_NON_SIMILAR.value: (0, 0, 0),
    PointVisibilityType.REINIT_FAILED.value: (255, 255, 255),
}


def add_mask_to_frame(frame: np.ndarray, mask: Optional[np.ndarray], color: Tuple[float, float, float],
                      alpha: float = 0.2) -> np.ndarray:
    """
    Add a colored mask to a frame.

    Parameters
    ----------
    frame : np.ndarray
        A 3D numpy array representing the frame. The frame is expected to have three color channels.
    mask : np.ndarray, optional
        A 2D numpy array representing the mask. If None, the original frame is returned.
    color : Tuple[float, float, float]
        A tuple of three floats representing the RGB color of the mask.
    alpha : float, default 0.2
        The alpha blending value. A higher value means the mask is more transparent.

    Returns
    -------
    masked_frame : np.ndarray
        A 3D numpy array representing the frame with the mask applied.
    """
    if mask is None:
        return frame

    frame = frame / 255

    overlay = np.ones((mask.shape[0], mask.shape[1], 3))
    overlay *= color
    overlay = np.where(mask[..., None] > 0, overlay, frame)

    masked_frame = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)
    masked_frame = masked_frame * 255
    masked_frame = masked_frame.astype(np.uint8)
    return masked_frame


def visualize_predictions(
        images,
        step,
        query_points,
        trajectories,
        visibilities,
        query_masks,
        query_scores,
        sam_masks_logits,
        sam_per_frame_scores=None,
        additional_log_images=None,
        additional_frame_annotations=None,
        query_point_color=(0, 0, 255),
        visibility_to_color=VISIBILITY_TO_COLOR,
        positive_points_per_mask=1e9,
        fps=5,
        annot_size=8,
        annot_line_width=4,
        visualize_query_masks=True,
        contour_radius=3,
        verbose=True,
        log_fmt="mp4",
):
    """
    Visualize predictions of query masks, predicted masks, and trajectories onto images and log them to wandb.

    Parameters
    ----------
    images : torch.Tensor
        The input frames to visualize, expected shape is (n_frames, channels, height, width).
    step : int
        The current step of training/validation. Please note that wandb requires steps to be monotonically increasing,
        otherwise the logged data might not be saved and displayed correctly.
    query_points : torch.Tensor
        List of query points for each mask, expected shape is (n_masks, n_points, 3).
    trajectories : torch.Tensor
        The predicted trajectories for each mask in each frame, expected shape is (n_frames, n_masks, n_points, 2).
    visibilities : torch.Tensor
        The visibility status of each point in each trajectory, expected shape is (n_frames, n_masks, n_points).
    query_masks : torch.Tensor
        Query masks for each frame, expected shape is (n_masks, height, width).
    query_scores : torch.Tensor
        The score for each query mask, expected shape is (n_masks,).
    sam_masks_logits : torch.Tensor
        The predicted mask logits for each frame, expected shape is (n_masks, n_frames, height, width).
    sam_per_frame_scores : torch.Tensor, optional
        The predicted IoU score for each mask in each frame. Default is None. Expected shape is (n_frames, n_masks).
    additional_log_images : torch.Tensor, optional
        Additional frames to log to wandb. Default is None. Expected shape is (n_frames, channels, height, width).
    additional_frame_annotations : List[str], optional
        Additional annotations to add to each frame. Default is None.
    query_point_color : tuple, optional
        The color to use for the query points. Default is (0, 0, 255).
    visibility_to_color : dict, optional
        A dictionary mapping visibility type to color. Default is VISIBILITY_TO_COLOR.
    positive_points_per_mask : float, optional
        The number of positive points to annotate in each mask, or None if all points are positive. Default is None.
    fps : int, optional
        The frames per second to use when logging videos to wandb. Default is 5.
    annot_size : int, optional
        The size of the annotation point. Default is 8.
    annot_line_width : int, optional
        The line width of the annotation. Default is 4.
    visualize_query_masks : bool, optional
        If True, visualize the query masks. Default is True.
    contour_radius : int, optional
        The radius for the contour around each mask. Default is 3.
    verbose : bool, optional
        If True, log verbose visualisations to wandb. Takes more time. Default is True.

    Returns
    -------
    List[ndarray]
        List of images with the predictions and trajectories visualized.

    Notes
    -----
    This function uses wandb for logging the images and videos, make sure to have wandb initialized.
    """
    print(f"Visualizing predictions to wandb, for step {step}...")
    start_time = time.time()

    sam_masks = sam_masks_logits > 0

    n_frames = len(images)
    n_masks = len(query_points)
    cmap = plt.get_cmap('tab10')
    cmap_colors = cmap(list(range(n_masks)))

    frame_debug_text = [[] for _ in range(n_frames)]
    for frame_idx in range(n_frames):
        frame_debug_text[frame_idx] += [f"Frame {frame_idx}"]
        if sam_per_frame_scores is not None:
            for mask_idx in range(n_masks):
                frame_debug_text[frame_idx] += [
                    f"[{mask_idx}] Sam IoU prediction: "
                    f"{sam_per_frame_scores[frame_idx][mask_idx] * 100:.2f}"
                ]
    if additional_frame_annotations is not None:
        for frame_idx in range(n_frames):
            frame_debug_text[frame_idx] += additional_frame_annotations[frame_idx]
    frame_debug_text = ["\n".join(frame_debug_text[frame_idx]) for frame_idx in range(n_frames)]

    def get_point_color(mask_idx: int, is_query_point: bool, visibility_value: int):
        # if is_query_point:
        #     return query_point_color

        if visibility_value == PointVisibilityType.VISIBLE:
            # Get matplotlib colormap tab10 and assign the corresponding mask_idx color to the point
            color = cmap_colors[mask_idx]
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            return color

        # If the point is not visible, return the color that corresponds to the (in)visibility type
        return visibility_to_color[int(visibility_value.item())]

    # Extract the query masks from Sam if they are not given
    if query_masks is None:
        query_masks = torch.stack([
            sam_masks[mask_idx, int(query_points[mask_idx][0][0])]
            for mask_idx in range(n_masks)
        ])

    # 1. Visualize the input frames
    frames = images.permute(0, 2, 3, 1).cpu().numpy()
    if verbose:
        log_video_to_wandb("verbose/input-only", frames, fps=fps, step=step, fmt=log_fmt)

    # 2. Visualize the query masks
    if visualize_query_masks:
        query_masks = query_masks.cpu().numpy()
        # 2.1. version 1 - using wandb mask overlays
        for mask_idx, mask_query_points in enumerate(query_points):
            query_timestep = int(mask_query_points[0][0])
            frame = frames[query_timestep].copy()
            for query_point in mask_query_points:
                frame = cv2.circle(frame, (int(query_point[1]), int(query_point[2])), annot_size,
                                   query_point_color, annot_line_width)
            query_mask = query_masks[mask_idx, :, :]
            wandb_masks = {"mask2former_prediction": {"mask_data": query_mask}}
            wandb_caption = f"mask={mask_idx}: m2f_score={query_scores[mask_idx].item():.3f} query_frame={query_timestep}"
            wandb_image = wandb.Image(frame, caption=wandb_caption, masks=wandb_masks)
            if verbose:
                wandb.log({f"verbose/query-proposals/mask-{mask_idx}": wandb_image}, step=step)
        # 2.1. version 2 - using cv2 to create the mask overlay by hand
        for mask_idx, mask_query_points in enumerate(query_points):
            query_timestep = int(mask_query_points[0][0])
            frame = frames[query_timestep].copy() / 255
            query_mask = query_masks[mask_idx, :, :]
            overlay = np.ones_like(query_mask[..., None], dtype=np.float64)
            overlay = overlay * np.array([[[0.3, 0.6, 0.1]]])
            overlay = np.where(query_mask[..., None] > 0, overlay, frame)
            overlaid_frame = cv2.addWeighted(frame, 0.1, overlay, 0.9, 0)
            overlaid_frame = overlaid_frame * 255
            overlaid_frame = overlaid_frame.astype(np.uint8)
            for query_point in mask_query_points:
                overlaid_frame = cv2.circle(overlaid_frame, (int(query_point[1]), int(query_point[2])), annot_size,
                                            query_point_color, annot_line_width)
            wandb_caption = f"mask={mask_idx}: m2f_score={query_scores[mask_idx].item():.3f} query_frame={query_timestep}"
            wandb_image = wandb.Image(overlaid_frame, caption=wandb_caption)
            wandb.log({f"query-proposals/mask-{mask_idx}": wandb_image}, step=step)

    # 3.1. Visualize the predicted masks
    trajectories = torch.nan_to_num(trajectories, nan=-1.)
    sam_frame_masks = list(zip(*sam_masks))
    masked_images = []
    n_frames = len(frames)
    n_masks = len(query_points)
    for frame_idx in range(n_frames):
        frame = frames[frame_idx, :, :, :]
        palette = cmap_colors[:, :3]
        masked_image = frame.copy()
        for mask_idx in range(n_masks):
            masked_image = add_mask_to_frame(masked_image, sam_frame_masks[frame_idx][mask_idx],
                                             palette[mask_idx], alpha=0.5)
            m_8int = sam_frame_masks[frame_idx][mask_idx].numpy().astype(np.uint8)
            dist_transform_fore = cv2.distanceTransform(m_8int, cv2.DIST_L2, 3)
            dist_transform_back = cv2.distanceTransform(1 - m_8int, cv2.DIST_L2, 3)
            dist_map = dist_transform_fore - dist_transform_back
            contour_mask = (dist_map <= contour_radius) & (dist_map >= -contour_radius)
            contour_mask = contour_mask.astype(np.uint8)
            masked_image = add_mask_to_frame(masked_image, contour_mask, [1, 1, 1], alpha=0.1)
        masked_image = put_debug_text_onto_image(masked_image, frame_debug_text[frame_idx])
        masked_images += [masked_image]
    if verbose:
        log_video_to_wandb("verbose/predictions-only", masked_images, fps=fps, step=step, fmt=log_fmt)

    # 3.2. Add the predicted trajectories on top of the predicted masks
    for frame_idx in range(n_frames):
        masked_image = masked_images[frame_idx].copy()
        for mask_idx in range(n_masks):
            query_timestep = int(query_points[mask_idx][0][0])
            for i, (point, vis) in enumerate(
                    zip(trajectories[frame_idx, mask_idx], visibilities[frame_idx, mask_idx])):
                x, y = (int(point[0]), int(point[1]))
                if vis != PointVisibilityType.VISIBLE:
                    # Color all invisible points the same, for paper and demos
                    vis = torch.tensor(PointVisibilityType.INVISIBLE)
                c = get_point_color(mask_idx, query_timestep == frame_idx, vis)
                if positive_points_per_mask is not None and i < positive_points_per_mask:
                    masked_image = cv2.circle(masked_image, (x, y), annot_size, c, annot_line_width)
                else:
                    line_size = annot_size // 2 + 1
                    masked_image = cv2.line(
                        masked_image,
                        (x - line_size, y - line_size),
                        (x + line_size, y + line_size),
                        c, annot_line_width,
                    )
                    masked_image = cv2.line(
                        masked_image,
                        (x + line_size, y - line_size),
                        (x - line_size, y + line_size),
                        c, annot_line_width,
                    )
                masked_image = cv2.putText(masked_image, f"{i:03}", (int(point[0]), int(point[1])),
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(250, 225, 100))
        masked_images[frame_idx] = masked_image
    if verbose:
        log_video_to_wandb("verbose/predictions-with-trajectories", masked_images, fps=fps, step=step, fmt=log_fmt)

    # 4. Visualize the input frames with the predicted trajectories
    frames_with_trajectories = frames.copy()
    for mask_idx in range(len(trajectories[0])):
        query_timestep = int(query_points[mask_idx][0][0])
        for frame_idx in range(n_frames):
            for i, (point, vis) in enumerate(zip(trajectories[frame_idx, mask_idx],
                                                 visibilities[frame_idx, mask_idx])):
                x, y = int(point[0]), int(point[1])
                c = get_point_color(mask_idx, query_timestep == frame_idx, vis)
                if positive_points_per_mask is not None and i < positive_points_per_mask:
                    frames_with_trajectories[frame_idx] = cv2.circle(
                        frames_with_trajectories[frame_idx], (x, y), annot_size, c, annot_line_width,
                    )
                else:
                    line_size = annot_size // 2 + 1
                    frames_with_trajectories[frame_idx] = cv2.line(
                        frames_with_trajectories[frame_idx],
                        (x - line_size, y - line_size),
                        (x + line_size, y + line_size),
                        c, annot_line_width,
                    )
                    frames_with_trajectories[frame_idx] = cv2.line(
                        frames_with_trajectories[frame_idx],
                        (x + line_size, y - line_size),
                        (x - line_size, y + line_size),
                        c, annot_line_width,
                    )
                frames_with_trajectories[frame_idx] = cv2.putText(
                    frames_with_trajectories[frame_idx], f"{i:03}", (int(point[0]), int(point[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                    color=(250, 225, 100)
                )
    for frame_idx in range(n_frames):
        frames_with_trajectories[frame_idx] = put_debug_text_onto_image(
            frames_with_trajectories[frame_idx], frame_debug_text[frame_idx]
        )
    if verbose:
        log_video_to_wandb("verbose/input-with-trajectories", frames_with_trajectories, fps=fps, step=step, fmt=log_fmt)

    # 5. Visualize the input frames with the predicted trajectories and the predicted masks
    concatenated = [
        np.concatenate((frame, pred_masked_frame), axis=0)
        for frame, pred_masked_frame
        in zip(frames_with_trajectories, masked_images)
    ]
    additional_log_images = (
        additional_log_images.permute(0, 2, 3, 1).cpu().numpy()
        if additional_log_images is not None else None
    )
    if additional_log_images is not None:
        concatenated = [
            np.concatenate((additional_log_frame, frame), axis=0)
            for additional_log_frame, frame
            in zip(additional_log_images, concatenated)
        ]
    log_video_to_wandb("predictions/sam_video_masks", concatenated, fps=fps, step=step, fmt=log_fmt)

    print("Done visualizing predictions. Time taken: {:.2f}s".format(time.time() - start_time))

    predictions_with_trajectories = masked_images
    return predictions_with_trajectories


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
