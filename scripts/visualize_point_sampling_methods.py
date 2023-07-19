"""
Script to visualize the different point sampling methods for the SAM-PT paper.

Usage: `python -m scripts.visualize_point_sampling_methods`
"""
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial

from sam_pt.utils.query_points import extract_corner_points
from sam_pt.utils.query_points import extract_kmedoid_points
from sam_pt.utils.query_points import extract_mixed_points
from sam_pt.utils.query_points import extract_random_mask_points
from sam_pt.utils.util import seed_all


def mixed_point_id_to_marker_and_rescale(n_points, point_id):
    n_kmedoid = n_points // 4
    n_shi_tomasi = n_points // 3
    if point_id < n_kmedoid:
        return "o", 1
    elif point_id < n_kmedoid + n_shi_tomasi:
        return "*", 3
    else:
        return "v", 1.2


def visualize_point_sampling_methods(
        rgb_image_path,
        annotation_image_path,
        output_image_path,
        point_sampling_method_name="kmedoids",
        n_points=8,
        seed=72,
):
    # Open image and convert it to numpy array
    image = cv2.imread(rgb_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image.dtype == np.uint8
    assert image.min() >= 0 and image.max() <= 255
    plt.imshow(image)
    plt.show()

    annotation_image = cv2.imread(annotation_image_path)
    annotation_image = cv2.cvtColor(annotation_image, cv2.COLOR_BGR2RGB)
    assert annotation_image.dtype == np.uint8
    assert annotation_image.min() >= 0 and annotation_image.max() <= 255
    plt.imshow(annotation_image)
    plt.show()

    # The number of masks is the number of unique colors in the image
    n_masks = len(np.unique(annotation_image.reshape(-1, annotation_image.shape[2]), axis=0)) - 1
    print(f"Number of masks: {n_masks}")

    # Prepare the point sampling methods
    point_sampling_methods = {
        "kmedoids": {
            "function": extract_kmedoid_points,
            "marker": ["o" for _ in range(n_points)],
            "rescale": [1 for _ in range(n_points)],
        },
        "shi-tomasi": {
            "function": partial(extract_corner_points, image=torch.from_numpy(image).permute(2, 0, 1)),
            "marker": ["*" for _ in range(n_points)],
            "rescale": [3 for _ in range(n_points)],
        },
        "random": {
            "function": extract_random_mask_points,
            "marker": ["v" for _ in range(n_points)],
            "rescale": [1.2 for _ in range(n_points)]
        },
        "mixed": {
            "function": lambda mask, n_points_to_select: extract_mixed_points(
                query_masks=mask[None, ...],
                query_points_timestep=torch.zeros(n_masks),
                images=torch.from_numpy(image).permute(2, 0, 1)[None, ...],
                n_points=n_points_to_select,
            )[0],
            "marker": [mixed_point_id_to_marker_and_rescale(n_points, point_id)[0] for point_id in range(n_points)],
            "rescale": [mixed_point_id_to_marker_and_rescale(n_points, point_id)[1] for point_id in range(n_points)]
        },
    }

    # Take each mask separately and create a binary mask, remember the color of each mask
    masks = []
    colors = np.unique(annotation_image.reshape(-1, annotation_image.shape[2]), axis=0)
    assert (colors[0] == [0, 0, 0]).all()
    colors = colors[1:]
    for mask_idx in range(n_masks):
        mask = (annotation_image == colors[mask_idx][None, None, :]).all(-1)
        masks.append(mask)

    # Sample points from each mask
    mask_points = []
    for mask_idx in range(n_masks):
        seed_all(seed + 3)
        mask = torch.from_numpy(masks[mask_idx]).bool()
        points = point_sampling_methods[point_sampling_method_name]["function"](mask=mask, n_points_to_select=n_points)
        mask_points.append(points)

    # Create a contour mask for each mask
    contour_radius = 3
    contour_masks = []
    for mask_idx in range(n_masks):
        m_8int = masks[mask_idx].astype(np.uint8)
        dist_transform_fore = cv2.distanceTransform(m_8int, cv2.DIST_L2, 3)
        contour_mask = (dist_transform_fore <= contour_radius) & (dist_transform_fore > 0)
        contour_mask = contour_mask.astype(np.uint8)
        contour_masks.append(contour_mask)

    # Add contour and sampled points to the image
    output_image = np.zeros_like(annotation_image)
    for mask_idx in range(n_masks):
        output_image = np.where(contour_masks[mask_idx][:, :, None] == 1, colors[mask_idx][None, None, :], output_image)
    h, w, dpi = output_image.shape[0], output_image.shape[1], 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    plt.imshow(output_image)
    for mask_idx in range(n_masks):
        for point_idx in range(n_points):
            plt.scatter(
                x=mask_points[mask_idx][point_idx, 0],
                y=mask_points[mask_idx][point_idx, 1],
                s=90 * point_sampling_methods[point_sampling_method_name]["rescale"][point_idx],
                c=(colors[mask_idx][None, :] * 1.8 / 255).clip(min=0, max=1),
                linewidths=0,
                marker=point_sampling_methods[point_sampling_method_name]["marker"][point_idx]
            )
    plt.axis("off")
    plt.tight_layout(pad=0)
    print(f"Output image path: {output_image_path}")
    plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0)
    plt.show()

    # Save also RGBA image
    output_image = cv2.imread(output_image_path)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(output_image)
    a = 255 - (output_image == np.array([0, 0, 0])[None, None, :]).all(-1).astype(np.uint8) * 255
    output_image = cv2.merge([r, g, b, a], 4)
    print(f"RGBA image path: {output_image_path}.rgba.png")
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(output_image_path + ".rgba.png", output_image)
    print("Done.")


def main(args):
    n_points = args.n_points
    for psm in args.point_sampling_methods:
        output_image_path = f"{args.output_path_prefix}--point-sampling-method-{psm}.png"
        visualize_point_sampling_methods(
            rgb_image_path=args.rgb_path,
            annotation_image_path=args.annotation_path,
            output_image_path=output_image_path,
            point_sampling_method_name=psm,
            n_points=n_points,
            seed=args.seed,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', type=int, default=8)
    parser.add_argument('--rgb_path', type=str,
                        default="../../04-logs/system-figure/horse-input--frame-16--cropped.png")
    parser.add_argument('--annotation_path', type=str,
                        default="../../04-logs/system-figure/gt--mask-only--frame-16--cropped.png")
    parser.add_argument('--output_path_prefix', type=str,
                        default="../../04-logs/system-figure/gt--mask-only--frame-16--cropped")
    parser.add_argument('--point_sampling_methods', type=str, nargs='+',
                        default=["kmedoids", "shi-tomasi", "random", "mixed"])
    parser.add_argument('--seed', type=int, default=72)
    args = parser.parse_args()
    main(args)
