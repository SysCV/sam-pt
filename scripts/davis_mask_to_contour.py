"""
Script to convert DAVIS mask annotation images to contour images.
Used to prepare figures for the SAM-PT paper.
Note that paths are hardcoded in the script.

Usage: `python -m scripts.davis_mask_to_contour`
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def davis_mask_annotation_image_to_contour_image(input_image_path, output_image_path, contour_radius=5):
    # Open image and convert it to numpy array
    print(f"Input image path: {input_image_path}")
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image.dtype == np.uint8
    assert image.min() >= 0 and image.max() <= 255
    plt.imshow(image)
    plt.show()

    # The number of masks is the number of unique colors in the image
    n_masks = len(np.unique(image.reshape(-1, image.shape[2]), axis=0)) - 1
    print(f"Number of masks: {n_masks}")

    # Take each mask separately and create a binary mask, remember the color of each mask
    masks = []
    colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    assert (colors[0] == [0, 0, 0]).all()
    colors = colors[1:]
    for mask_idx in range(n_masks):
        mask = (image == colors[mask_idx][None, None, :]).all(-1)
        masks.append(mask)

    # Create a contour mask for each mask
    contour_masks = []
    for mask_idx in range(n_masks):
        m_8int = masks[mask_idx].astype(np.uint8)
        dist_transform_fore = cv2.distanceTransform(m_8int, cv2.DIST_L2, 3)
        contour_mask = (dist_transform_fore <= contour_radius) & (dist_transform_fore > 0)
        contour_mask = contour_mask.astype(np.uint8)
        contour_masks.append(contour_mask)
        plt.imshow(contour_mask)
        plt.show()

    # Add contour mask to the image
    output_image = np.zeros_like(image)
    for mask_idx in range(n_masks):
        output_image = np.where(contour_masks[mask_idx][:, :, None] == 1, colors[mask_idx][None, None, :], output_image)

    # Plot the image
    plt.imshow(output_image)
    plt.show()

    # Save the image
    print(f"Output image path: {output_image_path}")
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, output_image)

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


if __name__ == '__main__':
    for i in [1, 7, 16, 23, 32]:
        input_image_path = f"../../04-logs/system-figure/gt--mask-only--frame-{i}--cropped.png"
        output_image_path = f"../../04-logs/system-figure/gt--mask-only--contour--frame-{i}--cropped.png"
        davis_mask_annotation_image_to_contour_image(input_image_path, output_image_path)
    for i in [1, 7, 16, 23, 32]:
        input_image_path = f"../../04-logs/system-figure/gt--mask-only--frame-{i}--cropped.png"
        output_image_path = f"../../04-logs/system-figure/gt--mask-only--contour--thin--frame-{i}--cropped.png"
        davis_mask_annotation_image_to_contour_image(input_image_path, output_image_path, contour_radius=2)
