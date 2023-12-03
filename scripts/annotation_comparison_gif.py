"""
python scripts/annotation_comparison_gif.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio
from PIL import Image
from tqdm import tqdm


def create_gif(results_dir, annotations_dir, images_dir, output_gif_path):
    # Get a sorted list of image files and annotation files
    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.png')])
    images_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.png')])

    # Check if both folders have the same number of files
    assert len(result_files) == len(annotation_files) == len(images_files)

    # Create a list to store concatenated images
    concat_images = []

    for res_file, img_file, ann_file in tqdm(list(zip(result_files, images_files, annotation_files))):
        # Open the images
        result = Image.open(os.path.join(results_dir, res_file))
        image = Image.open(os.path.join(images_dir, img_file))
        annotation = Image.open(os.path.join(annotations_dir, ann_file))

        # Make sure the images can be concatenated
        assert image.size == annotation.size == result.size, "Image sizes do not match."

        # Concatenate the images vertically
        total_height = image.size[1] + annotation.size[1] + result.size[1]
        combined_image = Image.new('RGB', (image.size[0], total_height))
        combined_image.paste(image, (0, 0))
        combined_image.paste(annotation, (0, image.size[1]))
        combined_image.paste(result, (0, image.size[1] + annotation.size[1]))

        # Add to list of concatenated images
        concat_images.append(combined_image)

    # Save the frames as a GIF
    imageio.mimsave(output_gif_path, concat_images, duration=0.5, loop=0)

    print(f"GIF created at {output_gif_path}")


def create_gif_per_video(video, results_path, annotations_path, images_path):
    print(f"Creating GIF for {video}")
    result_path = os.path.join(results_path, video)
    annotation_path = os.path.join(annotations_path, video)
    image_path = os.path.join(images_path, video)
    output_gif_path = os.path.join(results_path, video + ".gif")
    create_gif(result_path, annotation_path, image_path, output_gif_path)


if __name__ == '__main__':
    # results_path = "/mnt/terra/xoding/eth-master-thesis/08-logs-september/bdd100k-results/K9.000--debug--cotracker-0--1-1024/"
    # annotations_path = "/mnt/terra/xoding/eth-master-thesis/08-logs-september/bdd100k-results/vos/val/Annotations/"
    # images_path = "/mnt/terra/xoding/eth-master-thesis/08-logs-september/bdd100k-results/vos/val/JPEGImages/"

    results_path = "outputs/K9.000--debug--cotracker-0--1-1024/eval_BDD100K_val"
    annotations_path = "data/bdd100k/vos/val/Annotations"
    images_path = "data/bdd100k/vos/val/JPEGImages"

    videos = [video for video in os.listdir(results_path) if not video.endswith(".gif") and not "." in video]

    with ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor
        future_to_video = {
            executor.submit(create_gif_per_video, video, results_path, annotations_path, images_path): video for video
            in videos}

        # Process the futures as they complete
        for future in tqdm(as_completed(future_to_video), total=len(videos), desc="Processing videos", unit="video"):
            video = future_to_video[future]
            try:
                future.result()
            except Exception as exc:
                print(f'{video} generated an exception: {exc}')

    print("All GIFs have been created.")
