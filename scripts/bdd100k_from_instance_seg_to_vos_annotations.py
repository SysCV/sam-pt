"""
To create the VOS annotations from the instance segmentation annotations, run:
```bash
# Prepare directories
mkdir -p data/bdd100k/vos/val/{Annotations,JPEGImages}

# Copy JPEGImages
cp -r data/bdd100k/images/seg_track_20/val/* data/bdd100k/vos/val/JPEGImages/

# Create the Annotations
python -m scripts.bdd100k_from_instance_seg_to_vos_annotations

# Link the chunks
# e.g., data/bdd100k/vos/val/JPEGImages/b1c66a42-6f7d68ca-chunk2 -> b1c66a42-6f7d68ca/
find data/bdd100k/vos/val/Annotations -type d -name "*-chunk*" | sed 's/Annotations/JPEGImages/' | while read -r src; do
    tgt=$(basename "$src" | sed 's/-chunk.*//')
    rm $src
    ln -s "$tgt" "$src"
done
```
"""
import json
import os

import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

np.random.seed(72)
palette = (np.multiply(np.random.rand(768), 255).astype(np.uint8).tolist())
palette[:3] = [0, 0, 0]


def remap_ids(ids):
    # Find the unique IDs and their new remapped positions
    unique_ids, inverse_indices = np.unique(ids, return_inverse=True)

    # Reshape the inverse_indices to the shape of the original IDs array
    remapped_ids = inverse_indices.reshape(ids.shape)

    return remapped_ids


def process_video(video_name, objects_per_chunk=100):
    print(f"Processing video {video_name}")
    frames = sorted(os.listdir(os.path.join(videos_path, video_name)))
    bitmasks = []
    for frame_name in frames:
        frame_path = os.path.join(videos_path, video_name, frame_name)
        bitmask = np.array(Image.open(frame_path))
        bitmasks.append(bitmask)
    bitmasks = np.stack(bitmasks)
    annotation_ids = (bitmasks[:, :, :, 2].astype(np.uint32) << 8) + bitmasks[:, :, :, 3]
    unique_ids = np.unique(annotation_ids).size
    print(f"Video {video_name} is loaded, it has {unique_ids} unique objects")

    annotation_ids = annotation_ids * (bitmasks[:, :, :, 1] & 1 == 0)  # Remove ignored instances
    annotation_ids = annotation_ids * (bitmasks[:, :, :, 1] & 2 == 0)  # Remove crowd instances
    # annotation_ids = annotation_ids * (bitmasks[:, :, :, 1] & 4 == 0)  # Remove occluded instances
    # annotation_ids = annotation_ids * (bitmasks[:, :, :, 1] & 8 == 0)  # Remove truncated instances
    unique_ids_old = unique_ids
    unique_ids = np.unique(annotation_ids).size
    print(f"Video {video_name} is filtered by ignored and crowd instances, "
          f"it has {unique_ids_old:>5d} --> {unique_ids:>5d} unique objects now")

    # # Randomly select max_objects objects
    # if unique_ids > max_objects:
    #     np.random.seed(72)
    #     selected_ids = np.random.choice(np.sort(np.unique(annotation_ids))[1:], max_objects, replace=False)
    #     annotation_ids = np.where(np.isin(annotation_ids, selected_ids), annotation_ids, 0)
    #     unique_ids_old = unique_ids
    #     unique_ids = np.unique(annotation_ids).size
    #     print(f"Video {video_name} is filtered by max_objects, "
    #           f"it has {unique_ids_old:>5d} --> {unique_ids:>5d} unique objects now")

    # # Select the first max_objects objects
    # if unique_ids > max_objects:
    #     selected_ids = np.sort(np.unique(annotation_ids))[1:max_objects + 1]
    #     annotation_ids = np.where(np.isin(annotation_ids, selected_ids), annotation_ids, 0)
    #     unique_ids_old = unique_ids
    #     unique_ids = np.unique(annotation_ids).size
    #     print(f"Video {video_name} is filtered by max_objects, "
    #           f"it has {unique_ids_old:>5d} --> {unique_ids:>5d} unique objects now")

    # Split the objects into chunks of objects_per_chunk objects
    annotation_ids_unique = np.unique(annotation_ids)[1:]
    for chunk_id in range(math.ceil(annotation_ids_unique.size / objects_per_chunk)):
        chunk_name = f"{video_name}-chunk{chunk_id + 1}" if chunk_id > 0 else video_name
        chunk = annotation_ids_unique[chunk_id * objects_per_chunk:(chunk_id + 1) * objects_per_chunk]
        print(f"Processing {chunk_name}, it has {chunk.size} objects: {chunk}")

        # Select the objects in the chunk
        annotation_ids_chunk = np.where(np.isin(annotation_ids, chunk), annotation_ids, 0)
        unique_ids = np.unique(annotation_ids_chunk).size

        # Remap annotation IDs to be continuous
        remapped_annotation_ids = remap_ids(annotation_ids_chunk)
        assert np.unique(remapped_annotation_ids).size == unique_ids
        assert np.unique(remapped_annotation_ids).size == remapped_annotation_ids.max() + 1
        print(f"Video {video_name} is remapped")

        output_dir = os.path.join(output_path, chunk_name)
        os.makedirs(output_dir, exist_ok=True)
        assert unique_ids <= 255, "The number of unique objects should be less than 255 to use uint8"
        for frame_id, frame_name in enumerate(frames):
            x = Image.fromarray(remapped_annotation_ids[frame_id].astype(np.uint8), mode="P")
            x.putpalette(palette)
            x.save(os.path.join(output_dir, frame_name))
        print(f"Video {video_name} is saved")


def sanity_check(output_path, rles_path):
    for i, video_json_name in enumerate(tqdm(sorted([vp for vp in os.listdir(rles_path) if vp.endswith("json")]))):
        video_name = video_json_name.replace(".json", "")
        # if i < 15:
        #     print(f"Skipping video {video_name}")
        #     continue
        with open(os.path.join(rles_path, video_json_name), "r") as fp:
            video = json.load(fp)
        df = pd.DataFrame([
            (label["category"], label["id"])
            for frame in video["frames"]
            for label in frame["labels"]
        ], columns=["cat", "id"])
        assert df[~df.duplicated()].groupby("id").count().max().item() == 1

        annotation_ids = [
            np.array(Image.open(os.path.join(output_path, video_name, frame_name)))
            for frame_name in sorted(os.listdir(os.path.join(output_path, video_name)))
        ]
        annotation_ids = np.stack(annotation_ids)
        assert np.unique(annotation_ids).size == annotation_ids.max() + 1
        if np.unique(annotation_ids).size != df.id.unique().size + 1:
            print(f"Video {video_name} has {np.unique(annotation_ids).size} unique objects, "
                  f"but RLE has {df.id.unique().size + 1} unique objects")
            # breakpoint()
        else:
            assert np.unique(annotation_ids).size == df.id.unique().size + 1

        print(f"Unique objects for video {i:02d}: {df.id.unique().size}")


if __name__ == '__main__':
    videos_path = "data/bdd100k/labels/seg_track_20/bitmasks/val"
    output_path = "data/bdd100k/vos/val/Annotations"
    video_names = sorted([name for name in os.listdir(videos_path) if os.path.isdir(os.path.join(videos_path, name))])

    # Create the VOS annotations
    process_map(process_video, video_names, chunksize=1)
    print("Done creating VOS annotations")

    # # Sanity check that the number of objects in the VOS annotations is the same as in the RLEs
    # print("Sanity check that the number of objects in the VOS annotations is the same as in the RLEs")
    # rles_path = "data/bdd100k/labels/seg_track_20/rles/val"
    # sanity_check(output_path, rles_path)
