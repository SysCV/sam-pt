"""
A utility script to split UVO videos into frames.

The script takes two command-line arguments:
1. --video_dir: The directory containing the videos you wish to split into frames.
2. --frames_dir: The directory where the frames will be saved.

Each video in the input directory will be split into frames, and these frames will be stored in a subdirectory of --frames_dir named after the video.

Usage:

```bash
python ../scripts/uvo_video2frames.py --video_dir UVOv1.0/uvo_videos_dense --frames_dir UVOv1.0/uvo_videos_dense_frames
python ../scripts/uvo_video2frames.py --video_dir UVOv1.0/uvo_videos_sparse --frames_dir UVOv1.0/uvo_videos_sparse_frames
```
"""
import argparse
import cv2
import os
import pathlib
from tqdm import tqdm


def split_single_video(video_path, frames_dir=""):
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            success, buffer = cv2.imencode(".png", frame)
            if success:
                with open(f"{frames_dir}{cnt}.png", "wb") as f:
                    f.write(buffer.tobytes())
                    f.flush()
                cnt += 1
        else:
            break
    return cnt


def get_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--video_dir", type=str, default="NonPublic/uvo_videos_dense/")
    arg_parser.add_argument("--frames_dir", type=str, default="NonPublic/uvo_videos_dense_frames/")
    return arg_parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    video_paths = os.listdir(args.video_dir)
    print(f"Splitting videos in {args.video_dir} to frames in {args.frames_dir}...")
    print(f"Total number of videos: {len(video_paths)}")
    for video_path in tqdm(video_paths):
        print(f"Splitting {video_path}...")
        v_frame_dir = pathlib.Path(os.path.join(args.frames_dir, video_path[:-4]))
        if not v_frame_dir.is_dir():
            v_frame_dir.mkdir(parents=True, exist_ok=False)
        n_frames = split_single_video(os.path.join(args.video_dir, video_path), frames_dir=v_frame_dir)
        print(f"Total number of frames extracted from {video_path}: {n_frames}")
    print(f"Done.")
