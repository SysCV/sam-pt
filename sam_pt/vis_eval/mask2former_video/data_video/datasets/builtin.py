# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os

from .uvo import (
    _get_uvo_v1_instances_meta,
)
from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}

# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train/instances.json"),
    "ytvis_2021_train_mini": ("ytvis_2021/train/JPEGImages",
                              "ytvis_2021/train/instances.mini.27.json"),
    "ytvis_2021_train_tiny": (
        # cat data/ytvis_2021/train/instances.mini.27.json | jq '.videos |= [.[0]] | .annotations |= [.[0,1]]' > data/ytvis_2021/train/instances.tiny.1.json
        "ytvis_2021/train/JPEGImages",
        "ytvis_2021/train/instances.tiny.1.json",
    ),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid/instances.json"),
    "ytvis_2021_val_mini": ("ytvis_2021/valid/JPEGImages",
                            "ytvis_2021/valid/instances.mini.27.json"),
    "ytvis_2021_val_tiny": ("ytvis_2021/valid/JPEGImages",
                            "ytvis_2021/valid/instances.mini.1.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test/instances.json"),
}

_PREDEFINED_SPLITS_UVO_V1 = {
    "uvo_v1_train": ("UVOv1.0/uvo_videos_dense_frames/",
                     "UVOv1.0/VideoDenseSet/UVO_video_train_dense.json"),
    "uvo_v1_val": ("UVOv1.0/uvo_videos_dense_frames/",
                   "UVOv1.0/VideoDenseSet/UVO_video_val_dense.json"),
    "uvo_v1_val_tiny": (
        # Contains only 1 video
        # Split created using jq: `cat data/UVOv1.0/VideoDenseSet/UVO_video_val_dense.json | jq '.videos |= [.[0]] | .annotations |= [.[0,1,2,3]]' > data/UVOv1.0/VideoDenseSet/UVO_video_val_dense.tiny.1.json`
        "UVOv1.0/uvo_videos_dense_frames/",
        "UVOv1.0/VideoDenseSet/UVO_video_val_dense.tiny.1.json",
    ),
    "uvo_v1_test": ("UVOv1.0/uvo_videos_dense_frames/",
                    "UVOv1.0/VideoDenseSet/UVO_video_test_dense.json"),
}

_PREDEFINED_SPLITS_UVO_V05 = {
    "uvo_v05_train": ("UVOv1.0/uvo_videos_dense_frames/",
                      "UVOv0.5/VideoDenseSet/UVO_video_train_dense.json"),
    "uvo_v05_val": ("UVOv1.0/uvo_videos_dense_frames/",
                    "UVOv0.5/VideoDenseSet/UVO_video_val_dense.json"),
    "uvo_v05_val_tiny": (
        # Contains only 1 video
        # Split created using jq: `cat data/UVOv0.5/VideoDenseSet/UVO_video_val_dense.json | jq '.videos |= [.[0]] | .annotations |= [.[0,1,2,3]]' > data/UVOv0.5/VideoDenseSet/UVO_video_val_dense.tiny.1.json`
        "UVOv1.0/uvo_videos_dense_frames/",
        "UVOv0.5/VideoDenseSet/UVO_video_val_dense.tiny.1.json",
    ),
    "uvo_v05_test": ("UVOv1.0/uvo_videos_dense_frames/",
                     "UVOv0.5/VideoDenseSet/UVO_video_test_dense.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_uvo_v1(_root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO_V1.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_uvo_v1_instances_meta(),
            os.path.join(_root, json_file) if "://" not in json_file else json_file,
            os.path.join(_root, image_root),
        )


def register_all_uvo_v05(_root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO_V05.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_uvo_v1_instances_meta(),
            os.path.join(_root, json_file) if "://" not in json_file else json_file,
            os.path.join(_root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_uvo_v1(_root)
    register_all_uvo_v05(_root)
