# Adapted from: https://github.com/hkchengrex/XMem/blob/083698bbb4c5ac0ffe1a8923a6c313de46169983/inference/data/test_datasets.py

import json
import os
from os import path

import numpy as np

from .video_reader import VideoReader


class LongTestDataset:
    def __init__(self, data_root, size=-1, longest_size=None):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size
        self.longest_size = longest_size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video,
                              path.join(self.image_dir, video),
                              path.join(self.mask_dir, video),
                              to_save=[
                                  name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                              ],
                              shortest_size=self.size,
                              longest_size=self.longest_size)

    def __len__(self):
        return len(self.vid_list)


class DAVISTestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1, longest_size=None, return_all_gt_masks=False):
        if size != 480:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')
        self.size_dir = path.join(data_root, 'JPEGImages', '480p')
        self.size = size
        self.longest_size = longest_size
        self.return_all_gt_masks = return_all_gt_masks

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video,
                              path.join(self.image_dir, video),
                              path.join(self.mask_dir, video),
                              shortest_size=self.size,
                              longest_size=self.longest_size,
                              size_dir=path.join(self.size_dir, video),
                              use_all_mask=self.return_all_gt_masks)

    def __len__(self):
        return len(self.vid_list)


class YouTubeVOSTestDataset:
    def __init__(self, data_root, split, size=480, longest_size=None):
        self.image_dir = path.join(data_root, 'all_frames', split + '_all_frames', 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.size = size
        self.longest_size = longest_size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video,
                              path.join(self.image_dir, video),
                              path.join(self.mask_dir, video),
                              shortest_size=self.size,
                              longest_size=self.longest_size,
                              to_save=self.req_frame_list[video],
                              use_all_mask=True)

    def __len__(self):
        return len(self.vid_list)


class MOSETestDataset:
    def __init__(self, data_root, split, shortest_size=-1, longest_size=None):
        if split == "val":
            split = "valid"

        self.shortest_size = shortest_size
        self.longest_size = longest_size

        self.image_dir = path.abspath(path.join(data_root, split, 'JPEGImages'))
        self.mask_dir = path.abspath(path.join(data_root, split, 'Annotations'))

        print(f'MOSE-{split}: {self.image_dir}')
        print(f'MOSE-{split}: {self.mask_dir}')
        assert path.exists(self.image_dir)
        assert path.exists(self.mask_dir)

        self.vid_list = sorted(os.listdir(self.image_dir))
        print(f'MOSE-{split}: Found {len(self.vid_list)} videos in {self.image_dir}')

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(
                vid_name=video,
                image_dir=path.join(self.image_dir, video),
                mask_dir=path.join(self.mask_dir, video),
                shortest_size=self.shortest_size,
                longest_size=self.longest_size,
                use_all_mask=True,
            )

    def __len__(self):
        return len(self.vid_list)


class BDD100KTestDataset:
    def __init__(self, data_root, split, shortest_size=-1, longest_size=None):
        self.shortest_size = shortest_size
        self.longest_size = longest_size

        self.image_dir = path.abspath(path.join(data_root, split, 'JPEGImages'))
        self.mask_dir = path.abspath(path.join(data_root, split, 'Annotations'))

        print(f'BDD100K-{split}: {self.image_dir}')
        print(f'BDD100K-{split}: {self.mask_dir}')
        assert path.exists(self.image_dir)
        assert path.exists(self.mask_dir)

        self.vid_list = sorted(os.listdir(self.image_dir))
        print(f'BDD100K-{split}: Found {len(self.vid_list)} videos in {self.image_dir}')

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(
                vid_name=video,
                image_dir=path.join(self.image_dir, video),
                mask_dir=path.join(self.mask_dir, video),
                shortest_size=self.shortest_size,
                longest_size=self.longest_size,
                use_all_mask=True,
                # mask_mode='I;16',
                # mask_dtype=np.int32,
            )

    def __len__(self):
        return len(self.vid_list)
