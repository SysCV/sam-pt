# Prepare Datasets

**Important:** This guide is designed for users operating on a Unix-like system such as Linux or MacOS. 

In this section, we'll walk you through the process of preparing the datasets required to conduct our experiments. For semi-supervised Video Object Segmentation (VOS), the following datasets are needed: [DAVIS 2016](#davis-2016), [DAVIS 2017](#davis-2017), [YouTube-VOS 2018](#youtube-vos-2018), [MOSE 2023](#mose-2023), and [BDD100K](#bdd100k). For Video Instance Segmentation (VIS) experiments, the [UVOv1.0](#uvo-v10) dataset will be utilized.

To begin, create a directory named `data` at the root of your project. This is where all the datasets will be stored:

```bash
mkdir -p data
```

If these datasets are already downloaded and available on your machine, you can create soft links to link the existing data to the structure we expect to find it in. For clarity, we will show you the required directory structures below using the `tree` command.


## Preliminary Checks

Before proceeding to download and process the datasets, it is essential to verify that your machine has sufficient disk space. You can use the `df -h` command in the terminal to check. After unpacking and discarding all compressed archives, the final disk space usage for the datasets we use is approximately:

```txt
 1.9G   data/DAVIS/2016
 7.3G   data/DAVIS/2017
 6.3G   data/YouTube2018
  26G   data/mose
 7.9G   data/bdd100k
 1.4T   data/UVOv1.0
```

Now that the preliminary checks are complete, we can move on to preparing each individual dataset. Please note that you are not required to download all the datasets that we have mentioned. Instead, you can selectively follow the instructions for the datasets you wish to download. Furthermore, we do not necessarily download and prepare all subsets of a dataset, but rather the ones we used in our experiments.

## DAVIS 2016

You can download the DAVIS 2016 dataset from their [official website](https://davischallenge.org/davis2016/code.html). For example, use the following commands to download and extract the dataset:

```bash
mkdir -p data/DAVIS
cd data/DAVIS

wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
unzip DAVIS-data.zip
mv DAVIS 2016 # Rename DAVIS to 2016

cd -
```

The directory structure of the DAVIS 2016 dataset should look like this:

```txt
data/DAVIS/2016
├── Annotations
│   ├── 1080p
│   ├── 480p
│   └── db_info.yml
├── ImageSets
│   ├── 1080p
│   └── 480p
├── JPEGImages
│   ├── 1080p
│   └── 480p
└── README.md
```

## DAVIS 2017

Similar to the DAVIS 2016 dataset, you can download the DAVIS 2017 dataset from their [official website](https://davischallenge.org/davis2017/code.html). The following commands will download, extract, and organize the data:

```bash
mkdir -p data/DAVIS/2017
cd data/DAVIS

# DAVIS 2017 Train and DAVIS 2017 Validation
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip
unzip DAVIS-2017-trainval-480p.zip
unzip -o DAVIS-2017-trainval-Full-Resolution.zip
mv DAVIS 2017/trainval # Move and rename DAVIS to 2017/trainval
rm DAVIS-2017-trainval-480p.zip
rm DAVIS-2017-trainval-Full-Resolution.zip

# DAVIS 2017 Test-dev
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-Full-Resolution.zip
unzip DAVIS-2017-test-dev-480p.zip
unzip -o DAVIS-2017-test-dev-Full-Resolution.zip
mv DAVIS 2017/test-dev # Move and rename DAVIS to 2017/test-dev
rm DAVIS-2017-test-dev-480p.zip
rm DAVIS-2017-test-dev-Full-Resolution.zip

cd -
```

The directory structure of the DAVIS 2017 dataset should look like this:

```txt
data/DAVIS/2017
├── test-dev
│   ├── Annotations
│   │   ├── 480p
│   │   └── Full-Resolution
│   ├── DAVIS
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   │   ├── README.md
│   │   └── SOURCES.md
│   ├── ImageSets
│   │   ├── 2016
│   │   └── 2017
│   ├── JPEGImages
│   │   ├── 480p
│   │   └── Full-Resolution
│   ├── README.md
│   └── SOURCES.md
└── trainval
    ├── Annotations
    │   ├── 480p
    │   └── Full-Resolution
    ├── ImageSets
    │   ├── 2016
    │   └── 2017
    ├── JPEGImages
    │   ├── 480p
    │   └── Full-Resolution
    ├── README.md
    └── SOURCES.md
```

## YouTube-VOS 2018

The YouTube-VOS 2018 dataset is available on their [official website](https://youtube-vos.org/dataset/vos/#data-download). Run the following commands to download and extract the validation subset:

```bash
# Create a directory for the dataset
mkdir -p data/YouTube2018
cd data/YouTube2018

# Install gdown to download files from Google Drive
pip install gdown

# Download and extract the YouTube-VOS 2018 Validation dataset
gdown https://drive.google.com/uc?id=1-QrceIl5sUNTKz7Iq0UsWC6NLZq7girr
gdown https://drive.google.com/uc?id=1yVoHM6zgdcL348cFpolFcEl4IC1gorbV
unzip valid.zip
unzip valid_all_frames.zip

# Move the folder
mkdir all_frames
mv valid_all_frames all_frames/

# Clean up the zip files
rm valid.zip
rm valid_all_frames.zip

# Return to the project root
cd -
```

Your YouTube-VOS 2018 dataset should have the following directory structure:

```txt
data/YouTube2018/
├── valid
│   ├── Annotations
│   ├── JPEGImages
│   └── meta.json
└── all_frames
    └── valid_all_frames
        └── JPEGImages
```

## MOSE 2023

The MOSE 2023 dataset is available on their [official website](https://henghuiding.github.io/MOSE/). The train and valid subsets can be downloaded and prepared using the following commands:

```bash
mkdir data/mose
cd data/mose

gdown 'https://drive.google.com/uc?id=10HYO-CJTaITalhzl_Zbz_Qpesh8F3gZR'  # train.tar.gz
gdown 'https://drive.google.com/uc?id=1yFoacQ0i3J5q6LmnTVVNTTgGocuPB_hR'  # valid.tar.gz

tar xvfz train.tar.gz
tar xvfz valid.tar.gz

rm train.tar.gz
rm valid.tar.gz

cd -
```

After performing these steps, the directory structure for the MOSE 2023 dataset should look as follows:

```txt
data/mose
├── train
│   ├── Annotations
│   └── JPEGImages
└── valid
    ├── Annotations
    └── JPEGImages
```

With MOSE 2023 ready, all VOS datasets are ready for use in the experiments.

## BDD100K

The BDD100K dataset is available on their [official website](https://bdd-data.berkeley.edu/portal.html#download). You need to download `MOTS 2020 Images` and `MOTS 2020 Labels` and organize the downloaded data as follows:

```txt
data/bdd100k
├── images
│   └── seg_track_20
│       ├── test
│       ├── train
│       └── val
├── jsons
└── labels
    └── seg_track_20
        ├── bitmasks
        ├── colormaps
        ├── polygons
        └── rles
```

Then, to convert the MOTS data to a semi-supervised VOS dataset format, you can use our conversion script `scripts.bdd100k_from_instance_seg_to_vos_annotations` as follows:

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

This then gives the following directory structure:

```txt
data/bdd100k/vos
└── val
    ├── Annotations
    │   ├── b1c66a42-6f7d68ca
    │   ├── b1c66a42-6f7d68ca-chunk2
    │   ├── b1c81faa-3df17267
    │   ├── b1c81faa-c80764c5
    │   └── ...
    └── JPEGImages
        ├── b1c66a42-6f7d68ca
        ├── b1c66a42-6f7d68ca-chunk2 -> b1c66a42-6f7d68ca
        ├── b1c81faa-3df17267
        ├── b1c81faa-c80764c5
        └── ...
```

---
## UVO v1.0

The UVO v1.0 dataset is available on their [official website](https://sites.google.com/view/unidentified-video-object/dataset). Follow the steps below to download and prepare the UVO v1.0 dataset. Note that I have zipped and reuploaded the UVOv1 folder for a more convenient download from Google Drive as it is otherwise hard to download a folder with a lot of files from Google Drive using gdown. I have downloaded and zipped the folder on April 8, 2023.

```bash
cd data

pip install gdown
gdown --no-check-certificate https://drive.google.com/uc?id=1AGu4BL-i_vDCMNtwsoSuo5wIyDVd5dRf
unzip  UVOv1.0.zip
rm UVOv1.0.zip

# Download the preprocessed videos
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1fOhEdHqrp_6D_tBsrR9hazDLYV2Sw1XC
unzip UVO_Videos/uvo_videos_dense.zip
unzip UVO_Videos/uvo_videos_sparse.zip
mv uvo_videos_dense/ UVOv1.0/
mv uvo_videos_sparse/ UVOv1.0/
rm -rf UVO_Videos/
rm -rf __MACOSX/
```

After running these commands, your UVO v1.0 directory should look like this:

```bash
tree -L 1 UVOv1.0
# UVOv1.0
# ├── EvaluationAPI
# ├── ExampleSubmission
# ├── FrameSet
# ├── README.md
# ├── VideoDenseSet
# ├── VideoSparseSet
# ├── YT-IDs
# ├── download_kinetics.py
# ├── preprocess_kinetics.py
# ├── uvo_videos_dense
# ├── uvo_videos_sparse
# └── video2frames.py

du -sch UVOv1.0/*
# 6.3M    UVOv1.0/EvaluationAPI
# 683M    UVOv1.0/ExampleSubmission
# 176M    UVOv1.0/FrameSet
# 8.0K    UVOv1.0/README.md
# 736M    UVOv1.0/VideoDenseSet
# 3.5G    UVOv1.0/VideoSparseSet
# 248K    UVOv1.0/YT-IDs
# 4.0K    UVOv1.0/download_kinetics.py
# 8.0K    UVOv1.0/preprocess_kinetics.py
# 1.3G    UVOv1.0/uvo_videos_dense
# 13G     UVOv1.0/uvo_videos_sparse
# 4.0K    UVOv1.0/video2frames.py
# 20G     total
```

I recommend using the [`scripts/uvo_video2frames.py`](../scripts/uvo_video2frames.py) script instead of the packaged `UVOv1.0/video2frames.py` script to split the preprocessed videos into frames. To do so, run the following commands to split the videos into frames, assuming that the root of SAM-PT is at `..` (if not, you can specify the full path to the script, e.g. like `/path/to/sam-pt/root/scripts/uvo_video2frames.py`):

```bash
python ../scripts/uvo_video2frames.py --video_dir UVOv1.0/uvo_videos_dense --frames_dir UVOv1.0/uvo_videos_dense_frames
python ../scripts/uvo_video2frames.py --video_dir UVOv1.0/uvo_videos_sparse --frames_dir UVOv1.0/uvo_videos_sparse_frames
```

Your updated UVO v1.0 directory should look like this:

```txt
UVOv1.0
├── EvaluationAPI
├── ExampleSubmission
├── FrameSet
├── README.md
├── VideoDenseSet
├── VideoSparseSet
├── YT-IDs
├── download_kinetics.py
├── preprocess_kinetics.py
├── uvo_videos_dense
├── uvo_videos_dense_frames
├── uvo_videos_sparse
├── uvo_videos_sparse_frames
└── video2frames.py
```

Finally, navigate back to your root directory:

```bash
cd -
```

Your UVO v1.0 dataset is now ready for use in your experiments.

## What's Next?

Now that you have prepared the datasets, you can [prepare the checkpoints](03-prepare-checkpoints.md) that are necessary for [running our VOS and VIS experiments](04-running-experiments.md).
