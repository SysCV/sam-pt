# Preparing Checkpoints

This guide will walk you through the process of preparing the model checkpoints needed for our experiments.

Start by creating a `models` directory at the root of your project to store all the model checkpoints:

```bash
mkdir models
```

## Minimal Checkpoints

If you're running our default variant, SAM-PT or SAM-PT-reinit, you'll only need the following minimal set of checkpoints:

```py
# tree --du -h models
[2.5G]  models
├── [ 92M]  cotracker_ckpts
│   └── [ 92M]  reference_model
│       └── [ 92M]  cotracker_stride_4_wind_8.pth
└── [2.4G]  samhq_ckpts
    └── [2.4G]  sam_hq_vit_h.pth
```

These can be fetched by following the instructions for [HQ-SAM](#hq-sam-and-light-hq-sam) and [CoTracker](#cotracker) provided below.

## Complete Checkpoints

For replicating all experiments in the paper, you will need additional checkpoints. These require 9.1GB of disk space in total. Here is the complete structure of the required checkpoints:

```py
# tree --du -h models
[9.3G]  models
├── [277M]  cotracker_ckpts
│   ├── [ 92M]  cotracker_stride_4_wind_12.pth
│   ├── [ 92M]  cotracker_stride_4_wind_8.pth
│   └── [ 92M]  cotracker_stride_8_wind_16.pth
├── [328M]  pips_ckpts
│   └── [328M]  reference_model
│       └── [328M]  model-000200000.pth
├── [402M]  pips_plus_plus_ckpts
│   └── [402M]  reference_model
│       └── [402M]  model-000200000.pth
├── [ 84M]  raft_ckpts
│   ├── [ 20M]  raft-chairs.pth
│   ├── [ 20M]  raft-kitti.pth
│   ├── [ 20M]  raft-sintel.pth
│   ├── [3.8M]  raft-small.pth
│   └── [ 20M]  raft-things.pth
├── [3.9G]  sam_ckpts
│   ├── [358M]  sam_vit_b_01ec64.pth
│   ├── [2.4G]  sam_vit_h_4b8939.pth
│   └── [1.2G]  sam_vit_l_0b3195.pth
├── [ 39M]  sam_mobile_ckpts
│   └── [ 39M]  sam_mobile_vit_t.pth
├── [4.0G]  samhq_ckpts
│   ├── [362M]  sam_hq_vit_b.pth
│   ├── [2.4G]  sam_hq_vit_h.pth
│   ├── [1.2G]  sam_hq_vit_l.pth
│   └── [ 41M]  sam_hq_vit_t.pth
├── [ 97M]  superglue_ckpts
│   ├── [ 46M]  superglue_indoor.pth
│   ├── [ 46M]  superglue_outdoor.pth
│   └── [5.0M]  superpoint_v1.pth
├── [240M]  tapir_ckpts
│   └── [240M]  open_source_ckpt
│       ├── [121M]  causal_tapir_checkpoint.npy
│       └── [119M]  tapir_checkpoint_panning.npy
└── [ 43M]  tapnet_ckpts
    └── [ 43M]  open_source_ckpt
        ├── [ 32M]  checkpoint.npy
        └── [ 11M]  checkpoint_wo_optstate.npy
```

Additionally, these are the md5 sums of the checkpoints we have used:

```bash
# find models -type f -print0 | sort -z | xargs -r0 md5sum
f13ab80f04b2cb58945e2dffb5a3a44c  models/cotracker_ckpts/cotracker_stride_4_wind_12.pth
82c458ad5de9bf98bc337c34ccbc436a  models/cotracker_ckpts/cotracker_stride_4_wind_8.pth
d0d25fe323b20d11c447aaa05a923650  models/cotracker_ckpts/cotracker_stride_8_wind_16.pth
9f34c4cd5d6f54cb11e1911841ac702c  models/pips_ckpts/reference_model/model-000200000.pth
e599fd3ba978d67c4cf35f225be3c2af  models/pips_plus_plus_ckpts/reference_model/model-000200000.pth
37d7c11dccc199c915580562651d85dd  models/raft_ckpts/raft-chairs.pth
e5882fe9b35e1a7cb80537f6f859179f  models/raft_ckpts/raft-kitti.pth
cc69e5da1f38673ab10d1849859ebe91  models/raft_ckpts/raft-sintel.pth
925642262acc623b3996690eac7d14c9  models/raft_ckpts/raft-small.pth
55b58de5d9022eb37893916d246e14a3  models/raft_ckpts/raft-things.pth
01ec64d29a2fca3f0661936605ae66f8  models/sam_ckpts/sam_vit_b_01ec64.pth
4b8939a88964f0f4ff5f5b2642c598a6  models/sam_ckpts/sam_vit_h_4b8939.pth
0b3195507c641ddb6910d2bb5adee89c  models/sam_ckpts/sam_vit_l_0b3195.pth
f3c0d8cda613564d499310dab6c812cd  models/sam_mobile_ckpts/sam_mobile_vit_t.pth
c6b8953247bcfdc8bb8ef91e36a6cacc  models/samhq_ckpts/sam_hq_vit_b.pth
3560f6b6a5a6edacd814a1325c39640a  models/samhq_ckpts/sam_hq_vit_h.pth
08947267966e4264fb39523eccc33f86  models/samhq_ckpts/sam_hq_vit_l.pth
3a661ab92d4088ccd6232fa542998a65  models/samhq_ckpts/sam_hq_vit_t.pth
48053342712ef9a8e4663490b812ad50  models/superglue_ckpts/superglue_indoor.pth
01191e832e901537324543963bea09a4  models/superglue_ckpts/superglue_outdoor.pth
938af9f432d327751dcbc0d6c7a0448b  models/superglue_ckpts/superpoint_v1.pth
763a9dbdf9e077395217aaddc6c6f048  models/tapir_ckpts/open_source_ckpt/causal_tapir_checkpoint.npy
73e86cdcfef0e8afea6a060f44be5fc7  models/tapir_ckpts/open_source_ckpt/tapir_checkpoint_panning.npy
b8a1ad6eab94ce53be3ce870ed829552  models/tapnet_ckpts/open_source_ckpt/checkpoint.npy
3e915fcb27a6fe39c46e1d5ed443d11a  models/tapnet_ckpts/open_source_ckpt/checkpoint_wo_optstate.npy
```

The following sections provide the sources and instructions for downloading these checkpoints. Please refer to the original sources for more detailed instructions or if the links below become inactive. After downloading the checkpoints, they should be placed in the corresponding directories as shown above.

## Downloading Checkpoints

### SAM

Source: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

```bash
mkdir models/sam_ckpts
wget --output-document models/sam_ckpts/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget --output-document models/sam_ckpts/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget --output-document models/sam_ckpts/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### PIPS

Source: [aharley/pips](https://github.com/aharley/pips)

```bash
mkdir models/pips_ckpts
wget --output-document models/pips_ckpts/reference_model.tar.gz https://www.dropbox.com/s/glk6jmoa9yeervl/reference_model.tar.gz
tar xvf models/pips_ckpts/reference_model.tar.gz --directory models/pips_ckpts
rm models/pips_ckpts/reference_model.tar.gz
```

### PIPS++

Source: [aharley/pips2](https://github.com/aharley/pips2)

```bash
mkdir models/pips_plus_plus_ckpts
wget --output-document models/pips_plus_plus_ckpts/reference_model.tar.gz https://www.dropbox.com/scl/fi/czdlt2zc2ji2b7zd0pvoe/reference_model.tar.gz?rlkey=56ebq4g5dk01kyq8kuismev14
tar xvf models/pips_plus_plus_ckpts/reference_model.tar.gz --directory models/pips_plus_plus_ckpts
rm models/pips_plus_plus_ckpts/reference_model.tar.gz
```

### HQ-SAM and Light HQ-SAM

Source: [SysCV/sam-hq](https://github.com/SysCV/sam-hq)

```bash
mkdir models/samhq_ckpts
wget --output-document models/samhq_ckpts/sam_hq_vit_b.pth https://huggingface.co/lkeab/hq-sam/resolve/67ab82412bc794d5ce2e9799b8b6a3c0a8cfe1d2/sam_hq_vit_b.pth
wget --output-document models/samhq_ckpts/sam_hq_vit_l.pth https://huggingface.co/lkeab/hq-sam/resolve/67ab82412bc794d5ce2e9799b8b6a3c0a8cfe1d2/sam_hq_vit_l.pth
wget --output-document models/samhq_ckpts/sam_hq_vit_h.pth https://huggingface.co/lkeab/hq-sam/resolve/67ab82412bc794d5ce2e9799b8b6a3c0a8cfe1d2/sam_hq_vit_h.pth
wget --output-document models/samhq_ckpts/sam_hq_vit_t.pth https://huggingface.co/lkeab/hq-sam/resolve/a3a77cd0a2e5e50eaa76faccf61b964732d9b35f/sam_hq_vit_tiny.pth
```

### MobileSAM

Source: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

```bash
mkdir models/sam_mobile_ckpts
wget --output-document models/sam_mobile_ckpts/sam_mobile_vit_t.pth https://github.com/ChaoningZhang/MobileSAM/raw/01ea8d0f5590082f0c1ceb0a3e2272593f20154b/weights/mobile_sam.pt
```

### RAFT

Source: [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)

```bash
mkdir models/raft_ckpts
wget --output-document models/raft_ckpts/models.zip https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip -j models/raft_ckpts/models.zip models/raft-chairs.pth  models/raft-kitti.pth models/raft-sintel.pth models/raft-small.pth models/raft-things.pth -d models/raft_ckpts
rm models/raft_ckpts/models.zip
```

### TAPIR

Source: [deepmind/tapnet](https://github.com/deepmind/tapnet)

```bash
mkdir models/tapir_ckpts
mkdir models/tapir_ckpts/open_source_ckpt
wget --output-document models/tapir_ckpts/open_source_ckpt/causal_tapir_checkpoint.npy https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
wget --output-document models/tapir_ckpts/open_source_ckpt/tapir_checkpoint_panning.npy https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
```

### TapNet

Source: [deepmind/tapnet](https://github.com/deepmind/tapnet)

```bash
mkdir models/tapnet_ckpts
mkdir models/tapnet_ckpts/open_source_ckpt
wget --output-document models/tapnet_ckpts/open_source_ckpt/checkpoint.npy https://storage.googleapis.com/dm-tapnet/checkpoint.npy
pip install gdown
gdown --output models/tapnet_ckpts/open_source_ckpt/checkpoint_wo_optstate.npy --fuzzy https://drive.google.com/file/d/1fGhoW33k87OQQHUTmFobMUjEnRixhuqg/view?usp=sharing
```

**Note:** We convert the original `checkpoint.npy` TapNet checkpoint to `checkpoint_wo_optstate.npy` as to remove the pickled objects that require tapnet to be installed, and put the checkpoint onto google drive for simplicity of retrieval. The script used to process the original checkpoint is provided under [`tools/clean_tapnet_checkpoint.py`](../scripts/clean_tapnet_checkpoint.py), for your reference.

### SuperGlue

Source: [magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)

```bash
mkdir models/superglue_ckpts
wget --output-document models/superglue_ckpts/superglue_indoor.pth https://github.com/magicleap/SuperGluePretrainedNetwork/raw/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/weights/superglue_indoor.pth
wget --output-document models/superglue_ckpts/superglue_outdoor.pth https://github.com/magicleap/SuperGluePretrainedNetwork/raw/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/weights/superglue_outdoor.pth
wget --output-document models/superglue_ckpts/superpoint_v1.pth https://github.com/magicleap/SuperGluePretrainedNetwork/raw/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/weights/superpoint_v1.pth 
```

### CoTracker

Source: [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)

```bash
mkdir models/cotracker_ckpts
wget --output-document models/cotracker_ckpts/cotracker_stride_4_wind_8.pth https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
wget --output-document models/cotracker_ckpts/cotracker_stride_4_wind_12.pth https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_12.pth
wget --output-document models/cotracker_ckpts/cotracker_stride_8_wind_16.pth https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_8_wind_16.pth
```

## What's Next?

With the necessary checkpoints prepared, you can head on to [running our VOS and VIS experiments](04-running-experiments.md).
