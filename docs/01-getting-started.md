# Getting Started

## Setting Up the Environment

This codebase has been tested and confirmed to be compatible with the package versions listed in [`requirements.txt`](../requirements.txt), along with PyTorch 1.12.0, and Python 3.8.16. These versions were tested on Manjaro Linux and Debian GNU/Linux 10 (buster) systems.

Start by cloning the repository:

```bash
git clone https://github.com/SysCV/sam-pt.git
cd sam-pt
```

With the repository now cloned, we recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment:

```bash
conda create --name sam-pt python=3.8.16 -y
conda activate sam-pt
```

Next, install [PyTorch](https://pytorch.org/) 1.12.0 and [torchvision](https://pytorch.org/vision/stable/index.html) 0.13.0, for example with CUDA 11 support:

```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Finally, install the required packages:

```bash
pip install -r requirements.txt
```

If you wish to use TapNet (or TAPIR) as a point tracker, it's necessary to configure JAX on your system. The required packages, including JAX library version [0.4.11](https://github.com/google/jax/tree/jax-v0.4.11) and others needed by TapNet, can be found in the [`requirements-jax.txt`](../requirements-jax.txt) file. To install JAX, we recommend following the [official installation instructions](https://github.com/google/jax#installation). In some environments, like ours, it may be necessary to build PyTorch and JAX from source.

## Running the Demo

To run the demo, start by preparing your demo data. This can either be one of the clips provided in `data/demo_data`, or a clip of your own. You can also use the horse jumping video `data/DAVIS/2017/trainval/JPEGImages/Full-Resolution/horsejump-high` from [DAVIS 2017](02-prepare-datasets.md#davis-2017).

The demo expects a sequence of images as input. If your data is a video clip, convert it to images ensuring their filenames are lexicographically ordered (e.g., `frame-000.png`, `frame-001.png`, etc.). For example, the `ffmpeg` command can be used to convert the provided demo clips as follows:

```bash
# List the content of the demo_data directory
ls data/demo_data
# bees.mp4  street.mp4 ...

# Convert bees.mp4 to png frames
mkdir data/demo_data/bees
ffmpeg -i data/demo_data/bees.mp4 -vf fps=5 data/demo_data/bees/frame-%05d.png

# Convert street.mp4 to png frames
mkdir data/demo_data/street
ffmpeg -i data/demo_data/street.mp4 -vf fps=10 data/demo_data/street/frame-%05d.png
```

Before running the demo, you additionally have to make sure to have the SAM and PIPS checkpoints downloaded, as described under [minimal checkpoints](03-prepare-checkpoints.md#minimal-checkpoints).

### Running the Interactive Demo

The interactive demo allows you to specify query points using mouse clicks on a pop-up window. This requires a GUI environment, which is typically available on personal computers. If you're using remote GPUs, you may need to set up X forwarding.


Note that the [`${hydra:runtime.cwd}`](https://hydra.cc/docs/1.3/configure_hydra/intro/#hydraruntime) prefix in the commands below needs to be used to prefix relative paths. This is because we launch demos within a [working directory created by Hydra](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/working_directory/). Follow the instructions displayed in your terminal after launching the interactive demo.


```bash
# Run demo on bees.mp4
export HYDRA_FULL_ERROR=1
python -m demo.demo \
  frames_path='${hydra:runtime.cwd}/data/demo_data/bees/' \
  query_points_path=null \
  longest_side_length=1024 frame_stride=1 max_frames=-1

# Run demo on street.mp4
export HYDRA_FULL_ERROR=1
python -m demo.demo \
  frames_path='${hydra:runtime.cwd}/data/demo_data/street/' \
  query_points_path=null \
  longest_side_length=1024 frame_stride=1 max_frames=-1
```

### Running the Non-interactive Demo

You also have the option to run the demo in a non-interactive mode where query points are predefined in a file. You can create the content of a query points file using the interactive demo, which will print a string of the query points. This string can be saved and used for running the non-interactive demo. More details about the format of the query points file can be found in [`data/demo_data/README.md`](../data/demo_data/README.md). Examples of query point files for the [bees](../data/demo_data/query_points__bees.txt) and [street](../data/demo_data/query_points__street.txt) clips are also provided and can be used as in the following commands:

```bash
# Run non-interactive demo on bees.mp4
export HYDRA_FULL_ERROR=1
python -m demo.demo \
  frames_path='${hydra:runtime.cwd}/data/demo_data/bees/' \
  query_points_path='${hydra:runtime.cwd}/data/demo_data/query_points__bees.txt' \
  longest_side_length=1024 frame_stride=1 max_frames=-1

# Run non-interactive demo on street.mp4
export HYDRA_FULL_ERROR=1
python -m demo.demo \
  frames_path='${hydra:runtime.cwd}/data/demo_data/street/' \
  query_points_path='${hydra:runtime.cwd}/data/demo_data/query_points__street.txt' \
  longest_side_length=1024 frame_stride=1 max_frames=-1
```

## Codebase Overview

Here's a quick overview of our project's codebase and its structure:

- [`assets`](../assets): Assets related to the GitHub repository
- [`configs`](../configs): YAML configuration files used with Hydra
- [`data`](../data): Directory to store data
  - [`demo_data`](../data/demo_data): Demo data with README for data sources and query points file format
- [`demo`](../demo): Code for running the demo
- [`docs`](../docs): Documentation on how to use the codebase
- [`sam_pt`](../sam_pt): Source for SAM-PT
  - [`modeling`](../sam_pt/modeling): Main code for SAM-PT
  - [`point_tracker`](../sam_pt/point_tracker): Code for different point trackers
  - [`utils`](../sam_pt/utils): Utilities used within the SAM-PT module
  - [`vis_eval`](../sam_pt/vis_eval): Code for evaluating on Video Instance Segmentation (VIS)
  - [`vos_eval`](../sam_pt/vos_eval): Code for evaluating on Video Object Segmentation (VOS)
- [`scripts`](../scripts): Scripts used for small tasks
- [`README.md`](../README.md): Main README file
- [`requirements.txt`](../requirements.txt): General project requirements
- [`requirements-jax.txt`](../requirements-jax.txt): Requirements for using the JAX-based TapNet and TAPIR point trackers


## What's Next?

Once you are comfortable with running the demo, you might want to explore [how to prepare the data](02-prepare-datasets.md) and [how to prepare the checkpoints](03-prepare-checkpoints.md) that are necessary for [running our VOS and VIS experiments](04-running-experiments.md).
