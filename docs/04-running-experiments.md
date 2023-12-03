# Running Experiments

**Note:** Before running the experiments, ensure you've correctly prepared your datasets and checkpoints. If you haven't done so, please refer back to the [Prepare Datasets](./02-prepare-datasets.md) and [Prepare Checkpoints](./03-prepare-checkpoints.md) tutorials.

## Overview

In this tutorial, we'll guide you through the process of running experiments using the SAM-PT model. We utilize the [Hydra](https://hydra.cc/) framework to manage configurations for our experiments, simplifying the customization of various model components, such as the point tracker, dataset, and model versions.

## Running SAM-PT

To run SAM-PT with the default VOS evaluation configuration [`vos_eval_root.yaml`](../configs/vos_eval_root.yaml), use the following command:

```bash
python -m sam_pt.vos_eval.eval model=sam_pt
```

### Using Different Point Trackers

The flexibility of SAM-PT allows for the usage of different point trackers. To specify a different point tracker, append it to the command. For instance, to use RAFT, TapNet, PIPS, etc., as the point tracker, execute the following:

```bash
python -m sam_pt.vos_eval.eval model=sam_pt model/point_tracker=raft

python -m sam_pt.vos_eval.eval model=sam_pt model/point_tracker=tapnet

python -m sam_pt.vos_eval.eval model=sam_pt \
  model/point_tracker=pips \
  model.positive_points_per_mask=8

python -m sam_pt.vos_eval.eval model=sam_pt \
  model/point_tracker=pips_plus_plus \
  model.positive_points_per_mask=8

python -m sam_pt.vos_eval.eval model=sam_pt \
  model/point_tracker=tapir \
  model.positive_points_per_mask=32 \
  model.negative_points_per_mask=72 \
  model.iterative_refinement_iterations=24

python -m sam_pt.vos_eval.eval model=sam_pt \
  model/point_tracker=superglue \
  model.negative_points_per_mask=0 \
  model.positive_point_selection_method=random \
  model.negative_point_selection_method=random \
  model.iterative_refinement_iterations=3
```

For more point trackers to chose from, see the available ones in [`configs/model/point_tracker`](../configs/model/point_tracker), or implement a new one by implementing the interface defined by [`sam_pt.point_tracker.tracker.PointTracker`](../sam_pt/point_tracker/tracker.py).

### Using Different Variants of SAM

Use SAM with different ViT backbones with the following commands:

```bash
# SAM's ViT-Huge (used throughout the ablation study)
python -m sam_pt.vos_eval.eval model=sam_pt \
  model/sam@model.sam_predictor.sam_model=sam_vit_huge \
  model.sam_predictor._target_=segment_anything.predictor.SamPredictor

# ViT-Large
python -m sam_pt.vos_eval.eval model=sam_pt \
  model/sam@model.sam_predictor.sam_model=sam_vit_large \
  model.sam_predictor._target_=segment_anything.predictor.SamPredictor

# ViT-Base
python -m sam_pt.vos_eval.eval model=sam_pt \
  model/sam@model.sam_predictor.sam_model=sam_vit_base \
  model.sam_predictor._target_=segment_anything.predictor.SamPredictor
```

Using smaller backbones results in slight inference speed gains, with a slight performance deterioration as measured on the validation subset of DAVIS 2017 (note that the numbers are reported for PIPS as the point tracker):

| SAM Variant  	| Backbone  	| J&F   	    | FPS 	    |
|--------------	|-----------	|-------	    |-----	    |
| SAM          	| ViT-Huge  	| **76.65** 	| 1.4 	    |
| SAM          	| ViT-Large 	| 76.43 	    | 1.8 	    |
| SAM          	| ViT-Base  	| 72.18 	    | **2.6** 	|

Replace the SAM model with HQ-SAM or MobileSAM using the following commands:

For HQ-SAM (used by default in [`configs/model/sam_pt.yaml`](../configs/model/sam_pt.yaml)):

```bash
python -m sam_pt.vos_eval.eval model=sam_pt \
  model/sam@model.sam_predictor.sam_model=samhq_vit_huge \
  model.sam_predictor._target_=segment_anything_hq.predictor.SamPredictor
```

For Light HQ-SAM:

```bash
python -m sam_pt.vos_eval.eval model=sam_pt \
  model/sam@model.sam_predictor.sam_model=samhq_light_vit_tiny \
  model.sam_predictor._target_=segment_anything_hq.predictor.SamPredictor \
  model.iterative_refinement_iterations=3
```

For MobileSAM:

```bash
python -m sam_pt.vos_eval.eval model=sam_pt \
  model/sam@model.sam_predictor.sam_model=sam_mobile_vit_tiny \
  model.iterative_refinement_iterations=3
```

Refer to the comparison table below to understand the differences between each variant with regard to the average $\mathcal{J\\&F}$ score on DAVIS 2017 and the inference speed. Note that the comparison is reported with PIPS as the point tracker.

| SAM Variant  	| Backbone  	| J&F   	    | FPS 	    |
|--------------	|-----------	|-------	    |-----	    |
| SAM          	| ViT-Huge  	| 76.65 	    | 1.4 	    |
| HQ-SAM       	| ViT-Huge  	| **77.64** 	| 1.3 	    |
| Light HQ-SAM 	| ViT-Tiny  	| 71.30 	    | 4.8 	    |
| MobileSAM    	| ViT-Tiny  	| 71.07 	    | **5.5** 	|

## Running VOS Experiments

Evaluate SAM-PT on various VOS datasets like DAVIS 2016, DAVIS 2017, YouTube-VOS 2018, MOSE 2023, and BDD100K using the following commands:

```bash
python -m sam_pt.vos_eval.eval model=sam_pt dataset=D16     split=val
python -m sam_pt.vos_eval.eval model=sam_pt dataset=D17     split=val
python -m sam_pt.vos_eval.eval model=sam_pt dataset=D17     split=test
python -m sam_pt.vos_eval.eval model=sam_pt dataset=Y18     split=val
python -m sam_pt.vos_eval.eval model=sam_pt dataset=MOSE    split=val  model.iterative_refinement_iterations=0
python -m sam_pt.vos_eval.eval model=sam_pt dataset=BDD100K split=val  model.iterative_refinement_iterations=0 model.point_tracker_mask_batch_size=100 visualize_results=false
```

Note that the validation subset of DAVIS 2017 is used for ablation study experiments.

Similarly, you can evaluate SAM-PT-reinit:

```bash
python -m sam_pt.vos_eval.eval model=sam_pt_reinit dataset=D17 split=val model.use_point_reinit=true
```

## Running VIS Experiments

To run VIS experiments with SAM-PT, use the following command:

```bash
python -m sam_pt.vis_eval.eval model@model.model=sam_pt \
  model.model.iterative_refinement_iterations=0 \
  model.model.add_other_objects_positive_points_as_negative_points=false \
  num_gpus_per_machine=8 num_machines=1 machine_rank=0 dist_url=tcp://localhost:27036 \
  DETECTRON2_CONFIG.SEED=36
```

Please adjust the number of GPUs and machines according to your available hardware setup. If you're experiencing out-of-memory issues, you can reduce the mask batch size using the `model.masks_batch_size=10` argument (default value is 100, as shown in [`vis_eval_root.yaml`](../configs/vis_eval_root.yaml)).

### Debugging with a Tiny Dataset

For debugging purposes, you may wish to use a subset of the UVO dataset. A tiny UVO subset can be created and used following the instructions below. We have already registered the [`uvo_v1_val_tiny`](../sam_pt/vis_eval/mask2former_video/data_video/datasets/builtin.py) split with Detectron2. Should you require the [`jq`](https://github.com/jqlang/jq) command utility, it can be installed without sudo using `conda install -c conda-forge jq`.

```bash
cat data/UVOv1.0/VideoDenseSet/UVO_video_val_dense.json \
  | jq '.videos |= [.[0]] | .annotations |= [.[0,1,2,3]]' \
  > data/UVOv1.0/VideoDenseSet/UVO_video_val_dense.tiny.json
```

When running the VIS evaluation, include the flag `DETECTRON2_CONFIG.DATASETS.TEST=[uvo_v1_val_tiny]` to use the tiny dataset.

## Running Interactive Point-Based Video Segmentation

To run our simulation of interactive point-based video segmentation with online HQ-SAM-PT on videos 4 and 12 (zero-indexed, correspond to `camel` and `drift-straight` videos, respectively) of the DAVIS 2017 validation subset, use the following command:

```bash
python -m sam_pt.vos_eval.eval model=sam_pt \
  simulate_interactive_point_correction=true \
  masks_batch_size=1 \
  visualize_results=false \
  vid_ids=\[4,12\] \
  model._target_=sam_pt.modeling.sam_pt_interactive.SamPtInteractive \
  model.iterative_refinement_iterations=0 \
  model.positive_points_per_mask=1 \
  model.negative_points_per_mask=0 \
  +model.visualize_all_interactions_separately=true \
  +model.visualize_all_interactions_as_mp4=true \
  +model.interactions_max=300 \
  +model.interactions_max_per_frame=4 \
  +model.online=true \
  +model.online_interactive_iou_threshold=0.95
```

Optionally, you can add `+model.font_path=/path/to/helvetica.ttf` to the command to use a custom font for the text overlay. For videos with a lot of interaction, the visualizations are very time consuming to create and can be disabled by setting the visualization flags to `false`.

To run the same experiment with point tracking disabled, i.e., to run the baseline that interacts with SAM separately for each frame, use the following command:

```bash
python -m sam_pt.vos_eval.eval model=sam_pt \
  simulate_interactive_point_correction=true \
  masks_batch_size=1 \
  visualize_results=false \
  vid_ids=\[4,12\] \
  model._target_=sam_pt.modeling.sam_pt_interactive.SamPtInteractive \
  model.iterative_refinement_iterations=0 \
  +model.visualize_all_interactions_separately=false \
  +model.visualize_all_interactions_as_mp4=true \
  +model.online=true \
  +model.disable_point_tracking=true \
  +model.interactions_max_per_frame=3
```

Adjust the `interactions_max_per_frame` parameter to control the number of points/interactions that can be used to prompt SAM per frame. 

## What's Next?

Now that you've successfully run your experiments with SAM-PT and come to the end of the documentation, there are several possibilities to proceed further:

1. **Analyze the Results:** You can analyze both qualitative and quantitative results of your runs on your [Weights & Biases (wandb)](https://wandb.ai/home) account.

2. **Integrate a New Point Tracker:** If you're interested in exploring new point tracking techniques within SAM-PT, you can integrate a new point tracker by following our API as defined in the [`PointTracker`](../sam_pt/point_tracker/tracker.py) class. For more details on how to do this, see how RAFT implements the API in the [RaftPointTracker](../sam_pt/point_tracker/raft/tracker.py) class.

3. **Use SAM-PT in Your Own Projects:** If you have specific projects or applications where video object segmentation can be beneficial, especially in zero-shot settings with arbitrary objects and segments, SAM-PT can be a useful tool. For instance, you may want to include it in a media editing tool or use it within a broader computer vision pipeline.

4. **Integrate Other VIS/VOS Methods:** Our codebase is designed to be flexible. You can integrate other Video Object Segmentation (VOS) or Video Instance Segmentation (VIS) methods by implementing the VOS evaluation API defined by the [VOSEvaluator](../sam_pt/vos_eval/evaluator.py) or by Detectron2 for VIS as we did in [SamBasedVisToVosAdapter](../sam_pt/modeling/vis_to_vos_adapter.py).

5. **Integrate Different SAM Variants:** You can also experiment with integrating different SAM variants into our framework. For example, see how we integrated HQ-SAM [here](../configs/model/sam/samhq_vit_huge.yaml) or MobileSAM [here](../configs/model/sam/sam_mobile_vit_tiny.yaml).

These next steps will allow you to dive deeper into the capabilities of SAM-PT, customize it to your liking, and potentially improve upon its performance. Happy experimenting!
