# Training

DAGE uses a three-stage training pipeline with progressive refinement. All stages use the same 18 training datasets with different resolution/augmentation settings.

## Data Preparation

We follow [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) to prepare training data. The following datasets are used:

| Dataset | Weight |
| :--- | :--- |
| CO3D | 94.0 |
| Wild-RGBD | 56.0 |
| ArkitScenes | 56.0 |
| ScanNet++ | 17.0 |
| ScanNet | 23.0 |
| Hypersim | 12.0 |
| MP3D | 19.0 |
| BlendedMVS | 22.0 |
| MegaDepth | 22.0 |
| Waymo | 50.0 |
| Virtual KITTI 2 | 6.0 |
| TartanAir | 56.0 |
| Dynamic Replica | 26.0 |
| Spring | 1.2 |
| Bedlam | 20.0 |
| MVS Synth | 2.4 |
| GTA SFM | 10.0 |
| Point Odyssey | 24.0 |

Follow CUT3R to download and preprocess the raw data first. Then, for each dataset, run the corresponding `generate_parquet_file()` function in its dataset definition file to create the metadata parquet used during training. For example, see `training/dataloaders/datasets/video_datasets_new/depth_scannetpp_new.py`.

Dataset paths are configured in `training/dataloaders/config.py`. Update the paths to match your local setup.

## Training Stages

Before starting training, first merge the source checkpoints into a DAGE initialization checkpoint:

```bash
python scripts/merge_checkpoints.py \
  --pi3-ckpt <pi3_checkpoint_or_hf_repo> \
  --mogev2-ckpt <moge_v2_checkpoint_or_hf_repo> \
  --output checkpoints/merged_pi3_mogev2.pt
```

Use the merged checkpoint as the initialization checkpoint in the training config of stage1 

All stages are launched via the same training script:

```bash
bash training/train_fp16.sh <config.yaml> <training_script.py>
```

The launcher auto-detects GPU count and runs via `accelerate` with FP16 mixed precision. Multi-GPU training is automatically enabled when more than one GPU is available.

### Stage 1 — Base Training

```bash
bash training/train_fp16.sh training/training_configs/train_dage_stage1.yaml training/train_dage_stage1.py
```

| Setting | Value |
| :--- | :--- |
| Learning rate | `1e-4` |
| Max steps | 40,000 |
| Batch size | 48 frames/GPU (dynamic batching, 2-24 frames per clip) |
| LR resolution area | 100K-255K pixels |
| HR resolution area | 300K-900K pixels (2-6 frames per clip) |

Stage 1 uses feature distillation from a teacher model. The config uses two task groups: `tasks1` (with distillation) and `tasks2` (without distillation, higher HR resolution with fewer frames).

Set `pretrained_model_name_or_path` in the config to your starting checkpoint.

### Stage 2 — High-resolution Fine-tuning

```bash
bash training/train_fp16.sh training/training_configs/train_dage_stage2.yaml training/train_dage_stage2.py
```

| Setting | Value |
| :--- | :--- |
| Learning rate | `1e-5` |
| Max steps | 10,000 |
| Batch size | 48 frames/GPU |
| Resolution area | 100K-255K pixels |

Set `pretrained_model_name_or_path` to the Stage 1 checkpoint.

### Stage 3 — Sky Segmentation (optional)

```bash
bash training/train_fp16.sh training/training_configs/train_dage_stage3.yaml training/train_dage_stage3.py
```

| Setting | Value |
| :--- | :--- |
| Learning rate | `1e-5` |
| Max steps | 10,000 |
| Batch size | 48 frames/GPU |
| Resolution area | 100K-255K pixels |

Set `pretrained_model_name_or_path` to the Stage 2 checkpoint.

## Loss Functions

All stages use the following losses (configured in the YAML files):

| Loss | Config flag | Description |
| :--- | :--- | :--- |
| Point map L1 | `use_moge: true` | L1 loss on local 3D point maps |
| Edge loss | `use_edge_loss: true` | Multi-scale Scharr derivative loss for sharp boundaries |
| Depth loss | `use_real_loss: true` | Scale-shift invariant depth loss |
| Camera pose | `camera_loss_weight: 0.1` | Supervised camera pose loss |
| Feature distillation | `feat_distill_type: cosine` | Cosine similarity distillation from teacher (Stage 1 only) |
| Metric scale | `metric_scale_loss: true` | Metric scale prediction loss |

Loss implementations are in `training/loss/`.


## Data Augmentation

We implement a video-based extension of the augmentation pipeline from [MoGe](https://github.com/microsoft/MoGe) (originally designed for single images) to support temporally consistent augmentation across video frames. Augmentation is configured per-dataset via `moge_augmentation` in the YAML configs:
