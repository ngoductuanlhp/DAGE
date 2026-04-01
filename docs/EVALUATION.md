# Evaluation

DAGE is evaluated on four tasks: video depth estimation (based on [GeometryCrafter](https://github.com/TencentARC/GeometryCrafter) benchmark), monocular depth estimation (based on [MoGe](https://github.com/microsoft/MoGe) benchmark), camera pose estimation, and multi-view reconstruction.

## Video Depth Estimation

Evaluates depth prediction quality on the [GeometryCrafter](https://github.com/TencentARC/GeometryCrafter) benchmark.

### Data Preparation

Follow the dataset preparation instructions from the GeometryCrafter repo: [Dataset Evaluation](https://github.com/TencentARC/GeometryCrafter/blob/5956c844c2764db17f294618ff86e53f4bfb8620/README.md#-dataset-evaluation).

### Usage

```bash
bash scripts/eval/eval_video_depth_geocrafter/run_dage.sh checkpoints/model.pt eval_results/dage

# Run on a single benchmark
python evaluation/video_depth_geocrafter/run_dage.py \
    --checkpoint checkpoints/model.pt \
    --benchmark sintel \
    --output_dir eval_results/dage \
    --lr_max_size 252 \
    --seed 42
```

Edit the `BENCHMARKS` array in the shell script to select which datasets to evaluate.

Results are saved as `eval_metrics.json` per benchmark.


## Monocular Depth Estimation

Evaluates per-frame monocular depth/pointmap quality on the [MoGe](https://github.com/microsoft/MoGe) benchmark.

### Data Preparation

Follow the dataset preparation instructions from the MoGe repo: [Benchmarks](https://github.com/microsoft/MoGe/blob/main/docs/eval.md#benchmarks).


### Usage

```bash
# Run on all default benchmarks (NYUv2, ETH3D, Sintel, GSO, KITTI, Spring, HAMMER, iBims-1, DDAD, DIODE)
bash scripts/eval/eval_mono_depth_moge/run_dage.sh checkpoints/model.pt eval_results/mono_depth_moge/dage

# Run on a single benchmark
python evaluation/monodepth_moge/run_dage.py \
    --checkpoint checkpoints/model.pt \
    --benchmark NYUv2 \
    --output_dir eval_results/mono_depth_moge/dage \
    --lr_max_size 252 \
    --seed 42
```

Edit the `BENCHMARKS` array in the shell script to select which datasets to evaluate.


## Camera Pose Estimation

Evaluates camera pose estimation using trajectory metrics.

### Data Preparation

- **Angular**: We follow [VGGT](https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/README.md) to prepare Co3Dv2, and [Pi3](https://github.com/yyfz/Pi3/blob/evaluation/datasets/preprocess/prepare_re10k.sh) for RealEstate10k preprocessing.
- **Distance**: We follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) to prepare Sintel, TUM-dynamics, and ScanNetv2.

### Usage

```bash
# Run on all datasets (Sintel, TUM, ScanNet)
bash scripts/eval/eval_pose/run_dage.sh

# Run on a single dataset
python evaluation/relpose/eval_pose_dage.py \
    --checkpoint checkpoints/model.pt \
    --eval_dataset sintel \
    --output_dir eval_results/pose/dage/sintel
```

## Multi-View Reconstruction

Evaluates 3D pointmap reconstruction quality against ground-truth point clouds.

### Data Preparation

We follow [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare 7-Scenes and Neural-NRGBD.

### Usage

```bash
# Run evaluation
bash scripts/eval/eval_mv_recons/run_dage.sh

# Or directly
python evaluation/mv_recons/run_dage.py \
    --checkpoint checkpoints/model.pt \
    --output_dir eval_results/mv_recon/dage \
```
