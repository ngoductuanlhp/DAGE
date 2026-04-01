#!/bin/bash

# usage: bash demo.sh
set -e

# NOTE Run with default settings on demo data
CUDA_VISIBLE_DEVICES=0 python inference/infer_dage.py \
    --checkpoint="TuanNgo/DAGE" \
    --output_dir="quali_results/dage" 

# NOTE Run with higher LR resolution (better camera poses, more compute)
# CUDA_VISIBLE_DEVICES=0 python inference/infer_dage.py \
#     --checkpoint="checkpoints/model.pt" \
#     --output_dir="quali_results/dage" \
#     --lr_max_size=518 \

# NOTE Run with higher HR resolution up to 2K (sharper pointmaps)
# CUDA_VISIBLE_DEVICES=0 python inference/infer_dage.py \
#     --checkpoint="checkpoints/model.pt" \
#     --output_dir="quali_results/dage" \
#     --hr_max_size=1920

# NOTE Run with memory-efficient chunking for GPUs with <40GB VRAM (lower chunk_size if OOM)
# CUDA_VISIBLE_DEVICES=0 python inference/infer_dage.py \
#     --checkpoint="checkpoints/model.pt" \
#     --output_dir="quali_results/dage" \
#     --hr_max_size=1920 \
#     --chunk_size=8