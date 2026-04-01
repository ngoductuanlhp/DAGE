#!/bin/bash

set -e

# NOTE usage:
# bash scripts/eval/eval_pose/run_dage.sh

for dataset in "sintel" "tum" "scannet"; do
    python evaluation/relpose/eval_pose_dage.py \
        --output_dir "eval_results/pose/dage/${dataset}" \
        --eval_dataset "${dataset}" \
        --checkpoint "checkpoints/model.pt"
done