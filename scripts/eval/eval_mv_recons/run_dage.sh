#!/bin/bash

set -e

python evaluation/mv_recon/run_dage.py \
    --output_dir "eval_results/mv_recon/dage/" \
    --size 512 \
    --checkpoint "checkpoints/model.pt"