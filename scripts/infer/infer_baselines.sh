#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=0 python inference/infer_pi3.py \
    --checkpoint="yyfz233/Pi3" \
    --output_dir="quali_results/pi3"

CUDA_VISIBLE_DEVICES=0 python inference/infer_vggt.py \
    --checkpoint="facebook/VGGT-1B" \
    --output_dir="quali_results/vggt"

CUDA_VISIBLE_DEVICES=0 python inference/infer_cut3r.py \
    --checkpoint="checkpoint/cut3r_512_dpt_4_64.pth" \
    --output_dir="quali_results/cut3r"