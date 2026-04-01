#!/bin/bash

# $1: config file
# $2: training script

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Error: please provide a YAML config file."
    echo "Usage: $0 <config_file.yaml> <training_script.py>"
    exit 1
fi

config_file=$1

if [ -z "$2" ]; then
    echo "Error: please provide a training script."
    echo "Usage: $0 <config_file.yaml> <training_script.py>"
    exit 1
fi

# Get number of GPUs
num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")

if [ $num_gpus -gt 1 ]; then
    USE_MULTI_GPU="--multi_gpu"
else
    USE_MULTI_GPU=""
fi

# Launch training
set -e
set -x

# TRAINING_SCRIPT="training_unip/train_moge_temporal.py"
# TRAINING_SCRIPT="training_unip/train_moge_temporal_track.py"
# TRAINING_SCRIPT="training_unip/train_moge_temporal_scale.py"


accelerate launch \
    --num_machines 1 \
    $USE_MULTI_GPU \
    --num_processes $num_gpus \
    --mixed_precision no \
    --dynamo_backend no \
    $2 \
    $config_file

