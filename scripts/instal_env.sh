#!/usr/bin/env bash

# Re-exec with bash when the script is launched via `sh`.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

# Exit on error
set -euo pipefail

setup_conda() {
    local conda_bin=""

    if command -v conda >/dev/null 2>&1; then
        conda_bin="$(command -v conda)"
    else
        for candidate in \
            "$HOME/miniconda3/bin/conda" \
            "$HOME/anaconda3/bin/conda" \
            "$HOME/mambaforge/bin/conda" \
            "$HOME/miniforge3/bin/conda"
        do
            if [ -x "$candidate" ]; then
                conda_bin="$candidate"
                export PATH="$(dirname "$candidate"):$PATH"
                break
            fi
        done
    fi

    if [ -z "$conda_bin" ]; then
        echo "Conda was not found."
        echo "Install Miniconda/Anaconda first, or add conda to PATH, then rerun:"
        echo "  bash scripts/instal_env.sh"
        exit 1
    fi

    eval "$("$conda_bin" shell.bash hook)"
}

echo "Setting up environment..."

setup_conda


echo "Creating conda environment..."
conda create --name=dage python=3.10 -y
conda activate dage

# Install pytorch (check your cuda version and install the compatible pytorch)
echo "Installing pytorch..."
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130


# Install mkl
conda install -y "mkl<2024.1" "intel-openmp<2024.1"

# Install other dependencies and local package
echo "Installing local package..."
pip install -e .

# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900 
pip install evo einops kornia roma gdown

pip install decord h5py mediapy fire
pip install viser pyliblzfse open3d

# NOTE for sky segmentation
pip install albumentations
pip install segmentation_models_pytorch
pip install numpy==1.26.4 --force-reinstall

# sudo apt install -y ffmpeg
conda install -c conda-forge ffmpeg=4.2.2 -y
