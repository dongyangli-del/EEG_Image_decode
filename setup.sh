#!/bin/bash
# ============================================================
# EEG Image Decode - Environment Setup Script
# ============================================================
# This script creates a conda environment with all dependencies
# for reproducing the experiments in the paper.
#
# Usage: . setup.sh
# ============================================================

set -e

ENV_NAME="BCI"
PYTHON_VERSION="3.12"

echo "============================================================"
echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
echo "============================================================"

# Create conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME

echo "Installing base packages via conda..."
conda install numpy matplotlib tqdm scikit-image jupyterlab -y
conda install -c conda-forge accelerate -y

echo "Installing PyTorch ecosystem (CUDA 12.4)..."
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing Hugging Face packages..."
pip install transformers==4.36.0
pip install diffusers==0.30.0
pip install huggingface-hub==0.30.2
pip install accelerate==1.5.2

echo "Installing CLIP packages..."
pip install git+https://github.com/openai/CLIP.git
pip install open_clip_torch
pip install clip-retrieval

echo "Installing EEG processing packages..."
pip install braindecode==0.8.1
pip install mne==1.9.0

echo "Installing image generation packages..."
pip install dalle2-pytorch==1.15.6
pip install pytorch-msssim==1.0.0
pip install kornia==0.8.0

echo "Installing deep learning utilities..."
pip install einops==0.8.1
pip install info-nce-pytorch==0.1.0
pip install reformer_pytorch==1.4.4

echo "Installing logging and visualization..."
pip install wandb==0.19.10
pip install seaborn==0.13.2

echo "Installing other utilities..."
pip install ftfy==6.3.1
pip install regex==2024.11.6
pip install h5py==3.13.0
pip install pandas==2.3.3
pip install imageio==2.37.0
pip install scipy==1.15.3
pip install scikit-learn==1.6.1

echo "============================================================"
echo "Environment setup complete!"
echo "Activate with: conda activate $ENV_NAME"
echo "============================================================"
