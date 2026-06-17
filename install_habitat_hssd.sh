#!/bin/bash
# install_habitat_hssd.sh
# Automated Habitat and HSSD Installation for SOL Supercomputer (Headless A100)

set -e

echo "=========================================="
echo " Starting Habitat + HSSD Installation"
echo "=========================================="

# 1. Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogint

echo "[1/3] Installing Habitat-Sim and Habitat-Lab for Headless GPU (EGL)..."
# The 'headless' build uses EGL, which is critical for rendering on A100s without X server attachments
conda install -y -c conda-forge -c aihabitat habitat-sim headless withbullet
python -m pip install setuptools==65.5.0 wheel==0.38.4
python -m pip install git+https://github.com/facebookresearch/habitat-lab.git
python -m pip install git+https://github.com/facebookresearch/habitat-lab.git#subdirectory=habitat-baselines

# Install EnactToM dependencies
cd ~/COGINT/Others/EnactTom
python -m pip install -r requirements.txt
python -m pip install -e .
cd ~/COGINT

echo "[2/3] Downloading HSSD Dataset to Scratch Directory..."
export HABITAT_DATA_DIR=/scratch/$USER/habitat_data
mkdir -p $HABITAT_DATA_DIR

# Use habitat-sim's built-in downloader to pull the HSSD scenes
python -m habitat_sim.utils.datasets_download --uids hssd-hab --data-path $HABITAT_DATA_DIR

echo "[3/3] Linking HSSD Dataset to EnactToM..."
# EnactToM expects datasets in data/scene_datasets
mkdir -p ~/COGINT/Others/EnactTom/data/scene_datasets
ln -sf $HABITAT_DATA_DIR/scene_datasets/hssd-hab ~/COGINT/Others/EnactTom/data/scene_datasets/hssd-hab

echo "=========================================="
echo " Installation Complete!"
echo " You can now run: python Phase_1/enacttom_loader.py"
echo "=========================================="
