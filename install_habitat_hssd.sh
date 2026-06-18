#!/bin/bash
# install_habitat_hssd.sh
# Automated Habitat and HSSD Installation for SOL Supercomputer (Headless A100)

set -e

echo "=========================================="
echo " Starting Habitat + HSSD Installation"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh

echo "Configuring Conda to use Scratch space for packages..."
export CONDA_PKGS_DIRS=/scratch/$USER/tmp/conda_pkgs
export ENV_PREFIX=/scratch/$USER/tmp/envs/enacttom-habitat
mkdir -p $CONDA_PKGS_DIRS
mkdir -p /scratch/$USER/tmp/envs

echo "[1/4] Creating dedicated Python 3.9 environment for Habitat in Scratch..."
conda create -p $ENV_PREFIX python=3.9.2 cmake=3.14.0 -y
conda activate $ENV_PREFIX

echo "[2/4] Installing PyTorch and Headless Habitat-Sim..."
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 "mkl<2025" "intel-openmp<2025" -c pytorch -c nvidia -y
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y

echo "[3/4] Installing Habitat-Lab and ML dependencies..."
HABITAT_LAB_COMMIT=094d6be2f9d057e4781a68ae792132895fd4d3d0
python -m pip install "git+https://github.com/facebookresearch/habitat-lab.git@${HABITAT_LAB_COMMIT}#subdirectory=habitat-lab"
python -m pip install "git+https://github.com/facebookresearch/habitat-lab.git@${HABITAT_LAB_COMMIT}#subdirectory=habitat-baselines"

# Install EnactToM dependencies
if [ ! -f "./Others/EnactTom/setup.py" ]; then
    echo "EnactToM source files are missing (likely due to nested git tracking). Cloning fresh..."
    rm -rf ./Others/EnactTom
    git clone https://github.com/UCSB-AI/EnactTom ./Others/EnactTom
fi

cd ./Others/EnactTom
python -m pip install pillow==10.4.0 numpy-quaternion==2023.0.4 matplotlib==3.6.3 opencv-python==4.10.0.82 openai==2.24.0 pandas pytest unified-planning==1.3.0 up-fast-downward==0.5.2
python -m pip install -e . --no-deps

# Install ML libraries so our wrapper script works in this env
python -m pip install transformers accelerate bitsandbytes

echo "[4/4] Downloading HSSD Dataset and Assets via Git LFS to Scratch..."
export HABITAT_DATA_DIR=/scratch/$USER/habitat_data
mkdir -p $HABITAT_DATA_DIR

# Store current repo root
REPO_ROOT=$(pwd)
cd $HABITAT_DATA_DIR

# Initialize git-lfs 
git lfs install

if [ ! -d "objects_ovmm" ]; then
    echo "Cloning OVMM Objects..."
    git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects objects_ovmm --recursive
    git -C objects_ovmm lfs pull
fi

if [ ! -d "versioned_data/hssd-hab" ]; then
    echo "Cloning HSSD Scenes..."
    mkdir -p versioned_data
    git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab versioned_data/hssd-hab
    git -C versioned_data/hssd-hab lfs pull
    ln -sfn versioned_data/hssd-hab hssd-hab
fi

# Link to EnactToM
cd $REPO_ROOT
mkdir -p ./Others/EnactTom/data
ln -sfn $HABITAT_DATA_DIR/objects_ovmm ./Others/EnactTom/data/objects_ovmm
ln -sfn $HABITAT_DATA_DIR/versioned_data ./Others/EnactTom/data/versioned_data
ln -sfn $HABITAT_DATA_DIR/hssd-hab ./Others/EnactTom/data/hssd-hab

echo "=========================================="
echo " Installation Complete!"
echo " IMPORTANT: Because Habitat requires Python 3.9, you must run EnactToM in the new environment:"
echo "   conda activate /scratch/\$USER/tmp/envs/enacttom-habitat"
echo "   python Phase_1/enacttom_loader.py"
echo "=========================================="
