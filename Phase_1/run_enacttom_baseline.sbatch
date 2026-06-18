#!/bin/bash
#SBATCH --job-name=enacttom_baseline
#SBATCH --output=/scratch/%u/logs/enacttom_%j.out
#SBATCH --error=/scratch/%u/logs/enacttom_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Make sure we fail on error
set -e

echo "=========================================================="
echo "Starting EnactToM Baseline Execution on Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================================="

# Ensure output directory exists
mkdir -p /scratch/$USER/logs

# Initialize Conda using the correct system path
# If conda is not in standard paths, we assume it's in the user's home or opt
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Warning: Could not find conda.sh. Assuming conda is already in PATH."
fi

# Activate the dedicated habitat environment
conda activate /scratch/$USER/tmp/envs/enacttom-habitat

# Navigate to working directory
cd ~/COGINT

# Apply LLaVA-1.5 patches to the EnactToM repository
echo "Applying LLaVA local provider patches to EnactToM..."
cp Phase_1/patches/llava_local.py Others/EnactTom/habitat_llm/llm/llava_local.py
cp Phase_1/patches/llava_local.yaml Others/EnactTom/habitat_llm/conf/llm/llava_local.yaml

# Step 1: Initialize the static benchmark pool (if not already done)
echo "Initializing 300-task EnactToM static benchmark pool..."
python Phase_1/initialize_benchmark_pool.py

# Step 2: Run the loader via headless X server (for EGL/OpenGL rendering)
echo "Launching Habitat-Sim Headless Pipeline via enacttom_loader.py..."
export DISPLAY=:0
export EGL_DEVICE_ID=0
export PYTHONPATH=$PYTHONPATH:$(pwd)/Others/EnactTom

# Run the native EnactToM loader
python Phase_1/enacttom_loader.py

echo "=========================================================="
echo "EnactToM Baseline Completed at $(date)"
echo "=========================================================="
