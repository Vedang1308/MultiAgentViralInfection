#!/bin/bash
#SBATCH --job-name=phase1_baselines
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=phase1_baselines_%j.log

echo "Starting Phase 1: Establishing Golden Path Baselines"

# Set HuggingFace cache to scratch to save home directory space
export HF_HOME=/scratch/$USER/huggingface_cache
echo "HF_HOME set to: $HF_HOME"

# Setup Python environment
echo "Installing dependencies..."
pip install transformers accelerate bitsandbytes

# Install simulators
pip install ai2thor virtualhome

# Configure Xvfb for headless rendering (required for Unity-based AI2-THOR and VirtualHome)
echo "Starting Xvfb for headless rendering..."
# Check if xvfb is installed, if not, try to install it (might require sudo, but usually pre-installed on clusters)
# sudo apt-get install -y xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1

# Run the loaders
echo "Starting EnactToM Loader..."
python Phase_1/enacttom_loader.py

echo "Starting AI2-THOR Loader..."
python Phase_1/ai2thor_loader.py

echo "Starting VirtualHome Loader..."
python Phase_1/virtualhome_loader.py

echo "Phase 1 Complete."
