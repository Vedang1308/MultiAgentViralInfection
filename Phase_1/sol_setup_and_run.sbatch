#!/bin/bash
#SBATCH --job-name=phase1_baselines
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=phase1_%j.log
#SBATCH --error=phase1_%j.err

echo "Loading modules and environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogint

echo "Starting Xvfb for headless 3D rendering..."
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1

echo "Configuring scratch cache..."
export TMPDIR=/scratch/$USER/tmp
export TMP=/scratch/$USER/tmp
export TEMP=/scratch/$USER/tmp
mkdir -p $TMPDIR
export HF_HOME=$TMPDIR

echo "Executing Phase 1 Pure Baselines..."
python Phase_1/enacttom_loader.py
python Phase_1/ai2thor_loader.py
python Phase_1/virtualhome_loader.py

echo "Job complete!"
