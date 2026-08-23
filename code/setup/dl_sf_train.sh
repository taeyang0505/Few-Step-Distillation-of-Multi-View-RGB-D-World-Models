#!/bin/bash
set -ex
source ~/miniconda3/etc/profile.d/conda.sh
conda activate self_forcing
hf download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir ~/Self-Forcing
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir ~/Self-Forcing/prompts
echo "SF_TRAIN_ASSETS_DONE"
