#!/bin/bash
set -ex
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n self_forcing -c conda-forge --override-channels python=3.10
conda activate self_forcing
pip install --no-cache-dir torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir \
  "diffusers==0.31.0" "transformers>=4.49.0" tokenizers accelerate \
  tqdm imageio easydict ftfy imageio-ffmpeg "numpy==1.26.4" wandb \
  omegaconf einops "av==13.1.0" opencv-python open_clip_torch \
  lmdb matplotlib sentencepiece "pydantic==2.10.6" scikit-image \
  "huggingface_hub[cli]" flask flask-socketio dominate starlette
cd ~/Self-Forcing
python setup.py develop
mkdir -p wan_models
ln -sfn /home/sun4208/models/Wan2.1-T2V-1.3B wan_models/Wan2.1-T2V-1.3B
ls -la wan_models/
ls -lh checkpoints/self_forcing_dmd.pt
echo "SF_SETUP_DONE"
