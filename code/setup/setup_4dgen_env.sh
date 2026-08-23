#!/bin/bash
set -ex
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n video_policy -c conda-forge --override-channels python=3.10
conda activate video_policy
pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir \
  "pytorch-lightning==2.0.1" "lightning-utilities" \
  "diffusers==0.29.1" "transformers==4.45.2" "tokenizers==0.20.1" accelerate safetensors "peft==0.11.1" "timm==1.0.7" \
  "hydra-core==1.1.1" "omegaconf==2.1.2" "antlr4-python3-runtime==4.8" \
  einops "kornia==0.6.9" "open-clip-torch==2.26.1" \
  "open3d==0.18.0" "imgaug==0.4.0" "opencv-python==4.6.0.66" matplotlib imageio imageio-ffmpeg "av==12.1.0" "moviepy==1.0.3" \
  "zarr==2.18.3" "numcodecs==0.12.1" webdataset scipy scikit-image scikit-learn pandas "numpy==1.26.4" \
  fire tqdm rich wandb tensorboardx gdown ftfy regex fairscale lpips \
  "huggingface_hub" "datasets==2.14.4" ipykernel
echo "SETUP_DONE"
