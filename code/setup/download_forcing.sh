#!/bin/bash
set -ex
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
git clone https://github.com/guandeh17/Self-Forcing ~/Self-Forcing 2>/dev/null || echo "Self-Forcing 이미 존재"
git clone https://github.com/thu-ml/Causal-Forcing ~/Causal-Forcing 2>/dev/null || echo "Causal-Forcing 이미 존재"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --exclude "assets/*" "*.mp4" "*.gif" "examples/*" \
  --local-dir ~/models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
  --local-dir ~/Self-Forcing
huggingface-cli download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt \
  --local-dir ~/Causal-Forcing/ckpts
echo "FORCING_DONE"
