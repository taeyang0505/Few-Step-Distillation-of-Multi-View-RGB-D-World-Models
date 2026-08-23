#!/bin/bash
set -ex
source ~/miniconda3/etc/profile.d/conda.sh
conda activate self_forcing
hf download zhuhz22/Causal-Forcing chunkwise/ar_diffusion.pt chunkwise/causal_ode.pt chunkwise/causal_cd.pt --local-dir ~/Causal-Forcing/ckpts
echo "CF2_DONE"
