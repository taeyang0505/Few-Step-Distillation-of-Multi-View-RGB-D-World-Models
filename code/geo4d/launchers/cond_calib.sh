#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_cond_calib.py
echo CALIB_DONE
