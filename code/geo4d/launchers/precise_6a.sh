#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_precise_6a.py
echo PRECISE_DONE
