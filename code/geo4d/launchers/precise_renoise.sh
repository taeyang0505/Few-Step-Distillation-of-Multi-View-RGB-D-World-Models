#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_precise_6a.py --configs T25 T3r T1r S3 S1 --tag _renoise
echo PRECISE_RENOISE_DONE
