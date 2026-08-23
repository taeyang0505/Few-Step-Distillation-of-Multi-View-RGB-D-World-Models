#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_frame_diag.py --configs T25 S3
echo FRAME_DIAG_DONE
