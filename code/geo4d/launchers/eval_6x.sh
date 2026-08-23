#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_eval_6x.py --configs T25 T4 T1 S3 S3a S1 S1a --tag _6a_anchor
echo EVAL6X_DONE
