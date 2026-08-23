#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_precise_6a.py --configs T25 T4 T3r S3b S1b --n_batches 20 --n_div 3 --tag _main20
echo MAIN20_DONE
