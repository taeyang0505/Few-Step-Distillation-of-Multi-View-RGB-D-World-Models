#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_precise_6a.py --configs S3b S1b --n_batches 20 --n_div 3 --fast --compile --tag _main20_compile
echo COMPILE20_DONE
