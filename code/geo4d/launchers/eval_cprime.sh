#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_eval_6x.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 T25b S3 S3a S3b S1a S1b --tag _cprime
echo EVAL_CPRIME_DONE
