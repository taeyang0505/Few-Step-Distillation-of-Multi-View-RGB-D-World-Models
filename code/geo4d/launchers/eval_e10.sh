#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_eval_6x.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 T25c S3c S1c --extr_idx 1,0 --tag _affine_e10
echo EVAL_E10_DONE
