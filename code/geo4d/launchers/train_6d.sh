#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python notebooks/geo4d_dmd_train.py --anchor_weight 1 --max_steps 2000 --keep_steps 1000,1600 --out_dir ~/Geo4D/dmd_6d
echo TRAIN_6D_DONE
for s in 1600 2000; do ck=~/Geo4D/dmd_6d/dmd_gen_step$s.pt; [ $s = 2000 ] && ck=~/Geo4D/dmd_6d/dmd_gen.pt; python notebooks/bench_precise_6a.py --student_ckpt $ck --configs T25 S3 S3b --n_batches 20 --n_div 3 --fast --tag _6d_s$s; done
echo EVAL_6D_DONE
