#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python notebooks/geo4d_dmd_train.py --cv_weight 10 --cv_target gt --cv_frames 1 --out_dir ~/Geo4D/dmd_6b2 --max_steps 2000
echo TRAIN_6B2_DONE
python notebooks/bench_eval_6x.py --student_ckpt ~/Geo4D/dmd_6b2/dmd_gen_step1600.pt --configs T25 S3 S3a S1a --tag _6b2_s1600
python notebooks/bench_eval_6x.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 S3 S3a S1a --tag _6a_s1600_v2
echo EVAL_6B2_DONE
