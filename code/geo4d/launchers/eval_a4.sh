#!/bin/bash
# A4(3σ+마지막 x0 평균2 = 4호출)·A6(4σ+평균2 = 5호출) 평가 + 대리 지표. 6a-1600 기준, 학습 없음.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_precise_6a.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 S4b A4b A6b --n_batches 20 --n_div 3 --fast --tag _avgfinal
python notebooks/bench_policy_proxy.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 S4b A4b A6b --n_batches 20 --fast --tag _avgfinal
echo AVGFINAL_DONE
