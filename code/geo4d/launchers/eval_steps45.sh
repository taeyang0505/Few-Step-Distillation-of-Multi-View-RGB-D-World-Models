#!/bin/bash
# 4·5스텝 student 평가 (학습 없음): 학습된 σ 반복 스케줄. 6a-1600 기준
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_precise_6a.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 S3b S4b S5b --n_batches 20 --n_div 3 --fast --tag _steps45
python notebooks/bench_policy_proxy.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 S3b S4b S5b --n_batches 20 --fast --tag _steps45
echo STEPS45_DONE
