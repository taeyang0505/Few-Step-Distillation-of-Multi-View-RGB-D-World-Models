#!/bin/bash
# 6a 레시피 seed 재학습 2회 (run 간 편차 측정). 각 run: 2000 step + 평가 + 대리 지표.
# 공간 절약: 각 run 종료 후 dmd_fake.pt와 dmd_gen.pt(step2000)는 삭제하고 step1600만 보존.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
gpu_free() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 2000 ]; }
for seed in 1 2; do
  OUT=~/Geo4D/dmd_6a_s$seed
  for attempt in 1 2 3 4 5 6; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python notebooks/geo4d_dmd_train.py --seed $seed --max_steps 2000 --keep_steps 1600 --out_dir $OUT --resume
    rc=$?; [ $rc -eq 0 ] && break
    echo "TRAIN_CRASHED seed=$seed rc=$rc attempt=$attempt $(date '+%H:%M') — GPU 대기 후 재개"
    until gpu_free; do sleep 60; done; sleep 30
  done
  [ $rc -ne 0 ] && { echo "SEED${seed}_FAILED"; exit 1; }
  echo TRAIN_SEED${seed}_DONE
  rm -f $OUT/dmd_fake.pt $OUT/dmd_gen.pt
  python notebooks/bench_precise_6a.py --student_ckpt $OUT/dmd_gen_step1600.pt --configs T25 S3b --n_batches 20 --n_div 3 --fast --tag _6a_seed$seed
  python notebooks/bench_policy_proxy.py --student_ckpt $OUT/dmd_gen_step1600.pt --configs T25 S3b --n_batches 20 --fast --tag _6aseed$seed
  echo EVAL_SEED${seed}_DONE
done
echo SEEDVAR_ALL_DONE
