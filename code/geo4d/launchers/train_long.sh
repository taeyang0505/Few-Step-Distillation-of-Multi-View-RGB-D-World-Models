#!/bin/bash
# train_long: 6a 레시피 4000 step. OOM 등으로 죽으면 GPU가 빌 때까지 기다렸다가 마지막 저장(200 step 단위)부터 재개. 최대 8회.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
OUT=~/Geo4D/dmd_6a_long
gpu_free() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 2000 ]; }
for attempt in 1 2 3 4 5 6 7 8; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python notebooks/geo4d_dmd_train.py --max_steps 4000 --keep_steps 1600,2400,3200 --out_dir $OUT --resume
  rc=$?
  if [ $rc -eq 0 ]; then echo TRAIN_LONG_DONE; break; fi
  echo "TRAIN_CRASHED rc=$rc attempt=$attempt $(date '+%H:%M') — GPU가 빌 때까지 대기 후 재개"
  until gpu_free; do sleep 60; done
  sleep 30
done
[ $rc -ne 0 ] && { echo "TRAIN_LONG_FAILED after $attempt attempts"; exit 1; }
for s in 1600 2400 3200; do
  [ -f $OUT/dmd_gen_step$s.pt ] && python notebooks/bench_precise_6a.py --student_ckpt $OUT/dmd_gen_step$s.pt --configs T25 S3b --n_batches 20 --n_div 3 --fast --tag _long_s$s
done
python notebooks/bench_precise_6a.py --student_ckpt $OUT/dmd_gen.pt --configs T25 S3b --n_batches 20 --n_div 3 --fast --tag _long_s4000
echo EVAL_LONG_DONE
