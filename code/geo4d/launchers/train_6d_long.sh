#!/bin/bash
# 6d(자기 앵커 loss) 2000 -> 4000 step 연장. dmd_6d/dmd_gen.pt(step 2000)+dmd_fake.pt에서 resume. OOM 시 GPU 빌 때까지 대기 후 재개.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
OUT=~/Geo4D/dmd_6d
gpu_free() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 2000 ]; }
for attempt in 1 2 3 4 5 6 7 8; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python notebooks/geo4d_dmd_train.py --anchor_weight 1 --max_steps 4000 --keep_steps 2400,3200 --out_dir $OUT --resume
  rc=$?
  if [ $rc -eq 0 ]; then echo TRAIN_6D_LONG_DONE; break; fi
  echo "TRAIN_CRASHED rc=$rc attempt=$attempt $(date '+%H:%M') — GPU가 빌 때까지 대기 후 재개"
  until gpu_free; do sleep 60; done
  sleep 30
done
[ $rc -ne 0 ] && { echo "TRAIN_6D_LONG_FAILED after $attempt attempts"; exit 1; }
for s in 2400 3200; do
  [ -f $OUT/dmd_gen_step$s.pt ] && python notebooks/bench_precise_6a.py --student_ckpt $OUT/dmd_gen_step$s.pt --configs T25 S3 S3b --n_batches 20 --n_div 3 --fast --tag _6d_s$s
done
python notebooks/bench_precise_6a.py --student_ckpt $OUT/dmd_gen.pt --configs T25 S3 S3b --n_batches 20 --n_div 3 --fast --tag _6d_s4000
echo EVAL_6D_LONG_DONE
