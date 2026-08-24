#!/bin/bash
# 최종 표: 60샘플(뷰 120) + 다양성 20샘플×4시드. 설정: teacher / S3b(메인) / S4b / S5b / H3b / H4b.
# MDE(80%): AbsRel ~0.006, 다양성 ~7% — 엄격 마진 판정 가능. 이어서 대리 지표 60샘플.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
gpu_free() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 2000 ]; }
run_retry() {
  for attempt in 1 2 3; do
    "$@" && return 0
    echo "EVAL_CRASHED attempt=$attempt: $* — GPU 대기 후 재시도"
    until gpu_free; do sleep 60; done; sleep 30
  done
  return 1
}
run_retry python notebooks/bench_precise_6a.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt \
  --configs T25 S3b S4b S5b H3b H4b --n_batches 60 --n_div 20 --fast --tag _final60 || { echo FINAL60_FAILED; exit 1; }
echo PRECISE60_DONE
run_retry python notebooks/bench_policy_proxy.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt \
  --configs T25 S3b S5b H4b --n_batches 60 --fast --tag _final60 || { echo PROXY60_FAILED; exit 1; }
echo FINAL60_DONE
