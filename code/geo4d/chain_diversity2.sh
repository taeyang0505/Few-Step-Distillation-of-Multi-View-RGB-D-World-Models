#!/bin/bash
# ① 앵커 교란 해소: T3r에도 앵커를 걸어 best-of-N 재측정 (T3r vs T3rb, S3 vs S3b)
# ② 궤적 다양성: 그리퍼 3D 경로가 시드마다 갈리는지 (계획 관점의 다양성)
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate video_policy; cd ~/4dgen
wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 3000 ]; do sleep 300; done; sleep 20; }

echo "=== [1/2] best-of-N 앵커 교란 해소 (T3r/T3rb/S3/S3b, 10샘플×시드8, ~35분) ==="
wait_gpu
python notebooks/bench_best_of_n.py --configs T3r T3rb S3 S3b --n_batches 10 --seeds 8 --fast --tag _anchor_ctrl 2>&1 | tail -14
echo "BON_CTRL_RC=${PIPESTATUS[0]}"

echo "=== [2/2] 궤적 다양성 (T25/T3r/S3b/S5b, 10샘플×시드8, ~40분) ==="
wait_gpu
python notebooks/bench_traj_diversity.py --configs T25 T3r S3b S5b --n_batches 10 --seeds 8 --fast --tag _apple 2>&1 | tail -16
echo "TRAJ_RC=${PIPESTATUS[0]}"
df -h ~ | tail -1
echo CHAIN_DIVERSITY2_DONE
