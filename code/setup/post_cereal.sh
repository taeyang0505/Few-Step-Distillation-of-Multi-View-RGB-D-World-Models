#!/bin/bash
# cereal 파이프라인 완료 대기 → GPU 대기 → ① 영역별 지각 측정(apple) → ② 정성 패키지 생성
# 사용: setsid nohup bash ~/post_cereal.sh > ~/post_cereal.log 2>&1 < /dev/null &
set -u
LOG=~/d1_cereal.log
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
wait_gpu() { while [ "$(gpu_used)" -ge 3000 ]; do echo "[$(date +%H:%M)] GPU 사용 중($(gpu_used) MiB) — 대기"; sleep 300; done; sleep 20; }

echo "[$(date +%H:%M)] cereal 파이프라인 완료 대기"
while true; do
  if grep -q "ALL_DONE" $LOG 2>/dev/null; then echo "[$(date +%H:%M)] cereal 완료 감지"; break; fi
  if grep -q "FAILED" $LOG 2>/dev/null; then echo "[$(date +%H:%M)] cereal 실패 — 후속 작업 중단"; exit 1; fi
  if ! pgrep -f d1_pipeline.sh > /dev/null; then echo "[$(date +%H:%M)] 파이프라인 프로세스 없음(비정상 종료) — 중단"; exit 1; fi
  sleep 120
done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen

echo "=== [1/2] 영역별 지각 측정 (apple, 20샘플, ~15분) ==="
wait_gpu
python notebooks/bench_region_perceptual.py --configs T25 S3b S5b H4b --n_batches 20 --fast --tag _apple 2>&1 | tee ~/region_perceptual.log
echo "REGION_RC=${PIPESTATUS[0]}"

echo "=== [2/2] 정성 패키지 (8샘플 x 2뷰, ~10분) ==="
wait_gpu
python notebooks/bench_qual_package.py --n_batches 8 --views left right --fast --out ~/Geo4D/bench_out/qual_package 2>&1 | tee ~/qual_package.log
echo "QUAL_RC=${PIPESTATUS[0]}"

df -h ~ | tail -1
echo POST_CEREAL_ALL_DONE
