#!/bin/bash
# GPU가 빌 때까지 대기 → ① 시간 재측정(bench_timing_final, ~7분) → ② cereal 파이프라인(~5.5h) 순차 실행
# 사용: setsid nohup bash ~/run_when_free.sh > ~/run_when_free.log 2>&1 < /dev/null &
set -u
echo "[$(date +%H:%M:%S)] GPU 대기 시작 (5분 간격 확인, 3GB 미만이면 빈 것으로 판정)"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ -n "$used" ] && [ "$used" -lt 3000 ]; then
    echo "[$(date +%H:%M:%S)] GPU 비었음 (${used} MiB) — 시작"
    break
  fi
  echo "[$(date +%H:%M:%S)] 사용 중 (${used} MiB) — 대기"
  sleep 300
done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy

echo "=== [1/2] 시간 재측정 (bench_timing_final) ==="
cd ~/4dgen
python notebooks/bench_timing_final.py 2>&1 | tee ~/timing_final.log
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "[경고] 시간 측정 실패(exit $rc) — 파이프라인은 계속 진행"
fi

echo "=== [2/2] cereal 파이프라인 (object id=39) ==="
GEO4D_OBJECT_ID=39 bash ~/d1_pipeline.sh cereal_box 2>&1 | tee ~/d1_cereal.log
echo "[$(date +%H:%M:%S)] RUN_WHEN_FREE_ALL_DONE (pipeline exit ${PIPESTATUS[0]})"
