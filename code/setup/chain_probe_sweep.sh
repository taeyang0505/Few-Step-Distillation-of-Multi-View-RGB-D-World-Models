#!/bin/bash
# probe(step2000 평가) 완료를 기다렸다가 스윕을 이어서 실행
set -u
echo "[$(date +%H:%M)] probe 완료 대기"
while ! grep -q "CEREAL_PROBE_DONE" ~/cereal_probe.log 2>/dev/null; do
  pgrep -f cereal_ckpt_probe > /dev/null || { echo "probe 프로세스 종료됨(미완). 그래도 스윕 진행"; break; }
  sleep 120
done
echo "[$(date +%H:%M)] 스윕 시작"
bash ~/cereal_ckpt_sweep.sh
