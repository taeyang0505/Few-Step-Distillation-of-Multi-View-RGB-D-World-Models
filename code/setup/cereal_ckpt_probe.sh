#!/bin/bash
# cereal 과선명 원인 규명 1단계 — 이미 있는 step2000 체크포인트를 평가해 "학습이 길수록 과선명"인지 확인
# 재학습(1h, 디스크 12G 필요) 전에 20분·디스크 0으로 방향을 가린다.
#   가설(a) 체크포인트 문제  : 2000이 1600보다 더 과선명하면 지지 → 재학습해서 1200/1400 확인할 가치 있음
#   가설(b) 기준/태스크 문제 : 2000 ≈ 1600 이면 스텝 수와 무관 → std비 기준의 한계로 보고
# 사용: setsid nohup bash ~/cereal_ckpt_probe.sh > ~/cereal_probe.log 2>&1 < /dev/null &
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
export GEO4D_TEACHER_DIR=/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/cereal_box
export GEO4D_TASK_YAML=/home/sun4208/4dgen/config/task/inference_cereal_box.yaml
CK=~/Geo4D/dmd_cereal_box/dmd_gen.pt          # step 2000 (파이프라인이 마지막에 저장한 것)
[ -f "$CK" ] || { echo "체크포인트 없음: $CK"; exit 1; }

echo "[$(date +%H:%M)] GPU 대기"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 3000 ]; do sleep 300; done
sleep 20
cd ~/4dgen
echo "[$(date +%H:%M)] step2000 평가 시작 (T25/S3b/S5b, 20샘플, ~25분)"
python notebooks/bench_precise_6a.py --student_ckpt $CK --configs T25 S3b S5b \
  --n_batches 20 --n_div 3 --fast --tag _cereal_s2000 2>&1 | tee ~/cereal_s2000.log
echo "PROBE_RC=${PIPESTATUS[0]}"

echo "=== 비교: step1600(기존) vs step2000(신규) ==="
echo "--- step1600 ---"; head -9 ~/Geo4D/bench_out/precise_6a_cereal_box.txt | tail -6
echo "--- step2000 ---"; head -9 ~/Geo4D/bench_out/precise_6a_cereal_s2000.txt | tail -5
echo CEREAL_PROBE_DONE
