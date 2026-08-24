#!/bin/bash
# D1 파이프라인: 태스크 하나의 전 과정 무인 실행 (ODE 쌍 → 회귀 초기화 → DMD → 다이얼 평가 → 대리 지표)
# 사용: ~/d1_pipeline.sh cereal_box   (또는 spatula). GPU ~5.5시간. OOM 시 자동 대기·재개.
# 대리 지표는 GEO4D_OBJECT_ID(조작 물체 label id)가 설정된 경우에만 실행 (태스크별 id는 다운로드 후 별도 확인).
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
TASK=${1:?사용법: d1_pipeline.sh cereal_box|spatula}
export GEO4D_TEACHER_DIR=/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/$TASK
export GEO4D_TASK_YAML=/home/sun4208/4dgen/config/task/inference_$TASK.yaml
PAIRS=~/Geo4D/ode_pairs_$TASK
INIT=~/Geo4D/ode_init_$TASK.pt
OUT=~/Geo4D/dmd_$TASK
cd ~/4dgen
[ -d "$GEO4D_TEACHER_DIR" ] || { echo "teacher 없음: $GEO4D_TEACHER_DIR — 먼저 d1_download.sh $TASK"; exit 1; }
[ -f "$GEO4D_TASK_YAML" ] || { echo "task yaml 없음"; exit 1; }
FREE=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 12 ] || { echo "디스크 부족(${FREE}G < 12G)"; exit 1; }
gpu_free() { local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 2000 ]; }
run_retry() {  # 크래시(OOM 등) 시 GPU 빌 때까지 대기 후 재시도. ODE 쌍은 파일 단위 재개, DMD는 --resume 재개.
  for attempt in 1 2 3 4 5 6; do
    "$@" && return 0
    echo "STAGE_CRASHED attempt=$attempt: $1 ... $(date '+%H:%M') — GPU 대기 후 재시도"
    until gpu_free; do sleep 60; done; sleep 30
  done
  return 1
}
echo "[1/5] ODE 쌍 생성 (~2h)"
run_retry env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python notebooks/geo4d_ode_gen_v2.py --out_dir $PAIRS --max_pairs 1000 || { echo D1_${TASK}_FAILED_ODE; exit 1; }
echo D1_${TASK}_ODE_DONE
echo "[2/5] 회귀 초기화 1200 step (~1.5h)"
if [ ! -f $INIT ]; then
  run_retry env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python notebooks/geo4d_ode_init_train_v2.py --pairs_dir $PAIRS --out_ckpt $INIT || { echo D1_${TASK}_FAILED_INIT; exit 1; }
fi
echo D1_${TASK}_INIT_DONE
echo "[3/5] DMD 2000 step (~1h)"
run_retry env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python notebooks/geo4d_dmd_train.py --pairs_dir $PAIRS --init_ckpt $INIT --out_dir $OUT --max_steps 2000 --keep_steps 1600 --resume || { echo D1_${TASK}_FAILED_DMD; exit 1; }
echo D1_${TASK}_DMD_DONE
rm -f $OUT/dmd_fake.pt $INIT    # 공간 회수 (기록: init은 1.5h에 재생성 가능)
echo "[4/5] 다이얼 평가 (T25/S3b/S5b/H4b, 20샘플, ~40분)"
run_retry python notebooks/bench_precise_6a.py --student_ckpt $OUT/dmd_gen_step1600.pt --configs T25 S3b S5b H4b --n_batches 20 --n_div 3 --fast --tag _${TASK} || { echo D1_${TASK}_FAILED_EVAL; exit 1; }
echo D1_${TASK}_EVAL_DONE
if [ -n "${GEO4D_OBJECT_ID:-}" ]; then
  echo "[5/5] 대리 지표 (~15분, object id=$GEO4D_OBJECT_ID)"
  run_retry python notebooks/bench_policy_proxy.py --student_ckpt $OUT/dmd_gen_step1600.pt --configs T25 S3b --n_batches 20 --fast --tag _${TASK} || echo "proxy 실패(치명적 아님)"
else
  echo "[5/5] 대리 지표 생략 — GEO4D_OBJECT_ID 미설정 (물체 id 확인 후 별도 실행)"
fi
df -h ~ | tail -1
echo D1_${TASK}_ALL_DONE
