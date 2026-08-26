#!/bin/bash
# cereal 체크포인트 스윕 — 400/800/1200/1600/2000 을 순차 학습·평가 (GPT 요청 1순위)
# 디스크 제약(체크포인트 3.1G, 여유 ~13G) 때문에 "구간 학습 → 그 자리에서 평가 → 다음 구간 재개" 방식.
#   사본을 만들지 않고 작업 파일 dmd_gen.pt 를 직접 평가하므로 피크 사용량은 6.2G(gen+fake) 로 유지된다.
# 각 지점에서 S3b·S5b 둘 다, 같은 검증 셋(20샘플)에서:
#   전체 AbsRel/LPIPS/선명도/다양성(bench_precise_6a) + 영역별 AbsRel·LPIPS·선명도(bench_region_perceptual)
#   + 그리퍼·물체 중심오차(bench_policy_proxy).  전부 _raw.json 으로 샘플별 값 저장 → CSV 변환 가능.
# 주의(정직한 기록): 구간마다 --resume 이므로 optimizer 모멘트가 재시작된다. 연속 학습과 완전히 동일하지 않다.
# 사용: setsid nohup bash ~/cereal_ckpt_sweep.sh > ~/cereal_sweep.log 2>&1 < /dev/null &
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
TASK=cereal_box
export GEO4D_TEACHER_DIR=/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/$TASK
export GEO4D_TASK_YAML=/home/sun4208/4dgen/config/task/inference_$TASK.yaml
export GEO4D_OBJECT_ID=39
PAIRS=~/Geo4D/ode_pairs_$TASK
INIT=~/Geo4D/ode_init_$TASK.pt
OUT=~/Geo4D/dmd_sweep_$TASK
mkdir -p $OUT
cd ~/4dgen

wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 3000 ]; do sleep 300; done; sleep 20; }
free_g() { df --output=avail -BG ~ | tail -1 | tr -dc 0-9; }

[ -d "$PAIRS" ] || { echo "ODE 쌍 없음: $PAIRS"; exit 1; }
echo "[$(date +%H:%M)] 시작. 디스크 여유 $(free_g)G"

# 파이프라인이 공간 회수로 지운 옛 step2000 체크포인트를 먼저 정리 (probe 가 이미 평가함)
if [ -f ~/Geo4D/dmd_cereal_box/dmd_gen.pt ] && grep -q "CEREAL_PROBE_DONE" ~/cereal_probe.log 2>/dev/null; then
  echo "[정리] probe 평가 완료된 step2000 체크포인트 삭제 (3.1G 회수)"
  rm -f ~/Geo4D/dmd_cereal_box/dmd_gen.pt
fi

# 초기화 체크포인트도 회수됐으므로 ODE 쌍에서 재생성 (~80분)
if [ ! -f "$INIT" ]; then
  echo "=== [초기화 재생성] 회귀 1200 step (~80분) ==="
  wait_gpu
  python notebooks/geo4d_ode_init_train_v2.py --pairs_dir $PAIRS --out_ckpt $INIT 2>&1 | tail -4
  [ -f "$INIT" ] || { echo "초기화 실패 — 중단"; exit 1; }
fi

PREV=0
for STEP in 400 800 1200 1600 2000; do
  echo "=== [학습] → step $STEP (이전 $PREV) ==="
  wait_gpu
  if [ "$PREV" -eq 0 ]; then
    RES=""
  else
    RES="--resume"
  fi
  python notebooks/geo4d_dmd_train.py --pairs_dir $PAIRS --init_ckpt $INIT --out_dir $OUT \
      --max_steps $STEP $RES --save_every 200 --keep_steps "" 2>&1 | tail -5
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "학습 실패(step $STEP, rc=$rc) — 중단"; exit 1; }
  PREV=$STEP
  if [ "$STEP" -eq 400 ]; then rm -f "$INIT"; echo "[정리] 초기화 ckpt 삭제(이후 구간은 --resume 만 사용, 3.1G 회수)"; fi

  CK=$OUT/dmd_gen.pt
  echo "=== [평가] step $STEP (S3b/S5b, 20샘플) ==="
  wait_gpu
  python notebooks/bench_precise_6a.py --student_ckpt $CK --configs S3b S5b \
      --n_batches 20 --n_div 3 --fast --tag _sweep_s$STEP 2>&1 | tail -6
  wait_gpu
  python notebooks/bench_region_perceptual.py --student_ckpt $CK --configs S3b S5b \
      --n_batches 20 --fast --tag _sweep_s$STEP 2>&1 | tail -4
  wait_gpu
  python notebooks/bench_policy_proxy.py --student_ckpt $CK --configs S3b S5b \
      --n_batches 20 --fast --tag _sweep_s$STEP 2>&1 | tail -4
  echo "[$(date +%H:%M)] step $STEP 완료. 디스크 여유 $(free_g)G"
done

rm -f $OUT/dmd_fake.pt
echo "=== 요약: 체크포인트별 S3b ==="
for STEP in 400 800 1200 1600 2000; do
  f=~/Geo4D/bench_out/precise_6a_sweep_s$STEP.txt
  [ -f "$f" ] && echo "--- step $STEP ---" && sed -n '4,7p' "$f"
done
df -h ~ | tail -1
echo CEREAL_SWEEP_DONE
