#!/bin/bash
# D1: Task 1(cereal_box)·Task 2(spatula)의 teacher ckpt + 추론 에피소드 4개 다운로드 + 태스크별 config 생성 + zarr 캐시 빌드
# 사용: ~/d1_download.sh cereal_box   또는   ~/d1_download.sh spatula
# GPU 불필요 (네트워크 + CPU). 태스크당 디스크 ~31G (ckpt 22.7 + 에피소드 7.6 + zarr 0.6)
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
TASK=$1
case "$TASK" in
  cereal_box) TDIR=BimanualStoreCerealBoxUnderShelf; SESSION=2025-03-13T04-19-57+00-00 ;;
  spatula)    TDIR=BimanualPutSpatulaOnTableFromUtensilCrock; SESSION=2025-03-29T06-34-17+00-00 ;;
  *) echo "사용법: $0 cereal_box|spatula"; exit 1 ;;
esac
echo "[1/4] teacher 체크포인트 (22.7G)"
python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("Zeyi/4dgen-ckpts", local_dir="/home/sun4208/Geo4D/checkpoints",
                  allow_patterns=["checkpoints/outputs/$TASK/*"], max_workers=8)
print("ckpt ok")
PYEOF
echo "[2/4] 추론 에피소드 0-3 (~7.6G)"
python - <<PYEOF
from huggingface_hub import snapshot_download
pats=[f"$TDIR/$SESSION/diffusion_spartan/episode_{i}/**" for i in range(4)]
snapshot_download("Zeyi/4dgen-dataset", repo_type="dataset", local_dir="/home/sun4208/Geo4D/Data",
                  allow_patterns=pats, max_workers=8)
print("data ok")
PYEOF
echo "[3/4] 태스크 config 생성: ~/4dgen/config/task/inference_$TASK.yaml"
cd ~/4dgen
python - <<PYEOF
src=open("config/task/inference.yaml").read()
src=src.replace("/home/sun4208/Geo4D/Data/BimanualPlaceAppleFromBowlIntoBin/2025-03-29T08-08-56+00-00/diffusion_spartan/episode_*",
                "/home/sun4208/Geo4D/Data/$TDIR/$SESSION/diffusion_spartan/episode_*")
src=src.replace("/home/sun4208/Geo4D/Data/replay_buffer_infer.zarr",
                "/home/sun4208/Geo4D/Data/replay_buffer_infer_$TASK.zarr")
open("config/task/inference_$TASK.yaml","w").write(src)
print("config ok")
PYEOF
echo "[4/4] zarr 캐시 빌드 (CPU, ~10-20분; 없으면 자동 생성)"
python - <<PYEOF
import sys; sys.path.insert(0,"."); sys.path.insert(0,"notebooks")
from common import transformers_pre_import_mods
import hydra
from omegaconf import OmegaConf
output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"   # horizon 등 보간용 (apple config 재사용)
cfg = OmegaConf.load(f"{output_dir}/config.yaml")
for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and "desc" in cfg[key]:
        cfg[key] = cfg[key]["value"]
cfg.task = OmegaConf.load("config/task/inference_$TASK.yaml")
ds = hydra.utils.instantiate(cfg.task.dataset)
print("DATASET_LEN", len(ds))
PYEOF
du -sh ~/Geo4D/checkpoints/checkpoints/outputs/$TASK ~/Geo4D/Data/$TDIR ~/Geo4D/Data/replay_buffer_infer_$TASK.zarr 2>/dev/null
df -h ~ | tail -1
echo D1_DL_${TASK}_DONE
