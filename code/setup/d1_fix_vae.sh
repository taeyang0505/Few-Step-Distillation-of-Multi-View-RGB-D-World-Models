#!/bin/bash
# D1 수정: 태스크별 VAE 체크포인트 다운로드 + config.yaml의 원저자 절대경로를 로컬 경로로 패치
# 사용: ~/d1_fix_vae.sh cereal_box   (또는 spatula).  GPU 불필요(네트워크+CPU), ~5-10분, 디스크 ~2.2G
set -e
TASK=${1:?"사용법: $0 cereal_box|spatula"}
CKPT_ROOT=/home/sun4208/Geo4D/checkpoints
source ~/miniconda3/etc/profile.d/conda.sh; conda activate video_policy

echo "[1/3] VAE 체크포인트 다운로드 (checkpoints/VAE/$TASK)"
python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("Zeyi/4dgen-ckpts", local_dir="$CKPT_ROOT",
                  allow_patterns=["checkpoints/VAE/$TASK/*"], max_workers=8)
print("vae ok")
PYEOF
ls -la $CKPT_ROOT/checkpoints/VAE/$TASK/

echo "[2/3] config.yaml VAE 경로 패치"
python - <<PYEOF
import re, pathlib, shutil
d = pathlib.Path("$CKPT_ROOT/checkpoints/VAE/$TASK")
files = {p.stem: p for p in d.glob("*.ckpt")}
pm = files.get("pointmap")
col = files.get("color") or files.get("rgb")          # 태스크마다 color.ckpt / rgb.ckpt 로 이름이 다름
assert pm and col, f"VAE 파일 부족: {list(files)}"
cfg = pathlib.Path("$CKPT_ROOT/checkpoints/outputs/$TASK/config.yaml")
if not cfg.with_suffix(".yaml.orig").exists():
    shutil.copy(cfg, cfg.with_suffix(".yaml.orig"))   # 원본 백업(1회)
s = cfg.read_text()
# 원저자 절대경로 ckpt_path 를 로컬로: 등장 순서와 무관하게 경로 문자열 단위로 매핑
paths = sorted(set(re.findall(r"ckpt_path:\s*(/store/real/\S*autoencoder\S*\.ckpt)", s)))
assert len(paths) == 2, f"예상과 다른 VAE 경로 수: {paths}"
first_at = {p: s.index(p) for p in paths}
pm_src, col_src = sorted(paths, key=lambda p: first_at[p])   # 먼저 등장하는 쪽이 pointmap (apple config 와 동일 순서)
n1 = s.count(pm_src); n2 = s.count(col_src)
s = s.replace(pm_src, str(pm)).replace(col_src, str(col))
cfg.write_text(s)
print(f"패치 완료: pointmap {n1}곳 -> {pm.name}, color {n2}곳 -> {col.name}")
PYEOF

echo "[3/3] 검증 — 남은 /store/real ckpt_path (0이어야 정상; 데이터 glob 경로는 무시)"
grep -c "ckpt_path: /store/real" $CKPT_ROOT/checkpoints/outputs/$TASK/config.yaml || true
grep -n "ckpt_path: /home/sun4208/Geo4D/checkpoints/checkpoints/VAE" $CKPT_ROOT/checkpoints/outputs/$TASK/config.yaml
echo "D1_FIX_VAE_${TASK}_DONE"
