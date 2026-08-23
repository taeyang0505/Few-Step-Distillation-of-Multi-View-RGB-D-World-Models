#!/usr/bin/env bash
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/dreamzero-work && source .venv/bin/activate
say(){ printf "\n[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

say "1/3 최소 의존성 (로봇 시뮬 스택 제외 — 추론엔 불필요)"
uv pip install --quiet accelerate diffusers einops hydra-core omegaconf peft==0.5.0 \
  safetensors transformers tqdm huggingface_hub av imageio numpy scipy 2>&1 | tail -4
python -c "import torch,diffusers,transformers,peft;print(\"  torch\",torch.__version__,\"diffusers\",diffusers.__version__,\"peft\",peft.__version__)"

say "2/3 SO-101 LoRA (217MB)"
python - <<PY
from huggingface_hub import snapshot_download
p=snapshot_download("Vizuara/dreamzero-so101-lora", local_dir="checkpoints/dreamzero-so101-lora",
                    max_workers=4)
print("  ->", p)
PY

say "3/3 Wan2.1-I2V-14B-480P 베이스 (~79GB, 오래 걸림)"
python - <<PY
from huggingface_hub import snapshot_download
p=snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="checkpoints/Wan2.1-I2V-14B-480P",
                    max_workers=8)
print("  ->", p)
PY

say "완료"
du -sh ~/dreamzero-work/checkpoints/* 2>/dev/null
df -h / | tail -1
echo "===EXIT=0"
