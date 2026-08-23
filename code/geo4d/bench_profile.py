"""Geo4D 추론 병목 원인 분해: 컴포넌트별 시간 프로파일링"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import time
import functools
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
ckpt_path = f"{output_dir}/4dgen.ckpt"

print("[1/4] 모델 로드 중")
cfg = OmegaConf.load(f"{output_dir}/config.yaml")
for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and "desc" in cfg[key]:
        cfg[key] = cfg[key]["value"]
cls = hydra.utils.get_class(cfg._target_)
cfg.logging.mode = "offline"
cfg.model.params.ckpt_path = ckpt_path
cfg.training.seed = 42
cfg.training.output_dir = "/home/sun4208/Geo4D/bench_out"
workspace = cls(cfg)
device = "cuda"
model = workspace.lightning_module_wrapper.to(device)
model.eval()

print("[2/4] 데이터 준비 (zarr 캐시 재사용)")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
batch = next(iter(loader))
n = batch["pointmap"].shape[1]
input_batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
input_batch["num_video_frames"] = n

# ── 컴포넌트별 타이머 설치 (monkeypatch) ─────────────────────────
stats = {}
def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize(); t0 = time.time()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        s = stats.setdefault(name, [0.0, 0])
        s[0] += time.time() - t0; s[1] += 1
        return out
    return wrapper

model.model.forward = timed("UNet forward (denoising)", model.model.forward)
model.conditioner.forward = timed("Conditioner (CLIP+VAE 인코딩)", model.conditioner.forward)
model.decode_first_stage = timed("VAE 디코딩", model.decode_first_stage)
model.encode_first_stage = timed("VAE 인코딩 (GT/조건)", model.encode_first_stage)

print("[3/4] 워밍업 1회")
with torch.no_grad():
    model.log_images(input_batch)
stats.clear()

print("[4/4] 본 측정 1회")
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    model.log_images(input_batch)
torch.cuda.synchronize()
total = time.time() - t0

print()
print(f"=== 추론 병목 분해 (10프레임 1회 생성, 총 {total:.2f}초) ===")
accounted = 0.0
for name, (t, c) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
    print(f"  {name:32s} {t:7.2f}초  ({t/total*100:5.1f}%)  호출 {c}회  (호출당 {t/max(c,1)*1000:.0f}ms)")
    accounted += t
print(f"  {chr(39)}기타{chr(39)} (샘플러 오버헤드 등)      {total-accounted:7.2f}초  ({(total-accounted)/total*100:5.1f}%)")
print()
unet_t, unet_c = stats.get("UNet forward (denoising)", [0, 1])
per_call = unet_t / max(unet_c, 1)
others = total - unet_t
print(f"※ 증류로 denoising을 4스텝으로 줄이면(호출 {unet_c}→{max(4, unet_c*4//25)}회 가정):")
print(f"   예상 시간 ≈ {per_call * max(4, unet_c*4//25) + others:.1f}초  (현재 {total:.1f}초)")
