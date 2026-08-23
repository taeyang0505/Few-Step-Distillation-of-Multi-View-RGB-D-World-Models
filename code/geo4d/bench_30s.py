"""Geo4D 추론 병목 측정: 10프레임 생성 시간 (논문 주장: RTX 4090에서 ~30초)"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import time
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
ckpt_path = f"{output_dir}/4dgen.ckpt"

print("[1/4] config 로드 및 모델 초기화 (22GB ckpt — 수 분 소요)")
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
print("[2/4] 모델 로드 완료")

print("[3/4] 데이터셋 준비 (첫 실행은 zarr 캐시 생성으로 몇 분 걸림)")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
assert isinstance(dataset, SpartanVideoMultiViewDataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
batch = next(iter(loader))
n = batch["pointmap"].shape[1]
input_batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
input_batch["num_video_frames"] = n
print(f"[4/4] 준비 완료 — 프레임 수: {n}, 측정 시작 (1회차는 워밍업)")

for i in range(3):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        outputs = model.log_images(input_batch)
    torch.cuda.synchronize()
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 2**30
    tag = "워밍업" if i == 0 else "측정  "
    print(f"[{tag} {i+1}/3] {n}프레임 생성: {dt:.2f}초  |  피크 VRAM: {vram:.1f}GB  (논문: 4090에서 ~30초)")
