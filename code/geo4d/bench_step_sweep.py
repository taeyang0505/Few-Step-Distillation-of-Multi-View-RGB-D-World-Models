"""Step 1 예비실험: 샘플링 스텝 수(25/16/8/4)에 따른 기하·지각 지표 변화 측정
가설: 스텝을 줄이면 지각 지표(PSNR)보다 기하 지표(AbsRel, delta1, MSE_P)가 먼저 무너진다"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import time
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
ckpt_path = f"{output_dir}/4dgen.ckpt"


# ---- eval.py의 지표 함수 verbatim 복제 ----
def unnormalize(input, min, max):
    return torch.clamp(((input + 1.) / 2.) * (max - min) + min, min, max)


def compute_MSE_P(outputs, view):
    pred = unnormalize(outputs["video_dict"][f"sampled_video_{view}"][:, :3], -1, 2).cpu().detach()
    gt = unnormalize(outputs["video_dict"][f"gt_video_{view}"][:, :3], -1, 2).cpu().detach()
    return torch.mean((gt - pred).pow(2))


def compute_psnr(outputs, view):
    pred = unnormalize(outputs["video_dict"][f"sampled_video_{view}"][:, 3:], 0, 1).cpu().detach()
    gt = unnormalize(outputs["video_dict"][f"gt_video_{view}"][:, 3:], 0, 1).cpu().detach()
    mse = torch.mean((pred - gt).pow(2))
    return 10 * torch.log10(1. / mse)


def compute_abs_rel(outputs, view):
    d1 = unnormalize(outputs["video_dict"][f"sampled_video_{view}"][:, 2], -1, 2).cpu().detach()
    d2 = unnormalize(outputs["video_dict"][f"gt_video_{view}"][:, 2], -1, 2).cpu().detach()
    m = (d1 > 0) & (d2 > 0)
    d1, d2 = d1[m], d2[m]
    return torch.mean(torch.abs(d1 - d2) / d2)


def compute_delta_1(outputs, view, threshold=1.25):
    d1 = unnormalize(outputs["video_dict"][f"sampled_video_{view}"][:, 2], -1, 2).cpu().detach()
    d2 = unnormalize(outputs["video_dict"][f"gt_video_{view}"][:, 2], -1, 2).cpu().detach()
    m = (d1 > 0) & (d2 > 0)
    d1, d2 = d1[m], d2[m]
    if len(d1) == 0:
        return torch.tensor(0.0)
    delta = torch.maximum(d1 / d2, d2 / d1)
    return torch.sum(delta < threshold) / len(delta)


# ---- 모델·데이터 로드 ----
print("[1/3] 모델 로드 (수 분 소요)")
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
model = workspace.lightning_module_wrapper.to("cuda")
model.eval()
sampler = model.sampler
print("샘플러:", type(sampler).__name__, "| 기본 num_steps:", sampler.num_steps)

print("[2/3] 데이터 준비 (zarr 캐시 재사용, 배치 3개 고정)")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
batches = []
for i, b in enumerate(loader):
    if i >= 3:
        break
    n = b["pointmap"].shape[1]
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True))
    ib["num_video_frames"] = n
    batches.append(ib)
print(f"배치 {len(batches)}개 준비 완료")

with torch.no_grad():
    model.log_images(batches[0])  # 워밍업

print("[3/3] 스텝 수 스윕 시작")
rows = []
for steps in [25, 16, 8, 4]:
    sampler.num_steps = steps
    metrics = {k: [] for k in ["MSE_P", "PSNR", "AbsRel", "delta1"]}
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for ib in batches:
            out = model.log_images(ib)
            for view in ["left", "right"]:
                metrics["MSE_P"].append(compute_MSE_P(out, view).item())
                metrics["PSNR"].append(compute_psnr(out, view).item())
                metrics["AbsRel"].append(compute_abs_rel(out, view).item())
                metrics["delta1"].append(compute_delta_1(out, view).item())
    torch.cuda.synchronize()
    dt = (time.time() - t0) / len(batches)
    row = (steps, dt,
           np.mean(metrics["PSNR"]), np.mean(metrics["MSE_P"]),
           np.mean(metrics["AbsRel"]), np.mean(metrics["delta1"]))
    rows.append(row)
    print(f"[steps={steps:2d}] {dt:5.1f}초/배치 | PSNR {row[2]:.2f} | MSE_P {row[3]:.4f} | AbsRel {row[4]:.4f} | delta1 {row[5]:.4f}")

header = "steps | time(s) |  PSNR^ | MSE_P v | AbsRel v | delta1 ^"
report = ["=== Step 1 예비실험: 스텝 수 vs 기하·지각 지표 (배치 3개 x 2뷰 평균) ===", header]
for r in rows:
    report.append(f"{r[0]:>5} | {r[1]:>7.1f} | {r[2]:>6.2f} | {r[3]:>7.4f} | {r[4]:>8.4f} | {r[5]:>8.4f}")
base = rows[0]
report.append("--- 25스텝 대비 상대 변화(%) ---")
for r in rows[1:]:
    report.append(
        f"{r[0]:>5} | 시간 {100*r[1]/base[1]-100:+.0f}% | PSNR {100*r[2]/base[2]-100:+.1f}% | "
        f"MSE_P {100*r[3]/base[3]-100:+.1f}% | AbsRel {100*r[4]/base[4]-100:+.1f}% | delta1 {100*r[5]/base[5]-100:+.1f}%"
    )
text = "\n".join(report)
print()
print(text)
with open("/home/sun4208/Geo4D/bench_out/step_sweep_results.txt", "w") as f:
    f.write(text + "\n")
print()
print("결과 저장: ~/Geo4D/bench_out/step_sweep_results.txt")
