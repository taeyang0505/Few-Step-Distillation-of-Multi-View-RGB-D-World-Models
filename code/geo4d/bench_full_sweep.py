"""Step 1 확장: 스텝 수(25/16/8/4/2/1) x 배치 10개 — 픽셀 + cross-view 지표 통합 스윕
목적: (a) 진짜 품질 붕괴 지점 확인 (1~2스텝), (b) 배치 확대로 노이즈 여부 확정"""
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
N_BATCHES = 10
STEPS_LIST = [25, 16, 8, 4, 2, 1]
N_SUB = 4096


def unnormalize(input, min, max):
    return torch.clamp(((input + 1.) / 2.) * (max - min) + min, min, max)


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


def get_cloud(video, t, seed):
    pm = video[t]
    pts = pm.reshape(3, -1).T
    valid = pts[:, 2] > 0
    pts = pts[valid]
    if pts.shape[0] > N_SUB:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(pts.shape[0], generator=g)[:N_SUB]
        pts = pts[idx.to(pts.device)]
    return pts


def chamfer(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return torch.tensor(float("nan"))
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return 0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean())


def crossview_chamfer(outputs, key_l, key_r):
    vid_l = unnormalize(outputs["video_dict"][key_l][:, :3], -1, 2)
    vid_r = unnormalize(outputs["video_dict"][key_r][:, :3], -1, 2)
    vals = []
    for t in range(vid_l.shape[0]):
        vals.append(chamfer(get_cloud(vid_l, t, 1000 + t), get_cloud(vid_r, t, 2000 + t)).item())
    return float(np.mean(vals))


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

print(f"[2/3] 데이터 준비 (배치 {N_BATCHES}개 고정)")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
batches = []
for i, b in enumerate(loader):
    if i >= N_BATCHES:
        break
    n = b["pointmap"].shape[1]
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True))
    ib["num_video_frames"] = n
    batches.append(ib)
print(f"배치 {len(batches)}개 준비 완료 (전체 데이터셋 샘플 수: {len(dataset)})")

with torch.no_grad():
    model.log_images(batches[0])  # 워밍업

print(f"[3/3] 스윕 시작: steps={STEPS_LIST}, 예상 총 소요 ~{int(sum(4.6+0.84*s for s in STEPS_LIST)*len(batches)/60)}분")
gt_cv = None
rows = []
for steps in STEPS_LIST:
    sampler.num_steps = steps
    m = {k: [] for k in ["PSNR", "AbsRel", "CV"]}
    gt_vals = []
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for ib in batches:
            out = model.log_images(ib)
            for view in ["left", "right"]:
                m["PSNR"].append(compute_psnr(out, view).item())
                m["AbsRel"].append(compute_abs_rel(out, view).item())
            m["CV"].append(crossview_chamfer(out, "sampled_video_left", "sampled_video_right"))
            if gt_cv is None:
                gt_vals.append(crossview_chamfer(out, "gt_video_left", "gt_video_right"))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / len(batches)
    if gt_cv is None:
        gt_cv = float(np.mean(gt_vals))
    row = (steps, dt, np.mean(m["PSNR"]), np.std(m["PSNR"]),
           np.mean(m["AbsRel"]), np.std(m["AbsRel"]),
           np.mean(m["CV"]), np.std(m["CV"]), np.mean(m["CV"]) / gt_cv)
    rows.append(row)
    print(f"[steps={steps:2d}] {dt:5.1f}초/배치 | PSNR {row[2]:.2f}±{row[3]:.2f} | AbsRel {row[4]:.4f}±{row[5]:.4f} | CV-Chamfer {row[6]:.4f}±{row[7]:.4f} | ratio {row[8]:.3f}")

report = [f"=== 확장 스윕: 스텝 수 vs 품질 (배치 {len(batches)}개, 평균±표준편차, GT CV-Chamfer={gt_cv:.4f}) ===",
          "steps | time(s) | PSNR^ (std) | AbsRel v (std) | CV-Chamfer v (std) | ratio v"]
for r in rows:
    report.append(f"{r[0]:>5} | {r[1]:>7.1f} | {r[2]:>5.2f} ({r[3]:.2f}) | {r[4]:.4f} ({r[5]:.4f}) | {r[6]:.4f} ({r[7]:.4f}) | {r[8]:>6.3f}")
base = rows[0]
report.append("--- 25스텝 대비 상대 변화(%) ---")
for r in rows[1:]:
    report.append(f"{r[0]:>5} | 시간 {100*r[1]/base[1]-100:+.0f}% | PSNR {100*r[2]/base[2]-100:+.1f}% | AbsRel {100*r[4]/base[4]-100:+.1f}% | CV-Chamfer {100*r[6]/base[6]-100:+.1f}%")
text = "\n".join(report)
print()
print(text)
with open("/home/sun4208/Geo4D/bench_out/full_sweep_results.txt", "w") as f:
    f.write(text + "\n")
print()
print("결과 저장: ~/Geo4D/bench_out/full_sweep_results.txt")
