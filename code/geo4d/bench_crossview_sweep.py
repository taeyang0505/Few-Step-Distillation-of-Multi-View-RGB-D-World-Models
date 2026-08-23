"""Step 1-후속: 스텝 수(25/16/8/4)에 따른 cross-view 기하 정합 측정
지표: 두 뷰 예측 포인트맵(같은 참조 좌표계) 사이의 대칭 chamfer 거리.
GT 점군 간 chamfer를 기준선으로 사용 — ratio = pred / GT 가 1에서 멀어질수록 뷰 간 기하 불일치."""
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
N_SUB = 4096  # 프레임당 점군 서브샘플 수


def unnormalize(input, min, max):
    return torch.clamp(((input + 1.) / 2.) * (max - min) + min, min, max)


def get_cloud(video, t, seed):
    """video: (T, 3, H, W) 포인트맵 → t프레임의 유효 점군 (N,3), 서브샘플."""
    pm = video[t]                       # (3, H, W)
    pts = pm.reshape(3, -1).T           # (HW, 3)
    valid = pts[:, 2] > 0               # eval.py와 동일한 유효 기준 (z>0)
    pts = pts[valid]
    if pts.shape[0] > N_SUB:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(pts.shape[0], generator=g)[:N_SUB]
        pts = pts[idx.to(pts.device)]
    return pts


def chamfer(a, b):
    """대칭 chamfer 거리 (양방향 최근접 평균). a:(N,3) b:(M,3), GPU."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)  # (N, M)
    return 0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean())


def crossview_chamfer(outputs, key_l, key_r):
    """두 뷰 포인트맵 비디오의 프레임별 chamfer 평균."""
    vid_l = unnormalize(outputs["video_dict"][key_l][:, :3], -1, 2)
    vid_r = unnormalize(outputs["video_dict"][key_r][:, :3], -1, 2)
    T = vid_l.shape[0]
    vals = []
    for t in range(T):
        cl = get_cloud(vid_l, t, seed=1000 + t)
        cr = get_cloud(vid_r, t, seed=2000 + t)
        vals.append(chamfer(cl, cr).item())
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

print("[2/3] 데이터 준비 (배치 3개 고정)")
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

print("[3/3] 스텝 수 스윕: cross-view chamfer 측정")
gt_cv = None  # GT 기준선은 스텝 수와 무관 — 첫 스윕에서 1회 계산
rows = []
for steps in [25, 16, 8, 4]:
    sampler.num_steps = steps
    pred_vals, gt_vals = [], []
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for ib in batches:
            out = model.log_images(ib)
            pred_vals.append(crossview_chamfer(out, "sampled_video_left", "sampled_video_right"))
            if gt_cv is None:
                gt_vals.append(crossview_chamfer(out, "gt_video_left", "gt_video_right"))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / len(batches)
    if gt_cv is None:
        gt_cv = float(np.mean(gt_vals))
    pred_cv = float(np.mean(pred_vals))
    rows.append((steps, dt, pred_cv, gt_cv, pred_cv / gt_cv))
    print(f"[steps={steps:2d}] {dt:5.1f}초/배치 | CV-Chamfer(pred) {pred_cv:.4f} | CV-Chamfer(GT) {gt_cv:.4f} | ratio {pred_cv/gt_cv:.3f}")

report = ["=== 스텝 수 vs cross-view 기하 정합 (배치 3개, 프레임별 chamfer 평균, 단위: 좌표계 거리) ===",
          "steps | time(s) | CV-Chamfer(pred) v | CV-Chamfer(GT) | ratio(pred/GT) v"]
for r in rows:
    report.append(f"{r[0]:>5} | {r[1]:>7.1f} | {r[2]:>18.4f} | {r[3]:>14.4f} | {r[4]:>16.3f}")
base = rows[0]
report.append("--- 25스텝 대비 상대 변화(%) ---")
for r in rows[1:]:
    report.append(f"{r[0]:>5} | CV-Chamfer(pred) {100*r[2]/base[2]-100:+.1f}% | ratio {100*r[4]/base[4]-100:+.1f}%")
text = "\n".join(report)
print()
print(text)
with open("/home/sun4208/Geo4D/bench_out/crossview_sweep_results.txt", "w") as f:
    f.write(text + "\n")
print()
print("결과 저장: ~/Geo4D/bench_out/crossview_sweep_results.txt")
