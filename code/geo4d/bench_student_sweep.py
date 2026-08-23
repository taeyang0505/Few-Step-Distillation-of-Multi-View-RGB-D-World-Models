"""Step 6-2 평가: ODE 초기화 student의 1/2/4스텝 품질을 Step 1 기준선과 동일 지표로 측정
기준선(teacher, 학습 없는 축소, 배치 10):
  4스텝 PSNR 20.35/AbsRel 0.1255/CV 0.1796 | 2스텝 20.66/0.1318/0.1919 | 1스텝 20.64/0.1321/0.1943"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse
import time
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps

parser = argparse.ArgumentParser()
parser.add_argument("--student_ckpt", type=str, default="/home/sun4208/Geo4D/ode_init_geo4d.pt")
parser.add_argument("--steps", type=int, nargs="+", default=[4, 2, 1])
parser.add_argument("--n_batches", type=int, default=10)
parser.add_argument("--sampler", choices=["euler", "renoise"], default="euler",
                    help="renoise = DMD student 추론 방식(x0 예측→재노이징, geo4d_fewstep.RenoiseSampler)")
parser.add_argument("--tag", default="", help="결과 파일 접미사")
args_cli = parser.parse_args()

N_SUB = 4096
BASELINE = {4: (20.35, 0.1255, 0.1796), 2: (20.66, 0.1318, 0.1919), 1: (20.64, 0.1321, 0.1943)}


def unnormalize(x, mn, mx):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)


def compute_psnr(o, v):
    p = unnormalize(o["video_dict"][f"sampled_video_{v}"][:, 3:], 0, 1).cpu()
    g = unnormalize(o["video_dict"][f"gt_video_{v}"][:, 3:], 0, 1).cpu()
    return (10 * torch.log10(1. / torch.mean((p - g).pow(2)))).item()


def compute_abs_rel(o, v):
    d1 = unnormalize(o["video_dict"][f"sampled_video_{v}"][:, 2], -1, 2).cpu()
    d2 = unnormalize(o["video_dict"][f"gt_video_{v}"][:, 2], -1, 2).cpu()
    m = (d1 > 0) & (d2 > 0)
    return torch.mean(torch.abs(d1[m] - d2[m]) / d2[m]).item()


def get_cloud(video, t, seed):
    pts = video[t].reshape(3, -1).T
    pts = pts[pts[:, 2] > 0]
    if pts.shape[0] > N_SUB:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(pts.shape[0], generator=g)[:N_SUB]
        pts = pts[idx.to(pts.device)]
    return pts


def chamfer(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return (0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean())).item()


def crossview(o):
    vl = unnormalize(o["video_dict"]["sampled_video_left"][:, :3], -1, 2)
    vr = unnormalize(o["video_dict"]["sampled_video_right"][:, :3], -1, 2)
    return float(np.mean([chamfer(get_cloud(vl, t, 1000 + t), get_cloud(vr, t, 2000 + t))
                          for t in range(vl.shape[0])]))


print("[1/3] 모델 로드 + student 가중치 주입")
output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
cfg = OmegaConf.load(f"{output_dir}/config.yaml")
for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and "desc" in cfg[key]:
        cfg[key] = cfg[key]["value"]
cls = hydra.utils.get_class(cfg._target_)
cfg.logging.mode = "offline"
cfg.model.params.ckpt_path = f"{output_dir}/4dgen.ckpt"
cfg.training.seed = 42
cfg.training.output_dir = "/home/sun4208/Geo4D/bench_out"
workspace = cls(cfg)
model = workspace.lightning_module_wrapper.to("cuda")
model.eval()
sd = torch.load(args_cli.student_ckpt, map_location="cpu")
missing, unexpected = model.model.load_state_dict(sd["student"], strict=False)
print(f"student 주입: step {sd['step']}, missing={len(missing)}, unexpected={len(unexpected)}")

print("[2/3] 데이터 준비")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
batches = []
for i, b in enumerate(loader):
    if i >= args_cli.n_batches:
        break
    n = b["pointmap"].shape[1]
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True))
    ib["num_video_frames"] = n
    batches.append(ib)

with torch.no_grad():
    model.log_images(batches[0])  # 워밍업

euler_sampler = model.sampler
print(f"[3/3] student 스텝 스윕 (sampler={args_cli.sampler})")
rows = []
for steps in args_cli.steps:
    if args_cli.sampler == "renoise":
        model.sampler = RenoiseSampler(sigmas_for_steps(steps))
        print(f"  renoise σ: {[round(s,1) for s in model.sampler.sigmas]}")
    else:
        model.sampler = euler_sampler
        model.sampler.num_steps = steps
    m = {"PSNR": [], "AbsRel": [], "CV": []}
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for ib in batches:
            out = model.log_images(ib)
            for v in ["left", "right"]:
                m["PSNR"].append(compute_psnr(out, v))
                m["AbsRel"].append(compute_abs_rel(out, v))
            m["CV"].append(crossview(out))
    dt = (time.time() - t0) / len(batches)
    row = (steps, dt, np.mean(m["PSNR"]), np.mean(m["AbsRel"]), np.mean(m["CV"]))
    rows.append(row)
    b = BASELINE.get(steps)
    cmp = (f" | 기준선(teacher) PSNR {b[0]:.2f}/AbsRel {b[1]:.4f}/CV {b[2]:.4f}"
           f" -> CV {100*(row[4]-b[2])/b[2]:+.1f}%") if b else ""
    print(f"[student steps={steps}] {dt:.1f}초/배치 | PSNR {row[2]:.2f} | AbsRel {row[3]:.4f} | CV-Chamfer {row[4]:.4f}{cmp}")

report = [f"=== student({args_cli.student_ckpt.split('/')[-1]}, sampler={args_cli.sampler}) vs teacher(EulerEDM, 학습 없는 축소) — 배치 {len(batches)} ===",
          "steps | time(s) | PSNR^ | AbsRel v | CV-Chamfer v | (teacher CV | 개선율)"]
for r in rows:
    b = BASELINE.get(r[0])
    tail = f" | {b[2]:.4f} | {100*(r[4]-b[2])/b[2]:+.1f}%" if b else ""
    report.append(f"{r[0]:>5} | {r[1]:>7.1f} | {r[2]:>5.2f} | {r[3]:.4f} | {r[4]:.4f}{tail}")
text = "\n".join(report)
print(); print(text)
with open(f"/home/sun4208/Geo4D/bench_out/student_sweep_results{args_cli.tag}.txt", "w") as f:
    f.write(text + "\n")
print()
print(f"저장: ~/Geo4D/bench_out/student_sweep_results{args_cli.tag}.txt")
