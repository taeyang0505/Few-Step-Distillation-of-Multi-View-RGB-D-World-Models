"""오른쪽 뷰 잔여 오차 진단: 프레임별 전역 편향(시간 드리프트)인가?
각 (샘플, 뷰)에 대해 AbsRel을 다섯 가지로 계산:
  raw            : 보정 없음
  cond_affine    : 조건 프레임 기준 프레임0 robust affine (a,b)을 전 프레임에 적용 (현재 'c' 앵커, GT 불필요)
  bg_temporal    : cond_affine 후, 프레임 t의 '정적 배경'(마스크 밖) 깊이를 프레임0 배경에 robust affine으로 맞춤 (GT 불필요)
  oracle_global  : 전 프레임 공통 affine을 GT에 맞춤 (상한 1)
  oracle_frame   : 프레임별 affine을 GT에 맞춤 (상한 2) — 이게 teacher 수준이면 '프레임별 전역 편향' 확정
+ 프레임별 오라클 (a_t, b_t) 추세 출력 (드리프트 확인)
출력: ~/Geo4D/bench_out/frame_diag.txt"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, random, time, json
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps, fit_robust_affine, _unnorm, _affine_params_left, _affine_params_right

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--n_batches", type=int, default=10)
ap.add_argument("--data_seed", type=int, default=1234)
ap.add_argument("--configs", nargs="+", default=["T25", "S3"])
a = ap.parse_args()


def absrel(p, g, m):
    return torch.mean(torch.abs(p[m] - g[m]) / g[m]).item() if m.sum() > 0 else float("nan")


def apply_affine(d, ab):
    return d * ab[0] + ab[1]


def analyze(o, v, mask_lat):
    vd = o["video_dict"]
    pred = _unnorm(vd[f"sampled_video_{v}"][:, 2])            # (T,H,W)
    gt = _unnorm(vd[f"gt_video_{v}"][:, 2])
    valid = (pred > 0) & (gt > 0)
    T = pred.shape[0]
    # 로봇/물체 마스크 (latent 해상도 → 픽셀), 배경 = ~mask
    mk = F.interpolate(mask_lat.float().reshape(T, 1, *mask_lat.shape[-2:]), size=pred.shape[-2:], mode="nearest")[:, 0] > 0.5
    bg = ~mk
    r = {"view": v, "raw": absrel(pred, gt, valid)}
    # cond_affine (프레임0 조건 앵커, GT 불필요)
    abL = _affine_params_left(vd)
    ab = abL if v == "left" else _affine_params_right(vd, abL)
    pc = apply_affine(pred, ab)
    r["cond_affine"] = absrel(pc, gt, valid); r["ab_cond"] = ab
    # bg_temporal: 프레임 t 배경을 프레임0 배경(cond_affine 적용본)에 맞춤
    pb = pc.clone(); abt = []
    d0 = pc[0]
    for t in range(1, T):
        m = bg[t] & bg[0] & (pc[t] > 0) & (d0 > 0)
        at, bt = fit_robust_affine(pc[t], d0, m)
        pb[t] = pc[t] * at + bt; abt.append((at, bt))
    r["bg_temporal"] = absrel(pb, gt, valid); r["ab_bg"] = abt
    # oracle global affine
    ag, bg_ = fit_robust_affine(pred, gt, valid)
    r["oracle_global"] = absrel(pred * ag + bg_, gt, valid)
    # oracle per-frame affine
    pf = pred.clone(); abf = []
    for t in range(T):
        at, bt = fit_robust_affine(pred[t], gt[t], valid[t]); pf[t] = pred[t] * at + bt; abf.append((at, bt))
    r["oracle_frame"] = absrel(pf, gt, valid); r["ab_frame"] = abf
    r["raw_frames"] = [absrel(pred[t], gt[t], valid[t]) for t in range(T)]
    r["bg_share"] = float((bg & valid).float().sum() / valid.float().sum().clamp_min(1))
    return r


print("[1/3] 모델 로드", flush=True)
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
euler = model.sampler
teacher_sd = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
student_sd = torch.load(a.student_ckpt, map_location="cpu")["student"]

print("[2/3] 데이터 (RNG 고정)", flush=True)
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False; cfg.dataloader.batch_size = 1
cfg.dataloader.num_workers = 0; cfg.dataloader.persistent_workers = False
random.seed(a.data_seed); np.random.seed(a.data_seed); torch.manual_seed(a.data_seed)
loader = DataLoader(dataset, **cfg.dataloader)
batches = []
for i, b in enumerate(loader):
    if i >= a.n_batches:
        break
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True)); ib["num_video_frames"] = b["pointmap"].shape[1]
    batches.append(ib)
with torch.no_grad():
    model.log_images(batches[0])

print("[3/3] 측정", flush=True)
res = {}
for name in a.configs:
    who, steps = ("teacher", 25) if name.startswith("T") else ("student", int(name[1:]))
    model.model.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
    model.sampler = RenoiseSampler(sigmas_for_steps(steps)) if who == "student" else euler
    if who == "teacher":
        model.sampler.num_steps = steps
    rows, t0 = [], time.time()
    for ib in batches:
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            o = model.log_images(ib)
        for v in ["left", "right"]:
            rows.append(analyze(o, v, ib["masks" if v == "left" else "masks_right"][0]))
    res[name] = rows
    for v in ["left", "right"]:
        rs = [r for r in rows if r["view"] == v]; mean = lambda k: np.nanmean([r[k] for r in rs])
        print(f"[{name} {v:5s}] raw {mean('raw'):.4f} | cond_affine {mean('cond_affine'):.4f} | bg_temporal {mean('bg_temporal'):.4f} "
              f"| oracle_global {mean('oracle_global'):.4f} | oracle_frame {mean('oracle_frame'):.4f} | 배경비율 {mean('bg_share'):.2f}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

K = ["raw", "cond_affine", "bg_temporal", "oracle_global", "oracle_frame"]
L = [f"=== 프레임별 편향 진단 (data_seed {a.data_seed}, 배치 {a.n_batches}) ===", ""]
for v in ["left", "right"]:
    L.append(f"[{v}] 설정 | raw | cond_affine(GT불필요) | bg_temporal(GT불필요) | oracle_global | oracle_frame")
    for n in a.configs:
        rs = [r for r in res[n] if r["view"] == v]; mean = lambda k: np.nanmean([r[k] for r in rs])
        L.append(f"  {n:>4} | " + " | ".join(f"{mean(k):.4f}" for k in K))
    if len(a.configs) >= 2:
        rs0 = [r for r in res[a.configs[0]] if r["view"] == v]
        for n in a.configs[1:]:
            rs = [r for r in res[n] if r["view"] == v]
            L.append(f"  [{n}−{a.configs[0]}] " + " | ".join(f"{k} {np.nanmean([x[k]-y[k] for x,y in zip(rs, rs0)]):+.4f}" for k in K))
    L.append("")
L.append("[오른쪽 뷰, 프레임별 오라클 a_t / b_t 와 raw AbsRel — 드리프트 확인 (배치별)]")
for n in a.configs:
    L.append(f" {n}:")
    for i, r in enumerate([r for r in res[n] if r["view"] == "right"]):
        L.append(f"  b{i}: a_t " + " ".join(f"{ab[0]:.2f}" for ab in r["ab_frame"]) + " | b_t " + " ".join(f"{ab[1]:+.2f}" for ab in r["ab_frame"])
                 + " | raw_t " + " ".join(f"{x:.2f}" for x in r["raw_frames"]) + f" | cond(a,b)=({r['ab_cond'][0]:.2f},{r['ab_cond'][1]:+.2f})")
L += ["", "판정: oracle_frame이 oracle_global보다 뚜렷이 낮으면 프레임별(시간 가변) 전역 편향. bg_temporal이 oracle_frame에 근접하면 '정적 배경 시간 앵커'가 GT 없이 해결."]
text = "\n".join(L)
print(); print(text)
with open("/home/sun4208/Geo4D/bench_out/frame_diag.txt", "w") as f:
    f.write(text + "\n")
print("\n저장: ~/Geo4D/bench_out/frame_diag.txt")
