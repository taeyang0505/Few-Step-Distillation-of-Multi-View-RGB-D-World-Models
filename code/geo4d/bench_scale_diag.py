"""6-3 후속 진단: DMD student의 AbsRel 악화가 '깊이 스케일/오프셋 편향'인지 '구조 오류'인지 분리
각 샘플(뷰)에 대해 T25(teacher 25스텝)와 S3(DMD step1600, 3스텝)의 깊이를 GT에 정렬한 뒤 AbsRel 재측정:
  raw      : 정렬 없음 (기존 지표)
  scale    : s = median(gt/pred) 한 개로 스케일 정렬
  affine   : 최소제곱 (a·pred + b) 정렬
  + 피팅된 s, a, b 값 자체 (S3가 일관되게 1에서 벗어나면 편향)
  + 마스크(로봇/물체 vs 배경)별 raw AbsRel (batch에 masks가 있으면)
출력: ~/Geo4D/bench_out/scale_diag_6a.txt"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--n_batches", type=int, default=10)
a = ap.parse_args()


def unnormalize(x, mn, mx):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)


def depth_pair(o, v):
    d1 = unnormalize(o["video_dict"][f"sampled_video_{v}"][:, 2], -1, 2)
    d2 = unnormalize(o["video_dict"][f"gt_video_{v}"][:, 2], -1, 2)
    m = (d1 > 0) & (d2 > 0)
    return d1, d2, m


def absrel(p, g, m):
    return torch.mean(torch.abs(p[m] - g[m]) / g[m]).item()


def analyze(o, v, masks):
    d1, d2, m = depth_pair(o, v)
    p, g = d1[m], d2[m]
    s = torch.median(g / p).item()
    # affine LSQ: g ≈ a p + b
    A = torch.stack([p, torch.ones_like(p)], 1)
    sol = torch.linalg.lstsq(A, g.unsqueeze(1)).solution.squeeze(1)
    aa, bb = sol[0].item(), sol[1].item()
    r = {"raw": absrel(d1, d2, m), "scale": absrel(d1 * s, d2, m), "affine": absrel(d1 * aa + bb, d2, m),
         "s": s, "a": aa, "b": bb, "mean_ratio": (p.mean() / g.mean()).item()}
    if masks is not None:
        mk = masks.to(d1.device)
        if mk.dim() == 4:
            mk = mk[:, 0]
        if mk.shape[-2:] != d1.shape[-2:]:
            mk = F.interpolate(mk.float().unsqueeze(1), size=d1.shape[-2:], mode="nearest").squeeze(1)
        mk = mk.bool()
        if mk.shape[0] != d1.shape[0]:
            mk = mk[:d1.shape[0]]
        fg, bg = m & mk, m & ~mk
        r["raw_fg"] = absrel(d1, d2, fg) if fg.any() else float("nan")
        r["raw_bg"] = absrel(d1, d2, bg) if bg.any() else float("nan")
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

print("[2/3] 데이터", flush=True)
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
batches = []
for i, b in enumerate(loader):
    if i >= a.n_batches:
        break
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True))
    ib["num_video_frames"] = b["pointmap"].shape[1]
    batches.append(ib)
mask_keys = [k for k in batches[0] if "mask" in k]
print("mask keys:", mask_keys, {k: tuple(batches[0][k].shape) for k in mask_keys}, flush=True)
with torch.no_grad():
    model.log_images(batches[0])

print("[3/3] 측정", flush=True)
res = {}
for name, who, steps in [("T25", "teacher", 25), ("S3", "student", 3)]:
    model.model.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
    model.sampler = RenoiseSampler(sigmas_for_steps(steps)) if who == "student" else euler
    if who == "teacher":
        model.sampler.num_steps = steps
    rows = []
    t0 = time.time()
    for bi, ib in enumerate(batches):
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            o = model.log_images(ib)
        for v in ["left", "right"]:
            mk = None
            key = "masks" if v == "left" else "masks_right"
            if key in ib:
                mk = ib[key][0]
            rows.append(analyze(o, v, mk))
    res[name] = rows
    mean = lambda k: np.nanmean([r[k] for r in rows])
    print(f"[{name}] ({time.time()-t0:.0f}s) AbsRel raw {mean('raw'):.4f} | scale정렬 {mean('scale'):.4f} | affine정렬 {mean('affine'):.4f} "
          f"| s(median) {mean('s'):.3f} | a {mean('a'):.3f} b {mean('b'):.3f} | pred/gt 평균비 {mean('mean_ratio'):.3f}"
          + (f" | fg {mean('raw_fg'):.4f} bg {mean('raw_bg'):.4f}" if "raw_fg" in rows[0] else ""), flush=True)

L = ["=== 스케일 진단: AbsRel 악화가 스케일/오프셋 편향인가 구조 오류인가 (배치 10 = 뷰 20) ===", ""]
L.append("설정 | raw | scale정렬 | affine정렬 | s(median gt/pred) | a | b | pred/gt 평균비" + (" | raw_fg(마스크) | raw_bg(배경)" if "raw_fg" in res["T25"][0] else ""))
for n in ["T25", "S3"]:
    rows = res[n]; mean = lambda k: np.nanmean([r[k] for r in rows])
    L.append(f"{n:>3} | {mean('raw'):.4f} | {mean('scale'):.4f} | {mean('affine'):.4f} | {mean('s'):.3f} | {mean('a'):.3f} | {mean('b'):+.3f} | {mean('mean_ratio'):.3f}"
             + (f" | {mean('raw_fg'):.4f} | {mean('raw_bg'):.4f}" if "raw_fg" in rows[0] else ""))
L += ["", "[paired S3 − T25]"]
for k in ["raw", "scale", "affine"]:
    d = np.array([s[k] - t[k] for s, t in zip(res["S3"], res["T25"])])
    L.append(f"{k:>7}: 평균차 {d.mean():+.4f} ± {d.std():.4f} | S3가 나은 샘플 {100*np.mean(d<0):.0f}% | 격차 잔존율 {100*d.mean()/max(1e-9, np.mean([s['raw']-t['raw'] for s,t in zip(res['S3'],res['T25'])])):.0f}%")
L += ["", "[샘플별] view | T25 raw/scale/affine/s | S3 raw/scale/affine/s"]
for i, (t, s) in enumerate(zip(res["T25"], res["S3"])):
    L.append(f"{i//2}/{'lr'[i%2]} | {t['raw']:.3f}/{t['scale']:.3f}/{t['affine']:.3f}/{t['s']:.3f} | {s['raw']:.3f}/{s['scale']:.3f}/{s['affine']:.3f}/{s['s']:.3f}")
L += ["", "판정: scale/affine 정렬 후 S3−T25 격차가 대부분 사라지면(잔존율 <30%) '스케일 편향' → 6-4는 스케일 앵커 항. 남으면(>60%) '구조 오류' → 포인트맵 consistency loss."]
text = "\n".join(L)
print(); print(text)
with open("/home/sun4208/Geo4D/bench_out/scale_diag_6a.txt", "w") as f:
    f.write(text + "\n")
print("\n저장: ~/Geo4D/bench_out/scale_diag_6a.txt")
