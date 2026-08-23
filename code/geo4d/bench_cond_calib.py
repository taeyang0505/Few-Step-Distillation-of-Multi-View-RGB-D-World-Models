"""6-4 후보 ①: 추론 시 스케일 보정 (학습 없음, GT 불필요)
조건 프레임의 포인트맵(모델 입력)에 예측 프레임 0의 깊이를 정렬해 스케일 s(또는 affine a,b)를 구하고 전 프레임에 적용.
비교: raw / cond-scale / cond-affine / oracle(GT 프레임0 scale, 상한) — T25와 S3 각각, paired.
데이터셋 RNG 고정(num_workers=0 + 시드)으로 실행 간 동일 샘플 보장.
출력: ~/Geo4D/bench_out/cond_calib_6a.txt"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, random, time
import numpy as np
import torch
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
ap.add_argument("--data_seed", type=int, default=1234)
a = ap.parse_args()


def unnormalize(x, mn, mx):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)


def absrel(p, g, m):
    return torch.mean(torch.abs(p[m] - g[m]) / g[m]).item()


def fit_scale(p, g):
    m = (p > 0) & (g > 0)
    return torch.median(g[m] / p[m]).item()


def fit_affine(p, g):
    m = (p > 0) & (g > 0)
    A = torch.stack([p[m], torch.ones_like(p[m])], 1)
    sol = torch.linalg.lstsq(A, g[m].unsqueeze(1)).solution.squeeze(1)
    return sol[0].item(), sol[1].item()


def cond_depth(o, v, shape):
    cond = o["video_dict"][f"cond_pointmap_{v}"]
    cond = cond.reshape(-1, *cond.shape[-3:])[-1]
    cd = unnormalize(cond[2], -1, 2)
    if cd.shape != shape:
        cd = torch.nn.functional.interpolate(cd[None, None], size=shape, mode="nearest")[0, 0]
    return cd


def analyze(o, v, s_left):
    """s_left: 왼쪽 뷰 프레임0 vs 왼쪽 조건 프레임으로 구한 스케일 (참조 좌표계 공통 → 양 뷰에 적용)"""
    pred = unnormalize(o["video_dict"][f"sampled_video_{v}"][:, 2], -1, 2)
    gt = unnormalize(o["video_dict"][f"gt_video_{v}"][:, 2], -1, 2)
    cd = cond_depth(o, v, pred.shape[-2:])
    m = (pred > 0) & (gt > 0)
    r = {"view": v, "raw": absrel(pred, gt, m)}
    r["cond_vs_gt0"] = absrel(cd, gt[0], (cd > 0) & (gt[0] > 0))
    r["s_own"] = fit_scale(pred[0], cd); r["s_left"] = s_left
    r["own_scale"] = absrel(pred * r["s_own"], gt, m)
    r["left_scale"] = absrel(pred * s_left, gt, m)
    r["s_oracle0"] = fit_scale(pred[0], gt[0]); r["oracle_scale0"] = absrel(pred * r["s_oracle0"], gt, m)
    r["s_oracle_all"] = fit_scale(pred, gt); r["oracle_scale_all"] = absrel(pred * r["s_oracle_all"], gt, m)
    a_o, b_o = fit_affine(pred, gt); r["oracle_affine"] = absrel(pred * a_o + b_o, gt, m)
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
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
cfg.dataloader.num_workers = 0
cfg.dataloader.persistent_workers = False
random.seed(a.data_seed); np.random.seed(a.data_seed); torch.manual_seed(a.data_seed)
loader = DataLoader(dataset, **cfg.dataloader)
batches = []
for i, b in enumerate(loader):
    if i >= a.n_batches:
        break
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True))
    ib["num_video_frames"] = b["pointmap"].shape[1]
    batches.append(ib)
print("batch idx:", [int(b["idx"][0]) for b in batches], "| cond_pointmap shape:", tuple(batches[0]["cond_pointmaps_without_noise"].shape), flush=True)
with torch.no_grad():
    model.log_images(batches[0])

print("[3/3] 측정", flush=True)
res = {}
for name, who, steps in [("T25", "teacher", 25), ("S3", "student", 3)]:
    model.model.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
    model.sampler = RenoiseSampler(sigmas_for_steps(steps)) if who == "student" else euler
    if who == "teacher":
        model.sampler.num_steps = steps
    rows, t0 = [], time.time()
    for ib in batches:
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            o = model.log_images(ib)
        pl = unnormalize(o["video_dict"]["sampled_video_left"][:, 2], -1, 2)
        s_left = fit_scale(pl[0], cond_depth(o, "left", pl.shape[-2:]))
        for v in ["left", "right"]:
            rows.append(analyze(o, v, s_left))
    res[name] = rows
    for v in ["left", "right", "all"]:
        rs = [r for r in rows if v == "all" or r["view"] == v]; mean = lambda k: np.mean([r[k] for r in rs])
        print(f"[{name} {v:5s}] raw {mean('raw'):.4f} | own-scale {mean('own_scale'):.4f} | LEFT-scale {mean('left_scale'):.4f} | oracle0 {mean('oracle_scale0'):.4f} "
              f"| oracle-all {mean('oracle_scale_all'):.4f} | oracle-affine {mean('oracle_affine'):.4f} | s_own {mean('s_own'):.3f} s_left {mean('s_left'):.3f} s_oracle0 {mean('s_oracle0'):.3f} | cond vs gt0 {mean('cond_vs_gt0'):.4f}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

import json
K = ["raw", "own_scale", "left_scale", "oracle_scale0", "oracle_scale_all", "oracle_affine"]
L = [f"=== 추론 시 스케일 보정 v2 (왼쪽 조건 프레임 스케일을 양 뷰에 적용) — 배치 {a.n_batches}=뷰 20, data_seed {a.data_seed} ===", ""]
for v in ["left", "right", "all"]:
    L.append(f"[{v}] 설정 | raw | own-scale | LEFT-scale | oracle0 | oracle-all | oracle-affine | s_own | s_left | s_oracle0 | cond vs gt0")
    for n in ["T25", "S3"]:
        rs = [r for r in res[n] if v == "all" or r["view"] == v]; mean = lambda k: np.mean([r[k] for r in rs])
        L.append(f"  {n:>3} | " + " | ".join(f"{mean(k):.4f}" for k in K) + f" | {mean('s_own'):.3f} | {mean('s_left'):.3f} | {mean('s_oracle0'):.3f} | {mean('cond_vs_gt0'):.4f}")
    L.append("  [paired S3−T25] " + " | ".join(f"{k} {np.mean([x[k]-y[k] for x,y in zip([r for r in res['S3'] if v=='all' or r['view']==v],[r for r in res['T25'] if v=='all' or r['view']==v])]):+.4f}" for k in K))
    L.append("")
L.append("[S3 자체 개선, 뷰별] " + " | ".join(f"{v}: raw→LEFT-scale {np.mean([r['left_scale']-r['raw'] for r in res['S3'] if r['view']==v]):+.4f} ({100*np.mean([r['left_scale']/r['raw']-1 for r in res['S3'] if r['view']==v]):+.0f}%), 개선 샘플 {100*np.mean([r['left_scale']<r['raw'] for r in res['S3'] if r['view']==v]):.0f}%" for v in ["left","right"]))
L += ["", "[샘플별] view | T25 raw/LEFT/oracle0 | S3 raw/LEFT/oracle0 | S3 s_left/s_oracle0"]
for i, (t, r) in enumerate(zip(res["T25"], res["S3"])):
    L.append(f"{i//2}/{r['view'][0]} | {t['raw']:.3f}/{t['left_scale']:.3f}/{t['oracle_scale0']:.3f} | {r['raw']:.3f}/{r['left_scale']:.3f}/{r['oracle_scale0']:.3f} | {r['s_left']:.3f}/{r['s_oracle0']:.3f}")
text = "\n".join(L)
print(); print(text)
with open("/home/sun4208/Geo4D/bench_out/cond_calib_6a_v2.txt", "w") as f:
    f.write(text + "\n")
with open("/home/sun4208/Geo4D/bench_out/cond_calib_6a_v2_raw.json", "w") as f:
    json.dump(res, f)
print("\n저장: ~/Geo4D/bench_out/cond_calib_6a_v2.txt")
