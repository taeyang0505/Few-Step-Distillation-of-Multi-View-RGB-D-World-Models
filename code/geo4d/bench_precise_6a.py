"""6-3 정밀 분석: 샘플별 paired 비교 + LPIPS·선명도·시드 다양성 (teacher vs DMD student)
설정: T25/T4/T1 = teacher EulerEDM 25/4/1스텝, S3/S1 = DMD student(step1600) 재노이징 3/1스텝
지표(뷰 단위): PSNR, AbsRel, LPIPS, 선명도(Laplacian var) / 배치 단위: CV-Chamfer / 시드 다양성(앞 N_DIV 배치×시드 4)
출력: ~/Geo4D/bench_out/precise_6a.txt (+ precise_6a_raw.json)"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
import lpips as lpips_lib
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps, enable_cond_anchor, disable_cond_anchor

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--n_batches", type=int, default=10)
ap.add_argument("--n_div", type=int, default=3, help="시드 다양성 측정 배치 수")
ap.add_argument("--tag", default="")
ap.add_argument("--configs", nargs="+", default=["T25", "T4", "T1", "S3", "S1"])
ap.add_argument("--data_seed", type=int, default=1234)
ap.add_argument("--compile", action="store_true", help="--fast에 더해 UNet·VAE 디코더 torch.compile (Step 7-② 설정 F)")
ap.add_argument("--fast", action="store_true", help="student 설정에 bf16 autocast(UNet·conditioner·VAE 디코더) 적용 — Step 7-② 빠른 설정 D의 품질 재측정")
a = ap.parse_args()
SEEDS = [0, 1, 2, 3]
N_SUB = 4096
CONFIGS_ALL = {"T25": ("teacher", "euler", 25), "T4": ("teacher", "euler", 4), "T1": ("teacher", "euler", 1),
               "T3r": ("teacher", "renoise", 3), "T1r": ("teacher", "renoise", 1),
               "S3": ("student", "renoise", 3), "S1": ("student", "renoise", 1)}
def _cfg(n):
    anchor = n.endswith("b") or n.endswith("c")
    core = n[:-1] if anchor else n
    who, samp, steps = CONFIGS_ALL[core]
    return (n, who, samp, steps, (n[-1] if anchor else ""))
CONFIGS = [_cfg(n) for n in a.configs]


def unnormalize(x, mn, mx):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)


LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def sharpness(v):
    return F.conv2d(v.mean(dim=1, keepdim=True), LAP.to(v.device)).var(dim=(1, 2, 3)).mean().item()


def psnr(p, g):
    return (10 * torch.log10(1. / torch.mean((p - g) ** 2))).item()


def gt_valid_mask(o, v):
    """오른쪽 GT의 변환된 무효 픽셀(xyz=이동벡터 t) 제외 — 발견 10"""
    gt = unnormalize(o["video_dict"][f"gt_video_{v}"][:, :3], -1, 2)
    m = gt[:, 2] > 0
    if v == "right":
        ex = o["video_dict"].get("extra", {})
        if "cam_extr" in ex and "cam_extr_right" in ex:
            E1 = ex["cam_extr"].reshape(-1, 4, 4)[0].float(); E2 = ex["cam_extr_right"].reshape(-1, 4, 4)[0].float()
            t = (torch.linalg.inv(E1) @ E2)[:3, 3].to(gt.device).view(1, 3, 1, 1)
            m = m & ~((gt - t).abs().amax(dim=1) < 2e-3)
    return m


def abs_rel(o, v):
    d1 = unnormalize(o["video_dict"][f"sampled_video_{v}"][:, 2], -1, 2)
    d2 = unnormalize(o["video_dict"][f"gt_video_{v}"][:, 2], -1, 2)
    m = (d1 > 0) & gt_valid_mask(o, v)
    return torch.mean(torch.abs(d1[m] - d2[m]) / d2[m]).item()


def cloud(video, t, seed):
    pts = video[t].reshape(3, -1).T
    pts = pts[pts[:, 2] > 0]
    if pts.shape[0] > N_SUB:
        g = torch.Generator(device="cpu").manual_seed(seed)
        pts = pts[torch.randperm(pts.shape[0], generator=g)[:N_SUB].to(pts.device)]
    return pts


def chamfer(x, y):
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    d = torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
    return (0.5 * (d.min(1).values.mean() + d.min(0).values.mean())).item()


def crossview(o):
    vl = unnormalize(o["video_dict"]["sampled_video_left"][:, :3], -1, 2)
    vr = unnormalize(o["video_dict"]["sampled_video_right"][:, :3], -1, 2)
    return float(np.mean([chamfer(cloud(vl, t, 1000 + t), cloud(vr, t, 2000 + t)) for t in range(vl.shape[0])]))


print("[1/3] 모델·LPIPS 로드", flush=True)
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
lpips_net = lpips_lib.LPIPS(net="alex").cuda().eval()

# --fast: bf16 autocast 래핑 (student 설정에서만 켬)
_orig = {"unet": model.model.forward, "cond": model.conditioner.forward,
         "dec_pm": model.first_stage_pointmap_model.decode, "dec_col": model.first_stage_color_model.decode}
def _bf16(fn):
    def w(*x, **k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(*x, **k)
        return out.float() if torch.is_tensor(out) else out
    return w
_compiled = {"done": False}
def set_fast(on):
    if on and a.compile and not _compiled["done"]:
        model.model.diffusion_model = torch.compile(model.model.diffusion_model)
        model.model.diffusion_model_2 = model.model.diffusion_model      # 가중치 공유(발견 8)
        model.first_stage_pointmap_model.decoder = torch.compile(model.first_stage_pointmap_model.decoder)
        model.first_stage_color_model.decoder = torch.compile(model.first_stage_color_model.decoder)
        _orig["unet"] = model.model.forward
        _compiled["done"] = True
        print("  [compile] UNet·VAE 디코더 torch.compile 적용", flush=True)
    model.model.forward = _bf16(_orig["unet"]) if on else _orig["unet"]
    model.conditioner.forward = _bf16(_orig["cond"]) if on else _orig["cond"]
    model.first_stage_pointmap_model.decode = _bf16(_orig["dec_pm"]) if on else _orig["dec_pm"]
    model.first_stage_color_model.decode = _bf16(_orig["dec_col"]) if on else _orig["dec_col"]

print("[2/3] 데이터", flush=True)
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
cfg.dataloader.num_workers = 0
cfg.dataloader.persistent_workers = False
import random
random.seed(a.data_seed); np.random.seed(a.data_seed); torch.manual_seed(a.data_seed)
loader = DataLoader(dataset, **cfg.dataloader)
batches = []
for i, b in enumerate(loader):
    if i >= a.n_batches:
        break
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True))
    ib["num_video_frames"] = b["pointmap"].shape[1]
    batches.append(ib)
with torch.no_grad():
    model.log_images(batches[0])

gt_sharp = []
with torch.no_grad():
    o = model.log_images(batches[0])
    for v in ["left", "right"]:
        gt_sharp.append(sharpness(unnormalize(o["video_dict"][f"gt_video_{v}"][:, 3:], 0, 1)))
GT_SHARP = float(np.mean(gt_sharp))

print("[3/3] 측정", flush=True)
raw = {}   # name -> {"view": [dict per view-sample], "cv": [per batch], "div": [per view-sample], "time": s/batch}
cur = None
for name, who, samp, steps, anchor in CONFIGS:
    if who != cur:
        model.model.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
        cur = who
    model.sampler = RenoiseSampler(sigmas_for_steps(steps)) if samp == "renoise" else euler
    if samp == "euler":
        model.sampler.num_steps = steps
    if anchor:
        enable_cond_anchor(model, per_view=(anchor == "b"), affine=(anchor == "c"))
    else:
        disable_cond_anchor(model)
    set_fast(a.fast and who == "student")
    rec = {"view": [], "cv": [], "div": [], "time": 0.0}
    t0 = time.time()
    n_gen = 0
    for bi, ib in enumerate(batches):
        seeds = SEEDS if bi < a.n_div else [0]
        seed_vid = {"left": [], "right": []}
        for seed in seeds:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                o = model.log_images(ib)
            n_gen += 1
            for v in ["left", "right"]:
                pred = unnormalize(o["video_dict"][f"sampled_video_{v}"][:, 3:], 0, 1)
                gt = unnormalize(o["video_dict"][f"gt_video_{v}"][:, 3:], 0, 1)
                seed_vid[v].append(pred.cpu())
                if seed == 0:
                    with torch.no_grad():
                        lp = lpips_net(pred * 2 - 1, gt * 2 - 1).mean().item()
                    rec["view"].append({"batch": bi, "view": v, "PSNR": psnr(pred, gt), "AbsRel": abs_rel(o, v),
                                        "AbsRel_L" if v == "left" else "AbsRel_R": abs_rel(o, v),
                                        "LPIPS": lp, "sharp": sharpness(pred)})
            if seed == 0:
                rec["cv"].append(crossview(o))
        if len(seeds) > 1:
            for v in ["left", "right"]:
                rec["div"].append(torch.stack(seed_vid[v]).std(dim=0).mean().item())
    rec["time"] = (time.time() - t0) / n_gen
    raw[name] = rec
    mv = lambda k: np.mean([r[k] for r in rec["view"]])
    mvv = lambda vv: np.mean([r["AbsRel"] for r in rec["view"] if r["view"] == vv])
    print(f"[{name}] {rec['time']:.1f}s/생성 | PSNR {mv('PSNR'):.2f} | AbsRel {mv('AbsRel'):.4f} (L {mvv('left'):.4f} / R {mvv('right'):.4f}) | LPIPS {mv('LPIPS'):.4f} | "
          f"선명도 {mv('sharp'):.5f} (GT {GT_SHARP:.5f}) | CV {np.mean(rec['cv']):.4f} | 다양성 {np.mean(rec['div']):.5f}", flush=True)

# ── paired 분석 ──
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

L = [f"=== 정밀 분석 — paired 비교 (data_seed {a.data_seed}, 배치 {a.n_batches} = 뷰 샘플 {2*a.n_batches}, 다양성 배치 {a.n_div}×시드 {len(SEEDS)}) ===",
     f"student: {a.student_ckpt.split('/')[-1]} | GT 선명도 {GT_SHARP:.5f}", "",
     "[평균] 설정 | s/생성 | PSNR^ | AbsRel v (L/R) | LPIPS v | 선명도 | CV-Chamfer v | 시드 다양성  (AbsRel: 오른쪽 GT 가짜 픽셀 제외)"]
for name, *_ in CONFIGS:
    r = raw[name]; mv = lambda k: np.mean([x[k] for x in r["view"]]); mvv = lambda vv: np.mean([x["AbsRel"] for x in r["view"] if x["view"] == vv])
    L.append(f"{name:>4} | {r['time']:5.1f} | {mv('PSNR'):5.2f} | {mv('AbsRel'):.4f} (L {mvv('left'):.4f}/R {mvv('right'):.4f}) | {mv('LPIPS'):.4f} | {mv('sharp'):.5f} | {np.mean(r['cv']):.4f} | {np.mean(r['div']):.5f}")

L += ["", "[paired: X − T25, 샘플별 차이] 지표 | 설정 | 평균차 ± std | X가 나은 샘플 비율 | Wilcoxon p"]
better_sign = {"PSNR": +1, "AbsRel": -1, "LPIPS": -1, "sharp": +1, "CV": -1, "div": +1}
for k in ["PSNR", "AbsRel", "LPIPS", "sharp", "CV", "div"]:
    for name, *_ in CONFIGS[1:]:
        if k == "CV":
            xs, ts = np.array(raw[name]["cv"]), np.array(raw["T25"]["cv"])
        elif k == "div":
            xs, ts = np.array(raw[name]["div"]), np.array(raw["T25"]["div"])
        else:
            xs = np.array([r[k] for r in raw[name]["view"]]); ts = np.array([r[k] for r in raw["T25"]["view"]])
        d = xs - ts
        wins = np.mean(np.sign(d) == better_sign[k]) if len(d) else float("nan")
        p = wilcoxon(xs, ts).pvalue if (wilcoxon and len(d) >= 5 and np.any(d != 0)) else float("nan")
        L.append(f"{k:>6} | {name:>4} | {d.mean():+.4f} ± {d.std():.4f} | {100*wins:5.1f}% | {p:.3f}")

L += ["", "[샘플별 AbsRel] batch/view | " + " | ".join(n for n, *_ in CONFIGS)]
for i, r in enumerate(raw["T25"]["view"]):
    L.append(f"{r['batch']}/{r['view'][0]} | " + " | ".join(f"{raw[n]['view'][i]['AbsRel']:.3f}" for n, *_ in CONFIGS))
L += ["", "해석 가이드: S3가 T25 대비 PSNR·LPIPS·선명도·다양성은 동급(p>0.05 또는 개선)인데 AbsRel만 유의하게 나쁘면 '픽셀 복원, 기하 미달'이 샘플 수준에서 확정."]
text = "\n".join(L)
print(); print(text)
L.insert(1, f"fast(bf16) 적용: {a.fast}, compile: {a.compile} (student 설정에만)")
with open(f"/home/sun4208/Geo4D/bench_out/precise_6a{a.tag}.txt", "w") as f:
    f.write(text + "\n")
with open(f"/home/sun4208/Geo4D/bench_out/precise_6a{a.tag}_raw.json", "w") as f:
    json.dump(raw, f)
print(f"\n저장: ~/Geo4D/bench_out/precise_6a{a.tag}.txt")
