"""Step 8 대리 지표: 정책이 실제로 읽는 영역(그리퍼·사과)의 생성 품질 — teacher vs student, paired
시뮬레이터·포즈 추적기 없이, GT label 맵(그리퍼 id 29–35, 사과 id 44)으로 영역을 잘라 잰다.
영역별: AbsRel(깊이), PSNR(색), 3D 중심 오차[m] (예측 포인트맵 중심 vs GT 포인트맵 중심, 같은 마스크) — 포즈 추적의 위치 오차 대리.
프레임별 중심 오차(1→10)로 horizon에 따른 오차 증가도 본다.
출력: ~/Geo4D/bench_out/policy_proxy{tag}.txt (+ _raw.json)"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, json, time, random
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps, enable_cond_anchor, disable_cond_anchor

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--n_batches", type=int, default=20)
ap.add_argument("--tag", default="")
ap.add_argument("--configs", nargs="+", default=["T25", "T3r", "S3b"])
ap.add_argument("--data_seed", type=int, default=1234)
ap.add_argument("--fast", action="store_true", help="student 설정에 bf16 autocast")
ap.add_argument("--dilate", type=int, default=3, help="AbsRel/PSNR용 마스크 팽창 픽셀 (중심 오차는 팽창 없음)")
a = ap.parse_args()

GRIPPER_IDS = [29, 30, 31, 33, 34, 35]   # 데이터셋 masks와 동일 (오른팔 29–31, 왼팔 33–35)
APPLE_ID = 44                             # 확인: 왼팔이 집어 올려 오른팔에 건네 통에 놓는 물체
CONFIGS_ALL = {"T25": ("teacher", "euler", 25), "T4": ("teacher", "euler", 4), "T1": ("teacher", "euler", 1),
               "T3r": ("teacher", "renoise", 3), "T1r": ("teacher", "renoise", 1),
               "S3": ("student", "renoise", 3), "S1": ("student", "renoise", 1),
               "S4": ("student", "renoise", 4), "S5": ("student", "renoise", 5),
               "A4": ("student", "renoise_avg2", 3), "A6": ("student", "renoise_avg2", 4),
               "H3": ("hybrid", "renoise", 3), "H4": ("hybrid", "renoise", 4)}
def _cfg(n):
    anchor = n.endswith("b") or n.endswith("c")
    core = n[:-1] if anchor else n
    who, samp, steps = CONFIGS_ALL[core]
    return (n, who, samp, steps, (n[-1] if anchor else ""))
CONFIGS = [_cfg(n) for n in a.configs]

def unnormalize(x, mn, mx):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)

def gt_valid_mask(vd, v):
    gt = unnormalize(vd[f"gt_video_{v}"][:, :3], -1, 2)
    m = gt[:, 2] > 0
    ex = vd.get("extra", {})
    if v == "right" and "cam_extr" in ex and "cam_extr_right" in ex:
        E1 = ex["cam_extr"].reshape(-1, 4, 4)[0].float(); E2 = ex["cam_extr_right"].reshape(-1, 4, 4)[0].float()
        t = (torch.linalg.inv(E1) @ E2)[:3, 3].to(gt.device).view(1, 3, 1, 1)
        m = m & ~((gt - t).abs().amax(dim=1) < 2e-3)
    return m

def dilate(m, px):
    if px <= 0: return m
    return F.max_pool2d(m[:, None].float(), 2 * px + 1, 1, px)[:, 0] > 0.5

def region_metrics(vd, v, mask_raw):
    """mask_raw: (T,H,W) bool — 해당 영역. 반환: AbsRel, PSNR, 중심오차 평균/마지막, 프레임별 중심오차"""
    pred = vd[f"sampled_video_{v}"]; gt = vd[f"gt_video_{v}"]
    pxyz = unnormalize(pred[:, :3], -1, 2); gxyz = unnormalize(gt[:, :3], -1, 2)
    prgb = unnormalize(pred[:, 3:], 0, 1); grgb = unnormalize(gt[:, 3:], 0, 1)
    valid = (pxyz[:, 2] > 0) & gt_valid_mask(vd, v)
    md = dilate(mask_raw, a.dilate) & valid
    out = {}
    if md.sum() >= 50:
        out["AbsRel"] = torch.mean(torch.abs(pxyz[:, 2][md] - gxyz[:, 2][md]) / gxyz[:, 2][md]).item()
        m4 = md[:, None].expand_as(prgb)
        out["PSNR"] = (10 * torch.log10(1. / torch.mean((prgb[m4] - grgb[m4]) ** 2))).item()
    errs = []
    for t in range(pred.shape[0]):
        m = mask_raw[t] & valid[t]
        if m.sum() < 30:
            errs.append(float("nan")); continue
        errs.append(torch.norm(pxyz[t][:, m].mean(1) - gxyz[t][:, m].mean(1)).item())
    e = np.array(errs)
    if np.isfinite(e).sum() > 0:
        out["cent"] = float(np.nanmean(e)); out["cent_last"] = float(e[np.isfinite(e)][-1]); out["cent_t"] = [None if not np.isfinite(x) else float(x) for x in e]
    return out

print("[1/3] 모델 로드", flush=True)
output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
cfg = OmegaConf.load(f"{output_dir}/config.yaml")
for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and "desc" in cfg[key]:
        cfg[key] = cfg[key]["value"]
cls = hydra.utils.get_class(cfg._target_)
cfg.logging.mode = "offline"; cfg.model.params.ckpt_path = f"{output_dir}/4dgen.ckpt"
cfg.training.seed = 42; cfg.training.output_dir = "/home/sun4208/Geo4D/bench_out"
workspace = cls(cfg)
model = workspace.lightning_module_wrapper.to("cuda"); model.eval()
euler = model.sampler
teacher_sd = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
student_sd = torch.load(a.student_ckpt, map_location="cpu")["student"]
plain = model.model
import copy as _copy

class HybridWrapper(torch.nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student, self.teacher = student, teacher
        self.use_teacher = False
        self.bf16 = False

    def forward(self, *a, **k):
        m = self.teacher if self.use_teacher else self.student
        if self.bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(*a, **k)
            return out.float()
        return m(*a, **k)


_hybrid = {"mod": None}


def get_hybrid():
    if _hybrid["mod"] is None:
        plain.__dict__.pop("forward", None)
        t = _copy.deepcopy(plain)
        t.__dict__.pop("forward", None)
        t.load_state_dict(teacher_sd, strict=False)
        t.eval().requires_grad_(False)
        _hybrid["mod"] = HybridWrapper(plain, t.to("cuda"))
        print("  [hybrid] teacher 사본 탑재", flush=True)
    return _hybrid["mod"]


_orig = {"unet": model.model.forward, "cond": model.conditioner.forward,
         "dec_pm": model.first_stage_pointmap_model.decode, "dec_col": model.first_stage_color_model.decode}
def _bf16(fn):
    def w(*x, **k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(*x, **k)
        return out.float() if torch.is_tensor(out) else out
    return w
def set_fast(on, unet_on=None):
    unet_on = on if unet_on is None else unet_on
    plain.forward = _bf16(_orig["unet"]) if unet_on else _orig["unet"]
    model.conditioner.forward = _bf16(_orig["cond"]) if on else _orig["cond"]
    model.first_stage_pointmap_model.decode = _bf16(_orig["dec_pm"]) if on else _orig["dec_pm"]
    model.first_stage_color_model.decode = _bf16(_orig["dec_col"]) if on else _orig["dec_col"]

print("[2/3] 데이터 (+ label·카메라 기록)", flush=True)
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False; cfg.dataloader.batch_size = 1; cfg.dataloader.num_workers = 0; cfg.dataloader.persistent_workers = False
# __getitem__ 내부에서 실제 쓰인 원본 시퀀스(정적 필터로 idx가 바뀔 수 있음)와 선택된 카메라 2개를 가로채 기록
_last = {}
_orig_ss = dataset.sampler.sample_sequence
def _ss(idx, f, s):
    r = _orig_ss(idx, f, s); _last["raw"] = r; return r
dataset.sampler.sample_sequence = _ss
_orig_choice = np.random.choice
def _choice(arr, *args, **kw):
    out = _orig_choice(arr, *args, **kw)
    if isinstance(arr, (list, np.ndarray)) and len(args) > 0 and args[0] == 2:
        _last["cams"] = [str(x) for x in out]
    return out
np.random.choice = _choice
random.seed(a.data_seed); np.random.seed(a.data_seed); torch.manual_seed(a.data_seed)
loader = DataLoader(dataset, **cfg.dataloader)
batches, labels = [], []
for i, b in enumerate(loader):
    if i >= a.n_batches: break
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True)); ib["num_video_frames"] = b["pointmap"].shape[1]
    raw = _last["raw"]; cams = _last["cams"]
    lab = {"left": torch.from_numpy(raw[f"obs.{cams[0]}_label"][1:].astype(np.int64)),   # 프레임0은 조건, 1..10이 예측 대상
           "right": torch.from_numpy(raw[f"obs.{cams[1]}_label"][1:].astype(np.int64)), "cams": cams}
    assert lab["left"].shape[0] == ib["num_video_frames"], (lab["left"].shape, ib["num_video_frames"])
    batches.append(ib); labels.append(lab)
np.random.choice = _orig_choice
print(f"  배치 {len(batches)}개, 카메라 예: {labels[0]['cams']}", flush=True)
# 영역 크기 통계 (검증용)
gp = np.mean([torch.isin(l[v], torch.tensor(GRIPPER_IDS)).float().mean().item() for l in labels for v in ["left", "right"]])
apx = np.mean([(l[v] == APPLE_ID).float().mean().item() for l in labels for v in ["left", "right"]])
print(f"  영역 비율: 그리퍼 {gp*100:.2f}% 사과 {apx*100:.2f}% (픽셀)", flush=True)

print("[3/3] 측정", flush=True)
raw_out = {}; cur = None
REGIONS = ["grip", "apple", "bg"]
for name, who, samp, steps, anchor in CONFIGS:
    if who != cur:
        if who == "hybrid":
            hy = get_hybrid(); plain.load_state_dict(student_sd, strict=False); model.model = hy
        else:
            model.model = plain; plain.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
        cur = who
    if samp == "renoise_avg2":
        model.sampler = RenoiseSampler(sigmas_for_steps(steps), avg_final=2)
    elif who == "hybrid":
        model.sampler = RenoiseSampler(sigmas_for_steps(steps), final_toggle=model.model)
    else:
        model.sampler = RenoiseSampler(sigmas_for_steps(steps)) if samp == "renoise" else euler
    if samp == "euler": model.sampler.num_steps = steps
    if anchor: enable_cond_anchor(model, per_view=(anchor == "b"), affine=(anchor == "c"))
    else: disable_cond_anchor(model)
    if who == "hybrid":
        set_fast(a.fast, unet_on=False)
        model.model.bf16 = a.fast
    else:
        set_fast(a.fast and who == "student")
    rec = []; t0 = time.time()
    for bi, (ib, lab) in enumerate(zip(batches, labels)):
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            vd = model.log_images(ib)["video_dict"]
        for v in ["left", "right"]:
            L = lab[v].to(vd[f"gt_video_{v}"].device)
            mg = torch.isin(L, torch.tensor(GRIPPER_IDS, device=L.device)); ma = (L == APPLE_ID); mb = ~(mg | ma)
            r = {"batch": bi, "view": v}
            for reg, m in (("grip", mg), ("apple", ma), ("bg", mb)):
                for k, val in region_metrics(vd, v, m).items():
                    r[f"{reg}_{k}"] = val
            rec.append(r)
    raw_out[name] = {"view": rec, "time": (time.time() - t0) / len(batches)}
    mv = lambda k: np.nanmean([x.get(k, np.nan) for x in rec])
    print(f"[{name}] {raw_out[name]['time']:.1f}s/배치 | 그리퍼 AbsRel {mv('grip_AbsRel'):.4f} PSNR {mv('grip_PSNR'):.2f} 중심오차 {mv('grip_cent')*100:.2f}cm (마지막 {mv('grip_cent_last')*100:.2f}) | "
          f"사과 AbsRel {mv('apple_AbsRel'):.4f} PSNR {mv('apple_PSNR'):.2f} 중심오차 {mv('apple_cent')*100:.2f}cm | 배경 AbsRel {mv('bg_AbsRel'):.4f}", flush=True)

REF = "T25" if "T25" in raw_out else a.configs[0]
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
KEYS = [("grip_AbsRel", "그리퍼 AbsRel v", 1, ".4f"), ("grip_PSNR", "그리퍼 PSNR ^", 1, ".2f"), ("grip_cent", "그리퍼 중심오차 cm v", 100, ".2f"),
        ("grip_cent_last", "그리퍼 중심오차(마지막 프레임) cm v", 100, ".2f"),
        ("apple_AbsRel", "사과 AbsRel v", 1, ".4f"), ("apple_PSNR", "사과 PSNR ^", 1, ".2f"), ("apple_cent", "사과 중심오차 cm v", 100, ".2f"),
        ("bg_AbsRel", "배경 AbsRel v", 1, ".4f")]
L = [f"=== Step 8 대리 지표: 정책이 읽는 영역(그리퍼·사과)의 생성 품질 — paired (data_seed {a.data_seed}, 배치 {a.n_batches} = 뷰 샘플 {2*a.n_batches}) ===",
     f"student: {a.student_ckpt.split('/')[-1]} | 마스크: GT label (그리퍼 {GRIPPER_IDS}, 사과 {APPLE_ID}), AbsRel/PSNR은 {a.dilate}px 팽창, 중심오차는 원마스크 | 영역 비율 그리퍼 {gp*100:.2f}% 사과 {apx*100:.2f}%",
     "중심오차 = 같은 마스크 안 예측 포인트맵 중심 − GT 포인트맵 중심 [cm] (포즈 추적 위치 오차의 대리)", "",
     "[평균] 지표 | " + " | ".join(n for n, *_ in CONFIGS)]
for k, lab_, sc, fmt in KEYS:
    L.append(f"{lab_:<28} | " + " | ".join(format(np.nanmean([x.get(k, np.nan) for x in raw_out[n]['view']]) * sc, fmt) for n, *_ in CONFIGS))
L += ["", f"[paired vs {REF}] 지표 | 설정 | 차이(설정−{REF}) | Wilcoxon p | 더 나쁜 샘플 비율"]
for k, lab_, sc, fmt in KEYS:
    for n, *_ in CONFIGS:
        if n == REF: continue
        xa = np.array([x.get(k, np.nan) for x in raw_out[REF]["view"]]); xb = np.array([x.get(k, np.nan) for x in raw_out[n]["view"]])
        ok = np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 5: continue
        d = xb[ok] - xa[ok]
        p = wilcoxon(xb[ok], xa[ok]).pvalue if (wilcoxon and np.any(d != 0)) else float("nan")
        worse = (d > 0).mean() if "PSNR" not in k else (d < 0).mean()
        L.append(f"{lab_:<28} | {n:>4} | {format(d.mean()*sc, '+' + fmt)} | {p:.3g} | {worse*100:.0f}% (n={ok.sum()})")
L += ["", "[프레임별 그리퍼 중심오차 cm, 1→10 프레임 평균] 설정 | " + " ".join(f"f{t+1}" for t in range(10))]
for n, *_ in CONFIGS:
    ct = np.array([[np.nan if v is None else v for v in x["grip_cent_t"]] for x in raw_out[n]["view"] if "grip_cent_t" in x])
    L.append(f"{n:>4} | " + " ".join(f"{v*100:.1f}" for v in np.nanmean(ct, 0)))
L += ["", "판정: student의 그리퍼·사과 중심오차와 AbsRel이 teacher와 p>0.05이거나 차이가 1cm 미만이면 '정책이 읽는 부분은 보존' 근거.",
      "      배경 AbsRel은 악화됐는데 그리퍼·사과는 유지되면, 전체 AbsRel 차이(+0.015)가 정책에 덜 중요한 부분에서 온다는 뜻."]
text = "\n".join(L); print(); print(text)
out = f"/home/sun4208/Geo4D/bench_out/policy_proxy{a.tag}"
open(out + ".txt", "w").write(text + "\n")
json.dump({"configs": a.configs, "raw": raw_out, "cams": [l["cams"] for l in labels]}, open(out + "_raw.json", "w"))
print(f"\n저장: {out}.txt / _raw.json")
