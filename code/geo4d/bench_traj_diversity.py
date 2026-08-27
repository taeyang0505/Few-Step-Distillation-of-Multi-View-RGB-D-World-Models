"""궤적 다양성: 시드별 그리퍼 3D 궤적의 퍼짐과 best-of-N 오차. 깊이 오차가 아니라 "팔이 어디로 가는가"의 다양성을 잰다.
출력: ~/Geo4D/bench_out/traj_diversity{tag}.txt"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, os, json, time, random
import numpy as np
import torch
import torch.nn.functional as F
import lpips as lpips_lib
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps, enable_cond_anchor, disable_cond_anchor

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--n_batches", type=int, default=10)
ap.add_argument("--tag", default="")
ap.add_argument("--configs", nargs="+", default=["T25", "T3r", "S3b", "S5b"])
ap.add_argument("--frames", type=int, default=10)
ap.add_argument("--seeds", type=int, default=8, help="샘플당 생성할 시드 수 N")
ap.add_argument("--data_seed", type=int, default=1234)
ap.add_argument("--fast", action="store_true", help="student 설정에 bf16 autocast")
ap.add_argument("--dilate", type=int, default=3, help="AbsRel/PSNR용 마스크 팽창 픽셀")
ap.add_argument("--motion_thr", type=float, default=0.02, help="moving 마스크 임계 [m]")
a = ap.parse_args()

GRIPPER_IDS = [int(x) for x in os.environ.get("GEO4D_GRIPPER_IDS", "29,30,31,33,34,35").split(",")]   # 데이터셋 masks와 동일 (오른팔 29–31, 왼팔 33–35)
APPLE_ID = int(os.environ.get("GEO4D_OBJECT_ID", "44"))                             # 확인: 왼팔이 집어 올려 오른팔에 건네 통에 놓는 물체
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

LAP_K = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)

def erode(m, px=1):
    if px <= 0: return m
    return ~(F.max_pool2d((~m)[:, None].float(), 2 * px + 1, 1, px)[:, 0] > 0.5)

def lap_var(gray, mask):
    """라플라시안 분산을 마스크 안에서만. 커널이 마스크 밖을 물지 않도록 1px 침식."""
    r = F.conv2d(gray[:, None], LAP_K.to(gray.device), padding=1)[:, 0]
    m = erode(mask, 1)
    if m.sum() < 200: return float("nan")
    x = r[m]
    return float(x.var().item())

def motion_mask(vd, v, thr):
    """조건 프레임(=프레임0) 대비 GT 3D 위치가 thr 이상 바뀐 픽셀 = 실제로 움직인 영역.
    주의(08-26 버그 수정): 오른쪽 뷰의 GT 포인트맵은 참조(왼쪽) 좌표계로 변환돼 있는데 조건 포인트맵은
    오른쪽 카메라 좌표계다. 그냥 빼면 좌표계 차이가 전부 '움직임'으로 잡혀 moving 이 53%까지 부풀었다.
    앵커(cond_anchor_scale_right)와 동일하게 외부 파라미터로 조건을 참조 프레임에 옮긴 뒤 비교한다."""
    gt = unnormalize(vd[f"gt_video_{v}"][:, :3], -1, 2)                       # (T,3,H,W)
    cond = vd[f"cond_pointmap_{v}"]
    cond = unnormalize(cond.reshape(-1, *cond.shape[-3:])[-1], -1, 2)         # (3,H,W) 자기 카메라 프레임
    ex = vd.get("extra", {})
    if v == "right":
        if "cam_extr" not in ex or "cam_extr_right" not in ex:
            return torch.zeros(gt.shape[0], *gt.shape[-2:], dtype=torch.bool, device=gt.device)   # 변환 불가 → 측정 포기
        E1 = ex["cam_extr"].reshape(-1, 4, 4)[0].to(gt.device).float()
        E2 = ex["cam_extr_right"].reshape(-1, 4, 4)[0].to(gt.device).float()
        T = torch.linalg.inv(E1) @ E2
        valid_c = cond[2] > 0
        cond = (T[:3, :3] @ cond.reshape(3, -1) + T[:3, 3:4]).reshape(3, *cond.shape[-2:])
        cond[2] = torch.where(valid_c, cond[2], torch.zeros_like(cond[2]))
    if cond.shape[-2:] != gt.shape[-2:]:
        cond = F.interpolate(cond[None], size=gt.shape[-2:], mode="nearest")[0]
    valid = (gt[:, 2] > 0) & (cond[2][None] > 0)
    return (torch.linalg.norm(gt - cond[None], dim=1) > thr) & valid

def region_metrics(vd, v, mask_raw, lp_map, gray_p, gray_g):
    """mask_raw: (T,H,W) bool. 반환: AbsRel, PSNR, LPIPS, 선명도(pred/gt)"""
    pred = vd[f"sampled_video_{v}"]; gt = vd[f"gt_video_{v}"]
    pxyz = unnormalize(pred[:, :3], -1, 2); gxyz = unnormalize(gt[:, :3], -1, 2)
    prgb = unnormalize(pred[:, 3:], 0, 1); grgb = unnormalize(gt[:, 3:], 0, 1)
    valid = (pxyz[:, 2] > 0) & gt_valid_mask(vd, v)
    md = dilate(mask_raw, a.dilate) & valid
    out = {"px": float(mask_raw.float().mean().item())}
    if md.sum() >= 50:
        out["AbsRel"] = torch.mean(torch.abs(pxyz[:, 2][md] - gxyz[:, 2][md]) / gxyz[:, 2][md]).item()
        m4 = md[:, None].expand_as(prgb)
        out["PSNR"] = (10 * torch.log10(1. / torch.mean((prgb[m4] - grgb[m4]) ** 2))).item()
    md2 = dilate(mask_raw, a.dilate)
    if md2.sum() >= 200:
        out["LPIPS"] = float(lp_map[md2].mean().item())
        out["sharp"] = lap_var(gray_p, md2)
        out["sharp_gt"] = lap_var(gray_g, md2)
    return out

print("[1/3] 모델 로드", flush=True)
output_dir = os.environ.get("GEO4D_TEACHER_DIR", "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple")   # 태스크 전환: 환경변수
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

lpips_net = lpips_lib.LPIPS(net="alex", spatial=True).to("cuda").eval()

print("[2/3] 데이터 (+ label·카메라 기록)", flush=True)
cfg.task = OmegaConf.load(os.environ.get("GEO4D_TASK_YAML", "/home/sun4208/4dgen/config/task/inference.yaml"))
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



# ────────────── 궤적 다양성: 예측된 미래들이 "팔이 어디로 가는지"에서 실제로 갈리는가 ──────────────
# 깊이 오차의 다양성(best_of_n)과 달리, 그리퍼 3D 중심의 궤적이 시드마다 얼마나 벌어지는지를 잰다.
# 계획 관점에서 "여러 미래"의 원래 의미에 더 가깝다.
print(f"[3/3] 궤적 다양성 (설정 {len(CONFIGS)} × 배치 {len(batches)} × 시드 {a.seeds})", flush=True)

def grip_centroids(vd, v, L):
    """프레임별 그리퍼 3D 중심 (예측, GT) — 단위 m. 유효 픽셀 부족하면 NaN."""
    pred = vd[f"sampled_video_{v}"]; gt = vd[f"gt_video_{v}"]
    pxyz = unnormalize(pred[:, :3], -1, 2); gxyz = unnormalize(gt[:, :3], -1, 2)
    valid = (pxyz[:, 2] > 0) & gt_valid_mask(vd, v)
    mg = torch.isin(L, torch.tensor(GRIPPER_IDS, device=L.device))
    P, G = [], []
    for t in range(pxyz.shape[0]):
        m = mg[t] & valid[t]
        if m.sum() < 30:
            P.append([float("nan")] * 3); G.append([float("nan")] * 3); continue
        P.append(pxyz[t][:, m].mean(1).tolist()); G.append(gxyz[t][:, m].mean(1).tolist())
    return np.array(P), np.array(G)

raw_out = {}; cur = None
for name, who, samp, steps, anchor in CONFIGS:
    if who != cur:
        if who == "hybrid":
            hy = get_hybrid(); plain.load_state_dict(student_sd, strict=False); model.model = hy
        else:
            model.model = plain; plain.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
        cur = who
    if samp == "renoise_avg2": model.sampler = RenoiseSampler(sigmas_for_steps(steps), avg_final=2)
    elif who == "hybrid":      model.sampler = RenoiseSampler(sigmas_for_steps(steps), final_toggle=model.model)
    else:                      model.sampler = RenoiseSampler(sigmas_for_steps(steps)) if samp == "renoise" else euler
    if samp == "euler": model.sampler.num_steps = steps
    if anchor: enable_cond_anchor(model, per_view=(anchor == "b"), affine=(anchor == "c"))
    else: disable_cond_anchor(model)
    if who == "hybrid":
        set_fast(a.fast, unet_on=False); model.model.bf16 = a.fast
    else:
        set_fast(a.fast and who == "student")

    rec = []; t0 = time.time()
    for bi, (ib, lab) in enumerate(zip(batches, labels)):
        for v in ["left", "right"]:
            L = lab[v].to("cuda")
            traj = []                       # (시드, 프레임, 3)
            gtc = None
            for sd in range(a.seeds):
                torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
                with torch.no_grad():
                    vd = model.log_images(ib)["video_dict"]
                P, G = grip_centroids(vd, v, L)
                traj.append(P)
                if gtc is None: gtc = G
            traj = np.array(traj)           # (S, T, 3)
            # ① 궤적 퍼짐: 프레임별 시드간 3D 위치 표준편차의 크기 [cm]
            spread = np.nanmean(np.linalg.norm(np.nanstd(traj, axis=0), axis=-1)) * 100
            # ② 시드별 궤적 오차 [cm] (프레임 평균) → best-of-N 용
            err = np.nanmean(np.linalg.norm(traj - gtc[None], axis=-1), axis=1) * 100
            rec.append({"batch": bi, "view": v, "spread_cm": float(spread),
                        "err_per_seed_cm": [float(x) for x in err],
                        "spread_per_frame_cm": [float(x) for x in (np.linalg.norm(np.nanstd(traj, axis=0), axis=-1) * 100)]})
    raw_out[name] = {"view": rec, "time": (time.time() - t0) / max(len(batches) * a.seeds * 2, 1)}
    sp = np.nanmean([r["spread_cm"] for r in rec])
    e = np.array([r["err_per_seed_cm"] for r in rec], dtype=float)
    print(f"[{name}] 궤적 퍼짐 {sp:.2f}cm | 오차 평균 {np.nanmean(e):.2f}cm | best-of-1 {np.nanmean(e[:,0]):.2f} → best-of-{a.seeds} {np.nanmean(np.nanmin(e,1)):.2f}cm", flush=True)

L = [f"=== 궤적 다양성: 예측된 미래들이 그리퍼 경로에서 실제로 갈리는가 (배치 {a.n_batches} = 뷰 샘플 {2*a.n_batches}, 시드 {a.seeds}) ===",
     "궤적 퍼짐 = 프레임별 시드간 그리퍼 3D 중심 표준편차의 크기 [cm] 평균. 클수록 서로 다른 미래를 예측.",
     "best-of-N = 시드 N개 중 GT 궤적에 가장 가까운 것의 오차 [cm].", "",
     "설정 | 궤적 퍼짐(cm) | 오차 평균(cm) | " + " | ".join(f"N={n}" for n in range(1, a.seeds+1)) + " | 이득"]
summary = {}
for name, *_ in CONFIGS:
    rec = raw_out[name]["view"]
    e = np.array([r["err_per_seed_cm"] for r in rec], dtype=float)
    curve = [float(np.nanmean(np.nanmin(e[:, :n], axis=1))) for n in range(1, a.seeds+1)]
    sp = float(np.nanmean([r["spread_cm"] for r in rec]))
    gain = (curve[0] - curve[-1]) / curve[0] * 100
    summary[name] = {"spread_cm": sp, "mean_err_cm": float(np.nanmean(e)), "curve": curve, "gain_pct": gain}
    L.append(f"{name:>5} | {sp:.2f} | {np.nanmean(e):.2f} | " + " | ".join(f"{c:.2f}" for c in curve) + f" | {gain:.1f}%")
L += ["", "판정: 궤적 퍼짐이 크면서 best-of-N 이득도 크면 '쓸모 있는 다양성'.",
      "  퍼짐만 크고 이득이 없으면 무작위 흔들림이고, 둘 다 작으면 평균 붕괴다.",
      "주의: GT를 보고 최선을 고르는 oracle 상한이며 실제 로봇은 이렇게 못 고른다. 상한조차 작으면 어떤 선택 전략도 무의미하다는 뜻."]
text = "\n".join(L); print(); print(text)
out = f"/home/sun4208/Geo4D/bench_out/traj_diversity{a.tag}"
open(out + ".txt", "w").write(text + "\n")
json.dump({"configs": a.configs, "seeds": a.seeds, "summary": summary, "raw": raw_out}, open(out + "_raw.json", "w"))
print(f"\n저장: {out}.txt / _raw.json")
print("TRAJ_DIVERSITY_DONE")
