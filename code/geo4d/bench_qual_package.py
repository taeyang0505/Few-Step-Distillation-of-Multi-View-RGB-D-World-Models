"""논문·발표용 정성 비교 패키지 (4종 x 샘플 여러 개) — 세 방법을 완전히 동일한 조건에서 생성
행: Teacher 25-step(EulerEDM, CFG) / 학습 없는 Teacher re-noise 3-step / DMD Student 3-step + per-view anchor
동일하게 고정: 태스크·샘플·조건 프레임(입력 RGB-D)·생성 시드·카메라 뷰·미래 프레임·원본 해상도(320x256)
① RGB 비교  ② Depth 비교(공통 컬러범위+컬러바, 미터 단위 명시)  ③ 3-step 시드 다양성(실제 3스텝 재생성)
④ 앵커 전/후 깊이 오차맵(teacher 포함, 공통 컬러범위+컬러바)
확대는 그리퍼 crop 패널에서만, 정수배 nearest 로만 한다(보간·리샘플 없음). PNG(무손실)로 저장.
출력: --out 디렉터리 + manifest.txt/json (샘플·시드·뷰·프레임·스텝수·수치 대응 기록)"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, os, json, time, random
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
ap.add_argument("--n_batches", type=int, default=8, help="렌더링할 샘플 수")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="/home/sun4208/Geo4D/bench_out/qual_package")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--views", nargs="+", default=["left"])
ap.add_argument("--frames", type=int, nargs="+", default=[1, 4, 7, 10], help="1-indexed 미래 프레임")
ap.add_argument("--configs", nargs="+", default=["T25"])   # 미사용(호환용)
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
import matplotlib; matplotlib.use("Agg")
from matplotlib import cm as _cm, font_manager as _fm
from PIL import Image, ImageDraw, ImageFont

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
    """조건 프레임(=프레임0) 대비 GT 3D 위치가 thr 이상 바뀐 픽셀 = 실제로 움직인 영역."""
    gt = unnormalize(vd[f"gt_video_{v}"][:, :3], -1, 2)                       # (T,3,H,W)
    cond = vd[f"cond_pointmap_{v}"]
    cond = unnormalize(cond.reshape(-1, *cond.shape[-3:])[-1], -1, 2)         # (3,H,W)
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


# ─────────────────────────── 렌더링 유틸 (원본 픽셀 보존) ───────────────────────────
# 모든 타일은 1:1 로 붙이고, 글자는 여백에만 그린다. 확대는 crop 패널에서만, 정수배 nearest 로만 한다.
TILE_W, TILE_H = None, None                       # 첫 생성 후 확정
CROP_ZOOM = 3
_font_path = _fm.findfont(_fm.FontProperties(family="DejaVu Sans"))
F_ROW = ImageFont.truetype(_font_path, 15)
F_SUB = ImageFont.truetype(_font_path, 12)
F_COL = ImageFont.truetype(_font_path, 14)
F_TINY = ImageFont.truetype(_font_path, 11)

MARGIN_L, MARGIN_T, GAP, PAD = 304, 34, 8, 14
BG = (255, 255, 255)


def unnorm_rgb(v):                                # (T,6,H,W) -> (T,H,W,3) uint8
    x = unnormalize(v[:, 3:], 0, 1)
    return (x * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)


def depth_of(v):
    return unnormalize(v[:, 2], -1, 2).cpu().numpy()


def colorize(d, lo, hi, cmap="viridis"):
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return (getattr(_cm, cmap)(x)[..., :3] * 255).astype(np.uint8)


def err_map(pred_d, gt_d, cap=0.5):
    m = (pred_d > 0) & (gt_d > 0)
    e = np.where(m, np.abs(pred_d - gt_d) / np.maximum(gt_d, 1e-3), 0.0)
    out = (_cm.magma(np.clip(e, 0, cap) / cap)[..., :3] * 255).astype(np.uint8)
    out[~m] = 40
    return out, e, m


def colorbar(h, lo, hi, cmap, unit, w=18):
    """세로 컬러바 + 눈금 텍스트를 담은 이미지 (여백에 그림)"""
    grad = np.linspace(1, 0, h)[:, None].repeat(w, 1)
    bar = (getattr(_cm, cmap)(grad)[..., :3] * 255).astype(np.uint8)
    im = Image.new("RGB", (w + 88, h), BG)
    im.paste(Image.fromarray(bar), (0, 0))
    dr = ImageDraw.Draw(im)
    for frac, val in [(0.0, hi), (0.5, (lo + hi) / 2), (1.0, lo)]:
        y = int(frac * (h - 1))
        dr.text((w + 5, min(max(y - 6, 0), h - 14)), f"{val:.2f} {unit}", fill=(30, 30, 30), font=F_TINY)
    return im


def compose(rows, col_titles, row_labels, crops=None, cbar=None, title=None, note=None):
    """rows: [[HxWx3 uint8, ...], ...] 원본 타일 / crops: 같은 구조의 확대 패널(이미 확대됨)"""
    nr, nc = len(rows), len(rows[0])
    th, tw = rows[0][0].shape[:2]
    ch, cw = (crops[0][0].shape[:2] if crops else (0, 0))
    block_h = nr * th + (nr - 1) * GAP
    crop_h = (nr * ch + (nr - 1) * GAP + 30) if crops else 0
    cb_w = (cbar.size[0] + 16) if cbar is not None else 0
    W = MARGIN_L + nc * tw + (nc - 1) * GAP + cb_w + PAD
    H = MARGIN_T + block_h + crop_h + PAD + (26 if title else 0) + (34 if note else 0)
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    y0 = PAD
    if title:
        dr.text((PAD, y0), title, fill=(0, 0, 0), font=F_ROW); y0 += 26
    for c, t in enumerate(col_titles):
        dr.text((MARGIN_L + c * (tw + GAP), y0 + 8), t, fill=(30, 30, 30), font=F_COL)
    y0 += MARGIN_T
    for r in range(nr):
        y = y0 + r * (th + GAP)
        name, sub = row_labels[r]
        dr.text((PAD, y + th // 2 - 16), name, fill=(0, 0, 0), font=F_ROW)
        if sub:
            dr.text((PAD, y + th // 2 + 3), sub, fill=(90, 90, 90), font=F_SUB)
        for c in range(nc):
            im.paste(Image.fromarray(rows[r][c]), (MARGIN_L + c * (tw + GAP), y))
    if cbar is not None:
        im.paste(cbar, (MARGIN_L + nc * (tw + GAP) + 8, y0))
    if crops:
        ytop = y0 + block_h + 18
        dr.text((PAD, ytop - 16), f"gripper crop ({CROP_ZOOM}x nearest-neighbour, no interpolation)",
                fill=(60, 60, 60), font=F_SUB)
        for r in range(nr):
            y = ytop + 14 + r * (ch + GAP)
            dr.text((PAD, y + ch // 2 - 8), row_labels[r][0], fill=(0, 0, 0), font=F_SUB)
            for c in range(nc):
                im.paste(Image.fromarray(crops[r][c]), (MARGIN_L + c * (cw + GAP), y))
    if note:
        dr.text((PAD, H - 30), note, fill=(60, 60, 60), font=F_SUB)
    return im


def zoom(a_):
    return np.kron(a_, np.ones((CROP_ZOOM, CROP_ZOOM, 1), dtype=a_.dtype)) if a_.ndim == 3 else \
           np.kron(a_, np.ones((CROP_ZOOM, CROP_ZOOM), dtype=a_.dtype))


# ─────────────────────────── 생성 ───────────────────────────
plain_ref = plain


def gen(who, steps, samp, anchor, ib, seed):
    """who: teacher|student, samp: euler|renoise, anchor: bool"""
    global cur_who
    if cur_who != who:
        model.model = plain_ref
        plain_ref.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
        cur_who = who
    model.sampler = RenoiseSampler(sigmas_for_steps(steps)) if samp == "renoise" else euler
    if samp == "euler":
        model.sampler.num_steps = steps
    enable_cond_anchor(model, per_view=True) if anchor else disable_cond_anchor(model)
    set_fast(a.fast and who == "student")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        return model.log_images(ib)["video_dict"]


cur_who = None
os.makedirs(a.out, exist_ok=True)
FR = [f - 1 for f in a.frames]                                  # 0-indexed
DIV_FR = [f - 1 for f in a.frames[1:]] if len(a.frames) > 3 else FR[-3:]
manifest = []
print(f"[3/3] 정성 패키지 생성 — 샘플 {len(batches)}개 x 뷰 {a.views}", flush=True)

for bi, (ib, lab) in enumerate(zip(batches, labels)):
    t_start = time.time()
    runs = {"T25": gen("teacher", 25, "euler", False, ib, 0),
            "T3r": gen("teacher", 3, "renoise", False, ib, 0),
            "S3_raw": gen("student", 3, "renoise", False, ib, 0),
            "S3b": gen("student", 3, "renoise", True, ib, 0)}
    div = {"T3r": {0: runs["T3r"]}, "S3b": {0: runs["S3b"]}}
    for s in a.seeds[1:]:
        div["T3r"][s] = gen("teacher", 3, "renoise", False, ib, s)
        div["S3b"][s] = gen("student", 3, "renoise", True, ib, s)

    for v in a.views:
        gt = runs["T25"][f"gt_video_{v}"]
        gt_rgb = unnorm_rgb(gt); gt_d = depth_of(gt)
        H, W = gt_d.shape[-2:]
        valid = gt_d > 0
        lo, hi = (float(np.percentile(gt_d[valid], 2)), float(np.percentile(gt_d[valid], 98))) if valid.any() else (0., 1.)

        # 그리퍼 crop 박스: GT label(그리퍼 id) 우선, 없으면 움직임 마스크
        L = lab[v]
        mg = torch.isin(L, torch.tensor(GRIPPER_IDS)).numpy()
        src = mg if mg.sum() > 200 else (np.abs(gt_d - gt_d[0]) > 0.02)
        ys, xs = np.nonzero(src[FR[1]] if src[FR[1]].sum() > 50 else src.any(0))
        cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (H // 2, W // 2)
        chh, cww = 86, 106
        y0c = int(np.clip(cy - chh // 2, 0, H - chh)); x0c = int(np.clip(cx - cww // 2, 0, W - cww))
        box = (y0c, x0c, y0c + chh, x0c + cww)

        rgbs = {k: unnorm_rgb(r[f"sampled_video_{v}"]) for k, r in runs.items()}
        deps = {k: depth_of(r[f"sampled_video_{v}"]) for k, r in runs.items()}
        absrel = {}
        for k in runs:
            m = (deps[k] > 0) & valid
            absrel[k] = float(np.mean(np.abs(deps[k][m] - gt_d[m]) / gt_d[m])) if m.any() else float("nan")

        def crop(img, t):
            y0_, x0_, y1_, x1_ = box
            return zoom(img[t][y0_:y1_, x0_:x1_])

        ROWS3 = [("Teacher (Geo4D)", "EulerEDM, 25 steps, CFG"),
                 ("Teacher, training-free", "re-noising, 3 steps"),
                 ("DMD Student (ours)", "re-noising, 3 steps + per-view anchor")]
        keys3 = ["T25", "T3r", "S3b"]
        cols = [f"t+{f}" for f in a.frames]
        tag = f"{v}_s{bi}"
        stem = f"{a.out}/qual"

        # ① RGB
        im = compose([[rgbs[k][t] for t in FR] for k in keys3], cols, ROWS3,
                     crops=[[crop(rgbs[k], t) for t in FR] for k in keys3],
                     title=f"RGB comparison — sample {bi}, view {v}, seed 0, native 320x256 px",
                     note="Same sample, same conditioning frame, same generation seed, same view and frames for all three rows.")
        p1 = f"{stem}_rgb_{tag}.png"; im.save(p1)

        # ② Depth (공통 컬러범위 + 컬러바)
        cb = colorbar(3 * 256 + 2 * GAP, lo, hi, "viridis", "m")
        im = compose([[colorize(deps[k][t], lo, hi) for t in FR] for k in keys3], cols, ROWS3,
                     crops=[[crop(colorize(deps[k], lo, hi), t) for t in FR] for k in keys3], cbar=cb,
                     title=f"Depth comparison — sample {bi}, view {v}, seed 0, shared colour range",
                     note=f"Colour maps depth z in metres over [{lo:.2f}, {hi:.2f}] (2nd-98th percentile of the ground truth), identical for all rows.")
        p2 = f"{stem}_depth_{tag}.png"; im.save(p2)

        # ③ 3-step 시드 다양성 (실제 3스텝으로 재생성한 것)
        div_rows, div_labels = [], []
        for who_k, who_name in [("T3r", "Teacher training-free 3-step"), ("S3b", "DMD Student 3-step + anchor")]:
            for s in a.seeds:
                r = unnorm_rgb(div[who_k][s][f"sampled_video_{v}"])
                div_rows.append([r[t] for t in DIV_FR])
                div_labels.append((f"{who_name}", f"seed {s}"))
        std = {k: float(np.stack([depth_of(div[k][s][f"sampled_video_{v}"]) for s in a.seeds]).std(0).mean()) for k in ["T3r", "S3b"]}
        im = compose(div_rows, [f"t+{f}" for f in a.frames[1:]] if len(a.frames) > 3 else cols[-3:], div_labels,
                     title=f"Seed diversity at 3 steps — sample {bi}, view {v} (regenerated at 3 steps, NOT 1 step)",
                     note=f"Per-pixel depth std across seeds {a.seeds}: training-free {std['T3r']:.4f} vs DMD student {std['S3b']:.4f} (this sample).")
        p3 = f"{stem}_diversity3_{tag}.png"; im.save(p3)

        # ④ 앵커 전/후 깊이 오차맵
        e_rows, e_crops, e_stat = [], [], {}
        for k in ["T25", "S3_raw", "S3b"]:
            emap, e, m = err_map(deps[k], gt_d)
            e_stat[k] = float(e[m].mean())
            e_rows.append([emap[t] for t in FR]); e_crops.append([crop(emap, t) for t in FR])
        cb = colorbar(3 * 256 + 2 * GAP, 0.0, 0.5, "magma", "rel.")
        im = compose(e_rows, cols,
                     [("Teacher (Geo4D)", f"25 steps | AbsRel {absrel['T25']:.3f}"),
                      ("DMD Student, no anchor", f"3 steps | AbsRel {absrel['S3_raw']:.3f}"),
                      ("DMD Student + anchor", f"3 steps | AbsRel {absrel['S3b']:.3f}")],
                     crops=e_crops, cbar=cb,
                     title=f"Depth error before/after the per-view anchor — sample {bi}, view {v}, seed 0",
                     note="Error = |z_pred - z_gt| / z_gt, clipped at 0.50; grey = no valid depth. Same colour range for all rows.")
        p4 = f"{stem}_anchor_err_{tag}.png"; im.save(p4)

        manifest.append({"sample": bi, "view": v, "seed": 0, "seeds_diversity": a.seeds,
                         "frames_1indexed": a.frames, "cams": lab["cams"], "crop_box_yxyx": list(box),
                         "depth_range_m": [lo, hi], "AbsRel": absrel, "err_mean": e_stat,
                         "diversity_depth_std": std, "files": [os.path.basename(p) for p in (p1, p2, p3, p4)],
                         "task": os.environ.get("GEO4D_TEACHER_DIR", "apple").rstrip("/").split("/")[-1],
                         "student_ckpt": os.path.basename(a.student_ckpt)})
        print(f"  [샘플 {bi}/{v}] AbsRel T25 {absrel['T25']:.3f} | S3 raw {absrel['S3_raw']:.3f} -> anchor {absrel['S3b']:.3f} | "
              f"다양성 T3r {std['T3r']:.4f} vs S3b {std['S3b']:.4f} | {time.time()-t_start:.0f}s", flush=True)

# ─────────────────────────── 매니페스트 ───────────────────────────
rank = sorted([m for m in manifest if m["view"] == a.views[0]], key=lambda m: m["AbsRel"]["S3b"])
med = rank[len(rank) // 2 - 1:len(rank) // 2 + 2]
lines = ["=== 정성 패키지 매니페스트 ===",
         f"student ckpt: {a.student_ckpt}", f"data_seed {a.data_seed}, 샘플 {len(batches)}개, 뷰 {a.views}, 프레임 {a.frames}",
         "선정 규칙(사후, 임의 선택 없음): student(S3b) AbsRel 중앙값 부근 3개 = 일반 사례, 최댓값 1개 = 실패 사례", "",
         f"일반 사례 3개: samples {[m['sample'] for m in med]}",
         f"실패 사례 1개: sample {rank[-1]['sample']} (AbsRel {rank[-1]['AbsRel']['S3b']:.3f})", "",
         "sample | view | AbsRel T25 | S3 raw | S3b(anchor) | 다양성 T3r | 다양성 S3b | 파일"]
for m in manifest:
    lines.append(f"{m['sample']:>6} | {m['view']:>4} | {m['AbsRel']['T25']:.4f} | {m['AbsRel']['S3_raw']:.4f} | "
                 f"{m['AbsRel']['S3b']:.4f} | {m['diversity_depth_std']['T3r']:.4f} | {m['diversity_depth_std']['S3b']:.4f} | {', '.join(m['files'])}")
lines += ["", "주의: 위 수치는 각 샘플 1개에서 잰 값이라 논문의 20/60샘플 평균과 다르다. 이미지 옆에 붙일 때는 이 샘플 값을 쓸 것.",
          "생성 명령: python notebooks/bench_qual_package.py --n_batches 8 --views left right --fast"]
open(f"{a.out}/manifest.txt", "w").write("\n".join(lines) + "\n")
json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
print("\n".join(lines))
print(f"\n저장: {a.out}/  (PNG {4*len(manifest)}장 + manifest.txt/json)")
print("QUAL_PACKAGE_DONE")
