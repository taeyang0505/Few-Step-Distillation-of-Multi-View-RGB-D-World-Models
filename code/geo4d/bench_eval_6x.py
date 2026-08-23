"""통합 평가 (6-3 이후 표준 프로토콜): 고정 데이터 시드, teacher/student 여러 설정을 같은 샘플에서 paired 비교
설정 문자열: T25 / T4 / T1 (teacher EulerEDM) / S3 / S1 (student 재노이징) / 접미사 a=왼쪽 스케일 앵커, b=뷰별 스케일 앵커, c=뷰별 robust affine 앵커
지표: PSNR, AbsRel(뷰별), LPIPS, CV-Chamfer, 시간 + 앵커 s 통계. paired vs T25 (Wilcoxon)
사용: python notebooks/bench_eval_6x.py --student_ckpt X --configs T25 T4 S3 S3a --tag _6a_anchor"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, json, random, time
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
import lpips as lpips_lib
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps, enable_cond_anchor, disable_cond_anchor, cond_anchor_scale_right, set_extr_idx

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--configs", nargs="+", default=["T25", "T4", "T1", "S3", "S3a", "S1", "S1a"])
ap.add_argument("--n_batches", type=int, default=10)
ap.add_argument("--data_seed", type=int, default=1234)
ap.add_argument("--tag", default="")
ap.add_argument("--fix_mask", type=int, default=1, help="1=오른쪽 GT의 변환된 무효 픽셀 제외 (발견 10), 0=이전 방식")
ap.add_argument("--extr_idx", default="0,0", help="오른쪽 앵커 변환용 외부 파라미터 프레임 인덱스 (참조,조건). 카메라가 움직이면 1,0")
ap.add_argument("--swap", choices=["none", "dm2_out", "dm1_out", "backbone"], default="none",
                help="진단용 하이브리드 (wrappers.py: 두 뷰는 백본 공유, output_blocks만 뷰별): dm2_out=뷰2 디코더만 teacher, dm1_out=뷰1 디코더만 teacher, backbone=공유 백본(input/middle/time/label/out)만 teacher")
a = ap.parse_args()
set_extr_idx(*a.extr_idx.split(","))
print("extr_idx:", a.extr_idx, flush=True)
N_SUB = 4096


def parse(cfg_str):
    anchor = cfg_str[-1] if cfg_str[-1] in "abc" else ""
    core = cfg_str[:-1] if anchor else cfg_str
    who = "teacher" if core[0] == "T" else "student"
    return who, int(core[1:]), anchor


def unnormalize(x, mn, mx):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)


def psnr_of(o, v):
    p = unnormalize(o["video_dict"][f"sampled_video_{v}"][:, 3:], 0, 1)
    g = unnormalize(o["video_dict"][f"gt_video_{v}"][:, 3:], 0, 1)
    return (10 * torch.log10(1. / torch.mean((p - g) ** 2))).item(), p, g


def gt_valid_mask(o, v):
    """GT 유효 마스크. 오른쪽 뷰: 데이터셋이 무효(xyz=0) 픽셀까지 좌표 변환해 xyz=t(이동벡터)로 채움 → 제외 (발견 10)"""
    gt = unnormalize(o["video_dict"][f"gt_video_{v}"][:, :3], -1, 2)
    m = gt[:, 2] > 0
    if v == "right" and a.fix_mask:
        ex = o["video_dict"].get("extra", {})
        if "cam_extr" in ex and "cam_extr_right" in ex:
            E1 = ex["cam_extr"].reshape(-1, 4, 4)[0].float(); E2 = ex["cam_extr_right"].reshape(-1, 4, 4)[0].float()
            t = (torch.linalg.inv(E1) @ E2)[:3, 3].to(gt.device).view(1, 3, 1, 1)
            garbage = (gt - t).abs().amax(dim=1) < 2e-3
            m = m & ~garbage
    return m


def absrel_of(o, v):
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


def ratio_err(o):
    """뷰 간 깊이 비 r=mean z_R/mean z_L 의 GT 대비 |log| 오차 (프레임 평균) — 공통 마스크(pred>0 & GT 유효)"""
    mL = gt_valid_mask(o, "left"); mR = gt_valid_mask(o, "right")
    def r_of(key):
        zl = unnormalize(o["video_dict"][f"{key}_left"][:, 2], -1, 2); zr = unnormalize(o["video_dict"][f"{key}_right"][:, 2], -1, 2)
        ml, mr = (zl > 0) & mL, (zr > 0) & mR
        ml_ = (zl * ml).sum((1, 2)) / ml.sum((1, 2)).clamp_min(1); mr_ = (zr * mr).sum((1, 2)) / mr.sum((1, 2)).clamp_min(1)
        return mr_ / ml_.clamp_min(1e-3)
    return torch.abs(torch.log(r_of("sampled_video").clamp_min(1e-3)) - torch.log(r_of("gt_video").clamp_min(1e-3))).mean().item()


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
student_sd = torch.load(a.student_ckpt, map_location="cpu")["student"] if any(parse(c)[0] == "student" for c in a.configs) else None
if student_sd is not None and a.swap != "none":
    def is_target(k):
        if a.swap == "dm2_out":
            return k.startswith("diffusion_model_2.output_blocks.")
        if a.swap == "dm1_out":
            return k.startswith("diffusion_model.output_blocks.")
        # backbone: output_blocks 이외 전부 (두 prefix 모두 — 공유 텐서라 둘 다 맞춰야 로드 시 일관)
        return (k.startswith("diffusion_model.") or k.startswith("diffusion_model_2.")) and ".output_blocks." not in k
    n_sw = 0
    for k in list(student_sd.keys()):
        if is_target(k) and k in teacher_sd:
            student_sd[k] = teacher_sd[k].clone(); n_sw += 1
    print(f"[swap={a.swap}] {n_sw}개 텐서를 teacher 가중치로 교체", flush=True)
lpips_net = lpips_lib.LPIPS(net="alex").cuda().eval()

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
with torch.no_grad():
    model.log_images(batches[0])

print("[3/3] 측정:", a.configs, flush=True)
raw = {}
cur = None
for name in a.configs:
    who, steps, anchor = parse(name)
    if who != cur:
        model.model.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
        cur = who
    if who == "teacher":
        model.sampler = euler; model.sampler.num_steps = steps
    else:
        model.sampler = RenoiseSampler(sigmas_for_steps(steps))
    if anchor:
        enable_cond_anchor(model, per_view=(anchor == "b"), affine=(anchor == "c"))
    else:
        disable_cond_anchor(model)
    rec = {"view": [], "cv": [], "sr": [], "s": [], "time": 0.0}
    t0 = time.time()
    for bi, ib in enumerate(batches):
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            o = model.log_images(ib)
        rec["s"].append(o["video_dict"].get("anchor_scale", 1.0))
        if bi == 0 and name == a.configs[0]:
            gR = unnormalize(o["video_dict"]["gt_video_right"][:, 2], -1, 2) > 0
            print(f"  [마스크] 오른쪽 GT z>0 픽셀 중 제외된 가짜 유효 픽셀 비율: {1 - gt_valid_mask(o, 'right').float().sum().item() / gR.float().sum().item():.3f} (fix_mask={a.fix_mask})", flush=True)
        if anchor == "c" and bi < 3:
            print(f"  [affine] batch {bi}: " + ", ".join(f"{v} a={ab[0]:.3f} b={ab[1]:+.3f}" for v, ab in o["video_dict"]["anchor_affine"].items()), flush=True)
        if bi == 0 and anchor == "b":
            from geo4d_fewstep import cond_anchor_scale_right
            sR, cd = cond_anchor_scale_right(o["video_dict"])
            if cd is not None:
                g0 = unnormalize(o["video_dict"]["gt_video_right"][0, 2], -1, 2); mm = (cd > 0) & (g0 > 0)
                print(f"  [검증] 변환된 오른쪽 조건 vs GT 오른쪽 프레임0 AbsRel {torch.mean(torch.abs(cd[mm]-g0[mm])/g0[mm]).item():.4f} (0.01대면 변환 정상) | s_R {sR}", flush=True)
        for v in ["left", "right"]:
            ps, p, g = psnr_of(o, v)
            with torch.no_grad():
                lp = lpips_net(p * 2 - 1, g * 2 - 1).mean().item()
            rec["view"].append({"batch": bi, "view": v, "PSNR": ps, "AbsRel": absrel_of(o, v), "LPIPS": lp})
        rec["cv"].append(crossview(o)); rec["sr"].append(ratio_err(o))
    rec["time"] = (time.time() - t0) / len(batches)
    raw[name] = rec
    mv = lambda k, vv=None: np.mean([r[k] for r in rec["view"] if vv is None or r["view"] == vv])
    print(f"[{name:>4}] {rec['time']:.1f}s | PSNR {mv('PSNR'):.2f} | AbsRel {mv('AbsRel'):.4f} (L {mv('AbsRel','left'):.4f} / R {mv('AbsRel','right'):.4f}) "
          f"| LPIPS {mv('LPIPS'):.4f} | CV {np.mean(rec['cv']):.4f} | 뷰비오차 {np.mean(rec['sr']):.4f} | s {np.mean(rec['s']):.3f}±{np.std(rec['s']):.3f}", flush=True)
disable_cond_anchor(model)

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
ref = a.configs[0]
L = [f"=== 통합 평가 (data_seed {a.data_seed}, 배치 {a.n_batches}=뷰 {2*a.n_batches}) student={a.student_ckpt.split('/')[-1]} swap={a.swap} ===", "",
     "설정 | s/생성 | PSNR^ | AbsRel v | AbsRel L | AbsRel R | LPIPS v | CV v | 뷰비오차 v | 앵커 s"]
for name in a.configs:
    r = raw[name]; mv = lambda k, vv=None: np.mean([x[k] for x in r["view"] if vv is None or x["view"] == vv])
    L.append(f"{name:>4} | {r['time']:5.1f} | {mv('PSNR'):5.2f} | {mv('AbsRel'):.4f} | {mv('AbsRel','left'):.4f} | {mv('AbsRel','right'):.4f} | {mv('LPIPS'):.4f} | {np.mean(r['cv']):.4f} | {np.mean(r['sr']):.4f} | {np.mean(r['s']):.3f}")
L += ["", f"[paired X − {ref}] 지표 | 설정 | 평균차 ± std | X 우수 비율 | Wilcoxon p"]
sign = {"PSNR": 1, "AbsRel": -1, "AbsRel_L": -1, "AbsRel_R": -1, "LPIPS": -1, "CV": -1, "SR": -1}
for k in ["PSNR", "AbsRel", "AbsRel_L", "AbsRel_R", "LPIPS", "CV", "SR"]:
    for name in a.configs[1:]:
        def arr(n):
            if k == "CV":
                return np.array(raw[n]["cv"])
            if k == "SR":
                return np.array(raw[n]["sr"])
            if k.startswith("AbsRel_"):
                vv = "left" if k.endswith("L") else "right"
                return np.array([x["AbsRel"] for x in raw[n]["view"] if x["view"] == vv])
            return np.array([x[k] for x in raw[n]["view"]])
        xs, ts = arr(name), arr(ref)
        d = xs - ts
        wins = np.mean(np.sign(d) == sign[k])
        p = wilcoxon(xs, ts).pvalue if (wilcoxon and len(d) >= 5 and np.any(d != 0)) else float("nan")
        L.append(f"{k:>8} | {name:>4} | {d.mean():+.4f} ± {d.std():.4f} | {100*wins:5.1f}% | {p:.3f}")
text = "\n".join(L)
print(); print(text)
with open(f"/home/sun4208/Geo4D/bench_out/eval_6x{a.tag}.txt", "w") as f:
    f.write(text + "\n")
with open(f"/home/sun4208/Geo4D/bench_out/eval_6x{a.tag}_raw.json", "w") as f:
    json.dump(raw, f)
print(f"\n저장: ~/Geo4D/bench_out/eval_6x{a.tag}.txt")
