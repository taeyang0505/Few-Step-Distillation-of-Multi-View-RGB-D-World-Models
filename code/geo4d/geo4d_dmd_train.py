"""Step 6-3/6-4: Geo4D DMD 증류. --cv_weight 0 = 6a 바닐라, >0 = 6b (c) 뷰 간 스케일 consistency loss(실패), --anchor_weight >0 = 6-4(d) 자기 앵커 loss
6-4(c): L_cv = (log r_student − log r_teacher)², r = mean depth_R / mean depth_L (참조 좌표계, 스케일 불변).
  생성 latent의 프레임 cv_frames개만 VAE 포인트맵 디코더로 grad 디코드. r_teacher는 ODE 쌍 z를 디코드해 샘플별·프레임별 사전계산.

역할: generator = 6-2 v2 student(학습) / real_score = teacher 동결(CFG 가이더 포함)
      / fake_score = teacher 사본(학습, student 분포 추적)
손실: DMD (arXiv:2311.18828 eq.7-8, Self-Forcing model/dmd.py 이식) — σ-공간(EDM) 버전
  grad = (pred_fake - pred_real) / mean|x0 - pred_real| ;  L_gen = 0.5·MSE(x0, (x0 - grad).detach())
  L_critic = EDM 가중 MSE(fake_score(x0+σε), x0)  (teacher 학습 loss와 같은 가중, σ_data=1)
생성: DMD2 backward simulation (arXiv:2405.14867 §4.5) — k∈{0..n-1} 무작위, k스텝까지 no_grad 재노이징, k번째 예측만 grad
Step 5 레시피: 전부 bf16 / AdamW8bit(paged 옵션) / 컨디션 사전계산 후 conditioner·VAE를 GPU에서 제거 / 메모리 프로브
"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse
import copy
import glob
import json
import math
import os
import random
import time
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, DEFAULT_SIGMAS

p = argparse.ArgumentParser()
p.add_argument("--pairs_dir", default="/home/sun4208/Geo4D/ode_pairs_v2")
p.add_argument("--init_ckpt", default="/home/sun4208/Geo4D/ode_init_geo4d_v2.pt", help="generator 초기화(v2 student). 'teacher'면 teacher 가중치")
p.add_argument("--init_fake", default="", help="fake_score 초기화 ckpt (resume용, 기본 teacher 사본)")
p.add_argument("--resume", action="store_true", help="out_dir의 dmd_gen.pt/dmd_fake.pt가 있으면 그 step부터 이어서 학습 (OOM 등으로 죽었을 때; optimizer 모멘트는 재시작)")
p.add_argument("--out_dir", default="/home/sun4208/Geo4D/dmd_6a")
p.add_argument("--max_steps", type=int, default=2000)
p.add_argument("--gen_every", type=int, default=5, help="dfake_gen_update_ratio: critic N번당 generator 1번")
p.add_argument("--lr_gen", type=float, default=2e-6)
p.add_argument("--lr_critic", type=float, default=4e-7)
p.add_argument("--beta2", type=float, default=0.999)
p.add_argument("--max_grad_norm", type=float, default=10.0)
p.add_argument("--real_guidance", type=float, default=-1, help="<0이면 teacher 가이더(LinearPrediction 1.0→2.5) 그대로")
p.add_argument("--sigma_mode", choices=["edm", "loguniform"], default="edm", help="DMD/critic 노이즈 σ 샘플링")
p.add_argument("--sigma_min", type=float, default=0.02)
p.add_argument("--sigma_max", type=float, default=700.0)
p.add_argument("--n_student_steps", type=int, default=3)
p.add_argument("--max_cond_samples", type=int, default=0, help="컨디션 사전계산 샘플 수 제한(0=전부)")
p.add_argument("--paged_optim", action="store_true", help="bnb PagedAdamW8bit (VRAM 부족 시 상태를 CPU로 페이징)")
p.add_argument("--log_every", type=int, default=10)
p.add_argument("--diag_every", type=int, default=100)
p.add_argument("--save_every", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--cv_weight", type=float, default=0.0, help="6-4(c) 뷰 간 consistency 상대 강도 β: x0 그래디언트 크기 기준 DMD 대비 배율 (0=6a)")
p.add_argument("--anchor_weight", type=float, default=0.0, help="6-4(d) 자기 앵커 loss 상대 강도 β: 프레임0 깊이를 조건 프레임 깊이(입력, GT 아님)에 맞춤. 0=끔")
p.add_argument("--anchor_exclude_robot", action="store_true", help="앵커 loss에서 로봇(그리퍼) 픽셀 제외: 예측 프레임0은 조건 프레임보다 5스텝 뒤라 로봇만 움직임 → 정적 장면만 맞춤")
p.add_argument("--anchor_frames", default="0", help="앵커 loss를 거는 예측 프레임 인덱스(콤마). 조건 프레임은 클립 프레임0이므로 기본 0")
p.add_argument("--cv_target", choices=["teacher", "gt"], default="teacher", help="뷰 비 표적: teacher(ODE 쌍 z 디코드) 또는 gt(데이터셋 GT 포인트맵)")
p.add_argument("--cv_frames", type=int, default=1, help="consistency loss에 grad 디코드할 프레임 수")
p.add_argument("--keep_steps", default="1000,1600", help="이 스텝들의 generator ckpt를 별도 보존")
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
torch.manual_seed(args.seed); random.seed(args.seed)


def mem(tag):
    a = torch.cuda.memory_allocated() / 2**30
    m = torch.cuda.max_memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    print(f"[MEM {tag}] alloc {a:.2f}GB | max {m:.2f}GB | reserved {r:.2f}GB", flush=True)


# ───────────────────────── 1. 모델 로드 ─────────────────────────
print("[1/5] teacher 로드", flush=True)
output_dir = os.environ.get("GEO4D_TEACHER_DIR", "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple")   # 태스크 전환: 환경변수
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
teacher = model.model                      # OpenAIWrapper(diffusion_model, diffusion_model_2)
teacher.requires_grad_(False).to(torch.bfloat16).eval()
denoiser = model.denoiser                  # VScalingWithEDMcNoise (연속 σ)
guider = model.sampler.guider              # LinearPredictionGuider(num_frames=10, 1.0→2.5)
if args.real_guidance >= 0:
    guider.scale = torch.full_like(guider.scale, args.real_guidance)
n_params = sum(x.numel() for x in teacher.parameters()) / 1e9
print(f"UNet 파라미터 {n_params:.2f}B", flush=True)
mem("teacher bf16")

# ───────────────────────── 2. 컨디션 사전계산 → conditioner/VAE 제거 ─────────────────────────
print("[2/5] 컨디션 사전계산", flush=True)
with open(os.path.join(args.pairs_dir, "meta.json")) as f:
    meta = json.load(f)
pair_files = sorted(glob.glob(os.path.join(args.pairs_dir, "pair_*.pt")))
idx_set = sorted({int(os.path.basename(f).split("_")[1]) for f in pair_files})
if args.max_cond_samples > 0:
    idx_set = idx_set[:args.max_cond_samples]
    pair_files = [f for f in pair_files if int(os.path.basename(f).split("_")[1]) in set(idx_set)]
print(f"ODE 쌍 {len(pair_files)}개 / 샘플 {len(idx_set)}개", flush=True)

cfg.task = OmegaConf.load(os.environ.get("GEO4D_TASK_YAML", "/home/sun4208/4dgen/config/task/inference.yaml"))
dataset = hydra.utils.instantiate(cfg.task.dataset)

conds = {}
gt_ratio = {}
anchor_depth = {}
t0 = time.time()
with torch.no_grad():
    for n, idx in enumerate(idx_set):
        b_raw = default_collate([dataset[idx]])
        b_raw = dict_apply(b_raw, lambda x: x.to("cuda", non_blocking=True))
        b_raw["num_video_frames"] = b_raw["pointmap"].shape[1]
        batch = {k: v[0:1] for k, v in b_raw.items() if k != "num_video_frames" and torch.is_tensor(v)}
        batch.update({k: v for k, v in b_raw.items() if not torch.is_tensor(v)})
        batch["num_video_frames"] = b_raw["num_video_frames"]
        c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch)
        ami = {"num_video_frames": batch["num_video_frames"],
               "image_only_indicator": batch["image_only_indicator"].cpu()}
        conds[idx] = ({k: v.cpu() for k, v in c.items()}, {k: v.cpu() for k, v in uc.items()}, ami)
        # GT 뷰 간 깊이 비 r_gt(t) = mean z_R / mean z_L (포인트맵 채널 2, unnormalize(-1,2), z>0 마스크)
        zl = torch.clamp(((batch["pointmap"][0][:, 2] + 1.) / 2.) * 3.0 - 1.0, -1.0, 2.0)
        zr = torch.clamp(((batch["pointmap_right"][0][:, 2] + 1.) / 2.) * 3.0 - 1.0, -1.0, 2.0)
        ml, mr = (zl > 0).float(), (zr > 0).float()
        gt_ratio[idx] = ((zr * mr).sum((1, 2)) / mr.sum((1, 2)).clamp_min(1) / ((zl * ml).sum((1, 2)) / ml.sum((1, 2)).clamp_min(1)).clamp_min(1e-3)).cpu()
        # 자기 앵커용 조건 프레임 깊이 (입력 정보만 사용, GT 아님). 왼쪽: 조건 포인트맵 z. 오른쪽: 조건 포인트맵을 참조 프레임으로 변환한 z
        def _un(x): return torch.clamp(((x + 1.) / 2.) * 3.0 - 1.0, -1.0, 2.0)
        cL = batch["cond_pointmaps_without_noise"][0]; cL = cL.reshape(-1, *cL.shape[-3:])[-1]
        cR = batch["cond_pointmaps_without_noise_right"][0]; cR = cR.reshape(-1, *cR.shape[-3:])[-1]
        E1 = batch["cam_extr"].reshape(-1, 4, 4)[0].float(); E2 = batch["cam_extr_right"].reshape(-1, 4, 4)[0].float()
        T = torch.linalg.inv(E1) @ E2
        xyzR = _un(cR[:3]); validR = xyzR[2] > 0
        hom = torch.cat([xyzR.reshape(3, -1), torch.ones(1, xyzR.shape[1] * xyzR.shape[2], device=xyzR.device)], 0)
        zR_ref = (T @ hom)[2].reshape(xyzR.shape[1:])
        dL = _un(cL[2]); validL = dL > 0
        if args.anchor_exclude_robot:   # 데이터셋의 팽창된 로봇 마스크(H/8) → 예측 프레임0 기준, 8배 업샘플해 제외
            up = lambda m: torch.nn.functional.interpolate(m.reshape(1, 1, *m.shape[-2:]).float(), scale_factor=8, mode="nearest")[0, 0] > 0.5
            validL = validL & ~up(batch["masks"][0][0]).to(validL.device); validR = validR & ~up(batch["masks_right"][0][0]).to(validR.device)
        anchor_depth[idx] = (torch.where(validL, dL, torch.zeros_like(dL)).cpu(), torch.where(validR & (zR_ref > 0), zR_ref, torch.zeros_like(zR_ref)).cpu())
        if n == 0:
            mv = torch.cat([batch["pointmap"][0], batch["pointmap_right"][0]], dim=0)
            BT, C, H, W = mv.shape
            LATENT_SHAPE = (BT, 8, H // 8, W // 8)
            print("latent shape:", LATENT_SHAPE, "| cond keys:", {k: tuple(v.shape) for k, v in c.items()}, flush=True)
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(idx_set)} ({time.time()-t0:.0f}s)", flush=True)
print(f"컨디션 {len(conds)}개 사전계산 완료 ({time.time()-t0:.0f}s)", flush=True)
allg = torch.stack(list(gt_ratio.values()))
print(f"GT 뷰 비 r_gt 평균 {allg.mean():.3f} (프레임 std {allg.std(1).mean():.3f}, 샘플 std {allg.mean(1).std():.3f})", flush=True)

# GPU에서 동결 모듈 제거 (Step 5 교훈: 동결 모듈 잔류가 최대 복병)
model.conditioner.cpu()
model.first_stage_color_model.cpu()
if args.cv_weight > 0 or args.anchor_weight > 0:
    model.first_stage_pointmap_model.requires_grad_(False).eval()   # GPU 유지 (fp32, 디코더 activation에 grad 흐름)
else:
    model.first_stage_pointmap_model.cpu()
del dataset
torch.cuda.empty_cache()
mem("conditioner/VAE 제거 후")


def to_cuda(idx):
    c, uc, ami = conds[idx]
    return ({k: v.cuda(non_blocking=True) for k, v in c.items()},
            {k: v.cuda(non_blocking=True) for k, v in uc.items()},
            {"num_video_frames": ami["num_video_frames"],
             "image_only_indicator": ami["image_only_indicator"].cuda()})


# ───────────────────────── 3. generator / fake_score 구성 ─────────────────────────
print("[3/5] generator(v2 student) + fake_score(teacher 사본) 구성", flush=True)
fake = copy.deepcopy(teacher)
START_STEP = 1
if args.resume and os.path.exists(os.path.join(args.out_dir, "dmd_gen.pt")) and os.path.exists(os.path.join(args.out_dir, "dmd_fake.pt")):
    _g = torch.load(os.path.join(args.out_dir, "dmd_gen.pt"), map_location="cpu")
    if _g.get("step", 0) >= args.max_steps:
        print(f"[resume] 이미 {_g.get('step')} step 완료 — 학습 생략", flush=True); sys.exit(0)
    args.init_ckpt = os.path.join(args.out_dir, "dmd_gen.pt"); args.init_fake = os.path.join(args.out_dir, "dmd_fake.pt")
    START_STEP = int(_g.get("step", 0)) + 1
    print(f"[resume] {args.out_dir} step {START_STEP - 1}부터 이어서 학습 (optimizer 상태는 재시작)", flush=True)
if args.init_fake:
    sd = torch.load(args.init_fake, map_location="cpu")
    fake.load_state_dict(sd["fake"] if "fake" in sd else sd["student"], strict=False)
    print(f"fake_score 초기화: {args.init_fake}", flush=True)
fake.requires_grad_(True).train()

gen = copy.deepcopy(teacher)
if args.init_ckpt != "teacher":
    sd = torch.load(args.init_ckpt, map_location="cpu")
    missing, unexpected = gen.load_state_dict(sd["student"], strict=False)
    print(f"generator 초기화: {args.init_ckpt} (step {sd.get('step')}, missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    del sd
gen.requires_grad_(True).train()
mem("3벌 구성")

import bitsandbytes as bnb
Opt = bnb.optim.PagedAdamW8bit if args.paged_optim else bnb.optim.AdamW8bit
opt_g = Opt([q for q in gen.parameters() if q.requires_grad], lr=args.lr_gen, betas=(0.0, args.beta2), weight_decay=0.0)
opt_c = Opt([q for q in fake.parameters() if q.requires_grad], lr=args.lr_critic, betas=(0.0, args.beta2), weight_decay=0.0)

sigmas = DEFAULT_SIGMAS[:args.n_student_steps] if args.n_student_steps <= 3 else DEFAULT_SIGMAS
sampler = RenoiseSampler(sigmas)
print("student σ 스케줄:", [f"{s:.1f}" for s in sigmas], flush=True)



# ───────────────────────── 6-4(c) 뷰 간 스케일 consistency ─────────────────────────
NV = LATENT_SHAPE[0] // 2   # 뷰당 프레임 수 (10)


def _vae_decode(z4, n):
    return model.first_stage_pointmap_model.decode(z4, timesteps=n)


def decode_depth(z_frames):
    """latent (n,8,h,w) → 깊이 (n,H,W) [m], 포인트맵 VAE 디코더. decode_first_stage와 동일 경로.
    grad 모드: bf16 autocast + gradient checkpoint (OOM 방지) / no_grad: fp32"""
    zz = (z_frames / model.scale_factor)[:, :4]
    n = zz.shape[0]
    if torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = checkpoint(_vae_decode, zz.to(torch.bfloat16), n, use_reentrant=False)
        out = out.float()
    else:
        with torch.autocast("cuda", enabled=False):
            out = _vae_decode(zz.float(), n)
    return torch.clamp(((out[:, 2] + 1.) / 2.) * 3.0 - 1.0, -1.0, 2.0)   # unnormalize(-1, 2)


def view_ratio(x0, frames):
    """r_t = mean depth_R(t) / mean depth_L(t) (z>0 마스크), frames: 프레임 인덱스 리스트 → (len,)"""
    idx_l = torch.tensor(frames, device=x0.device)
    idx_r = idx_l + NV
    d = decode_depth(torch.cat([x0[idx_l], x0[idx_r]], 0))            # (2n,H,W)
    dl, dr = d[:len(frames)], d[len(frames):]
    ml, mr = (dl > 0).float().detach(), (dr > 0).float().detach()
    mean_l = (dl * ml).sum((1, 2)) / ml.sum((1, 2)).clamp_min(1)
    mean_r = (dr * mr).sum((1, 2)) / mr.sum((1, 2)).clamp_min(1)
    return mean_r / mean_l.clamp_min(1e-3)


teacher_ratio = {}
if args.cv_weight > 0:
    print("[3b] teacher 뷰 비 r_teacher 사전계산 (ODE 쌍 z 디코드)", flush=True)
    t0 = time.time()
    acc_r = {}
    with torch.no_grad():
        for f in pair_files:
            d = torch.load(f, map_location="cpu")
            z = d["z"].cuda().float()
            r = view_ratio(z, list(range(NV))).cpu()
            acc_r.setdefault(d["idx"], []).append(r)
    teacher_ratio = {k: torch.stack(v).mean(0) for k, v in acc_r.items()}
    allr = torch.stack(list(teacher_ratio.values()))
    print(f"  {len(teacher_ratio)}개 샘플, r_teacher 평균 {allr.mean():.3f} (프레임 std {allr.std(1).mean():.3f}, 샘플 std {allr.mean(1).std():.3f}) ({time.time()-t0:.0f}s)", flush=True)
    torch.cuda.empty_cache()
    mem("r_teacher 사전계산 후")


ANCHOR_FRAMES = [int(t) for t in args.anchor_frames.split(",")]


def anchor_loss(x0, idx):
    """자기 앵커: 예측 프레임 t(기본 0)의 깊이를 조건 프레임 깊이에 로그 L1로 맞춤. 양 뷰, GT 불필요, teacher와 무충돌"""
    dL_ref, dR_ref = anchor_depth[idx]
    dL_ref, dR_ref = dL_ref.to(x0.device), dR_ref.to(x0.device)
    idx_l = torch.tensor(ANCHOR_FRAMES, device=x0.device); idx_r = idx_l + NV
    d = decode_depth(torch.cat([x0[idx_l], x0[idx_r]], 0))               # (2n, H, W)
    n = len(ANCHOR_FRAMES)
    if d.shape[-2:] != dL_ref.shape:
        dL_ref = torch.nn.functional.interpolate(dL_ref[None, None], size=d.shape[-2:], mode="nearest")[0, 0]
        dR_ref = torch.nn.functional.interpolate(dR_ref[None, None], size=d.shape[-2:], mode="nearest")[0, 0]
    losses, stats_ = [], {}
    for v, ref, pred in (("L", dL_ref, d[:n]), ("R", dR_ref, d[n:])):
        m = (ref > 0)[None].expand_as(pred) & (pred > 0)                  # (n, H, W)
        if m.sum() < 100:
            continue
        diff = (torch.log(pred.clamp_min(1e-3)) - torch.log(ref.clamp_min(1e-3))[None]).abs()
        losses.append(diff[m].mean())
        m0 = m[0]
        stats_[f"s_{v}"] = torch.median(ref[m0] / pred[0][m0].clamp_min(1e-3)).item()   # 1.0이면 조건과 스케일 일치
    loss = torch.stack(losses).mean() if losses else x0.new_zeros(())
    return loss, stats_


def cv_loss(x0, idx):
    frames = sorted(random.sample(range(NV), args.cv_frames))
    r_s = view_ratio(x0, frames)
    r_t = (gt_ratio if args.cv_target == "gt" else teacher_ratio)[idx][frames].to(x0.device)
    loss = ((torch.log(r_s.clamp_min(1e-3)) - torch.log(r_t)) ** 2).mean()
    return loss, {"r_s": r_s.mean().item(), "r_t": r_t.mean().item()}


# ───────────────────────── 4. 핵심 함수 ─────────────────────────
def D(unet, x, sigma_val, c, ami):
    """denoiser: x_σ → x0 예측 (cond-only)"""
    sigma = x.new_full((x.shape[0],), float(sigma_val), dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return denoiser(unet, x, sigma, c, **ami).float()


def D_teacher_cfg(x, sigma_val, c, uc, ami):
    """real_score: teacher + 프레임별 CFG 가이더 (teacher 샘플링과 동일 경로)"""
    sigma = x.new_full((x.shape[0],), float(sigma_val), dtype=torch.float32)
    xx, ss, cc = guider.prepare_inputs(x, sigma, c, uc)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = denoiser(teacher, xx, ss, cc, **ami).float()
    return guider(out, ss)


def sample_sigma():
    if args.sigma_mode == "edm":       # teacher 학습 분포 EDMSampling(p_mean=1.0, p_std=1.6)
        s = math.exp(1.0 + 1.6 * random.gauss(0, 1))
    else:                              # log-uniform
        s = math.exp(random.uniform(math.log(args.sigma_min), math.log(args.sigma_max)))
    return min(max(s, args.sigma_min), args.sigma_max)


def run_generator(c, ami, k=None, generator=None, grad=True):
    """DMD2 backward simulation: k스텝까지 no_grad 재노이징, k번째 x0 예측만 grad"""
    n = len(sigmas)
    if k is None:
        k = random.randrange(n)
    x = torch.randn(LATENT_SHAPE, device="cuda", generator=generator) * math.sqrt(1.0 + sigmas[0] ** 2)
    with torch.no_grad():
        for i in range(k):
            x0 = D(gen, x, sigmas[i], c, ami)
            x = x0 + sigmas[i + 1] * torch.randn(x0.shape, device="cuda", generator=generator)
    if grad:
        x0 = D(gen, x, sigmas[k], c, ami)
    else:
        with torch.no_grad():
            x0 = D(gen, x, sigmas[k], c, ami)
    return x0, k


def dmd_loss(x0, c, uc, ami):
    with torch.no_grad():
        s = sample_sigma()
        x0_d = x0.detach()
        xs = x0_d + s * torch.randn_like(x0_d)
        pred_fake = D(fake, xs, s, c, ami)
        pred_real = D_teacher_cfg(xs, s, c, uc, ami)
        grad = pred_fake - pred_real
        normalizer = (x0_d - pred_real).abs().mean().clamp_min(1e-6)   # 샘플 1개 → 전역 평균 (DMD eq.8)
        grad = torch.nan_to_num(grad / normalizer)
        target = (x0_d - grad)
    loss = 0.5 * torch.nn.functional.mse_loss(x0.float(), target.float())
    return loss, {"sigma": s, "grad_abs": grad.abs().mean().item(),
                  "fake_real_gap": (pred_fake - pred_real).abs().mean().item()}


def critic_loss(c, ami):
    with torch.no_grad():
        x0, _ = run_generator(c, ami, grad=False)
        s = sample_sigma()
        xs = x0 + s * torch.randn_like(x0)
    pred = D(fake, xs, s, c, ami)
    w = (s ** 2 + 1.0) / (s ** 2)                       # EDMWeighting, σ_data=1 (teacher 학습과 동일)
    loss = torch.mean(w * (pred - x0) ** 2)
    return loss, {"sigma": s}


@torch.no_grad()
def diag(step, n=2):
    """고정 쌍 n개: 동일 시드로 full-step 생성 → teacher 최종 latent z와 비교 (붕괴 감지용)"""
    gen.eval()
    rows, rows_r = [], []
    for f in pair_files[:n]:
        d = torch.load(f, map_location="cpu")
        z = d["z"].cuda().float()
        c, uc, ami = to_cuda(d["idx"])
        g = torch.Generator(device="cuda").manual_seed(d["seed"])
        x0 = sampler(lambda x, s, cc: D(gen, x, s[0].item(), cc, ami),
                     torch.randn(LATENT_SHAPE, device="cuda", generator=g), c, generator=g)
        x1, _ = run_generator(c, ami, k=0, grad=False)
        rr = ""
        if args.cv_weight > 0 or True:
            try:
                if next(model.first_stage_pointmap_model.parameters()).is_cuda:
                    r_s = view_ratio(x0, list(range(NV))).mean().item(); r_z = view_ratio(z, list(range(NV))).mean().item()
                    rows_r.append((r_s, r_z, gt_ratio[d["idx"]].mean().item()))
            except Exception:
                pass
        rows.append((torch.nn.functional.mse_loss(x0, z).item(), (x0.std() / z.std()).item(),
                     (x1.std() / z.std()).item()))
    gen.train()
    m = [sum(r[i] for r in rows) / len(rows) for i in range(3)]
    rtxt = f" | 뷰비 r student {sum(r[0] for r in rows_r)/len(rows_r):.3f} vs teacher {sum(r[1] for r in rows_r)/len(rows_r):.3f} vs GT {sum(r[2] for r in rows_r)/len(rows_r):.3f}" if rows_r else ""
    print(f"[DIAG step {step}] full-step MSE(x0,z)={m[0]:.4f} | std비 full={m[1]:.3f} 1step={m[2]:.3f} "
          f"(1.0=teacher 수준, ↓=안개/평균 붕괴){rtxt}", flush=True)
    return m


def save(step, final=False):
    torch.save({"student": gen.state_dict(), "step": step, "sigmas": sigmas, "args": vars(args)},
               os.path.join(args.out_dir, "dmd_gen.pt"))
    torch.save({"fake": fake.state_dict(), "step": step}, os.path.join(args.out_dir, "dmd_fake.pt"))
    if str(step) in args.keep_steps.split(","):
        import shutil; shutil.copy(os.path.join(args.out_dir, "dmd_gen.pt"), os.path.join(args.out_dir, f"dmd_gen_step{step}.pt"))
    print(f"체크포인트 저장: {args.out_dir}/dmd_gen.pt, dmd_fake.pt (step {step})", flush=True)


# ───────────────────────── 5. 학습 루프 ─────────────────────────
print("[5/5] DMD 학습 시작", flush=True)
mem("학습 전")
diag(0)
t0 = time.time()
acc = {"g": [], "c": [], "gap": [], "gn_g": [], "gn_c": [], "cv": [], "r_s": [], "r_t": [], "lam": [], "an": [], "lam_a": [], "s_L": [], "s_R": []}
for step in range(START_STEP, args.max_steps + 1):
    f = random.choice(pair_files)
    idx = int(os.path.basename(f).split("_")[1])
    c, uc, ami = to_cuda(idx)

    # generator 업데이트 (gen_every 스텝마다)
    if step % args.gen_every == 0:
        fake.requires_grad_(False)
        x0, k = run_generator(c, ami, grad=True)
        loss_g, info = dmd_loss(x0, c, uc, ami)
        if args.cv_weight > 0:
            l_cv, cvi = cv_loss(x0, idx)
            # x0 수준 그래디언트 균형: |∂(λ·L_cv)/∂x0| = cv_weight × |∂L_dmd/∂x0|  (DMD 자체 정규화와 같은 정신)
            g_dmd = torch.autograd.grad(loss_g, x0, retain_graph=True)[0].abs().mean()
            g_cv = torch.autograd.grad(l_cv, x0, retain_graph=True)[0].abs().mean()
            lam = args.cv_weight * (g_dmd / g_cv.clamp_min(1e-12)).item()
            acc["cv"].append(l_cv.item()); acc["r_s"].append(cvi["r_s"]); acc["r_t"].append(cvi["r_t"]); acc["lam"].append(lam)
            loss_g = loss_g + lam * l_cv
        if args.anchor_weight > 0:
            l_an, ani = anchor_loss(x0, idx)
            if l_an.requires_grad:
                g_dmd2 = torch.autograd.grad(loss_g, x0, retain_graph=True)[0].abs().mean()
                g_an = torch.autograd.grad(l_an, x0, retain_graph=True)[0].abs().mean()
                lam_a = args.anchor_weight * (g_dmd2 / g_an.clamp_min(1e-12)).item()
                acc["an"].append(l_an.item()); acc["lam_a"].append(lam_a)
                acc["s_L"].append(ani.get("s_L", float("nan"))); acc["s_R"].append(ani.get("s_R", float("nan")))
                loss_g = loss_g + lam_a * l_an
        loss_g.backward()
        gn = torch.nn.utils.clip_grad_norm_(gen.parameters(), args.max_grad_norm)
        opt_g.step(); opt_g.zero_grad(set_to_none=True)
        fake.requires_grad_(True)
        acc["g"].append(loss_g.item()); acc["gap"].append(info["fake_real_gap"]); acc["gn_g"].append(gn.item())
        del x0, loss_g
        torch.cuda.empty_cache()
        if step == args.gen_every:
            mem("첫 generator step 후")

    # critic(fake_score) 업데이트 (매 스텝)
    loss_c, _ = critic_loss(c, ami)
    loss_c.backward()
    gn = torch.nn.utils.clip_grad_norm_(fake.parameters(), args.max_grad_norm)
    opt_c.step(); opt_c.zero_grad(set_to_none=True)
    acc["c"].append(loss_c.item()); acc["gn_c"].append(gn.item())
    del loss_c
    torch.cuda.empty_cache()
    if step == 1:
        mem("첫 critic step 후")

    if step % args.log_every == 0:
        el = time.time() - t0
        g = f"gen {sum(acc['g'])/len(acc['g']):.4f} gap {sum(acc['gap'])/len(acc['gap']):.4f} |g| {sum(acc['gn_g'])/len(acc['gn_g']):.2f}" if acc["g"] else "gen -"
        if acc["cv"]:
            g += f" | cv {sum(acc['cv'])/len(acc['cv']):.4f} λ {sum(acc['lam'])/len(acc['lam']):.2e} (r_s {sum(acc['r_s'])/len(acc['r_s']):.3f} vs r_t {sum(acc['r_t'])/len(acc['r_t']):.3f})"
        if acc["an"]:
            import math as _m
            sL = [x for x in acc["s_L"] if not _m.isnan(x)]; sR = [x for x in acc["s_R"] if not _m.isnan(x)]
            g += f" | anchor {sum(acc['an'])/len(acc['an']):.4f} λ {sum(acc['lam_a'])/len(acc['lam_a']):.2e} (s_L {sum(sL)/max(len(sL),1):.3f} s_R {sum(sR)/max(len(sR),1):.3f}; 1.0=조건과 일치)"
        print(f"[step {step}/{args.max_steps}] {g} | critic {sum(acc['c'])/len(acc['c']):.4f} |g| {sum(acc['gn_c'])/len(acc['gn_c']):.2f} "
              f"| {el/step:.1f}s/step | max {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
        acc = {k: [] for k in acc}
    if step % args.diag_every == 0:
        diag(step)
    if step % args.save_every == 0 or step == args.max_steps:
        save(step)

print("완료", flush=True)
