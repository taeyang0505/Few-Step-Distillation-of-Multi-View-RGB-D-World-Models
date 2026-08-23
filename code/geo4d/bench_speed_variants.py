"""Step 7-②: 순수 추론(uc 생략) 속도 변형 실측 — student 재노이징 3스텝
변형: A fp32 기준 / B VAE 디코더 bf16 / C +UNet bf16 / D +conditioner bf16 / E D+torch.compile(디코더) / F E+torch.compile(UNet)
각 변형: 시간(컴포넌트별) + 같은 시드 생성물의 PSNR·AbsRel(왼/오, 가짜 픽셀 제외)로 품질 드리프트 확인
출력: ~/Geo4D/bench_out/speed_variants.txt"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, time, random, contextlib, traceback
import numpy as np, torch, hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps, apply_cond_anchor_per_view

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--n", type=int, default=5, help="시간 측정 배치 수")
ap.add_argument("--no_compile", action="store_true")
a = ap.parse_args()

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
cfg = OmegaConf.load(f"{output_dir}/config.yaml")
for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and "desc" in cfg[key]:
        cfg[key] = cfg[key]["value"]
cls = hydra.utils.get_class(cfg._target_)
cfg.logging.mode = "offline"; cfg.model.params.ckpt_path = f"{output_dir}/4dgen.ckpt"
cfg.training.seed = 42; cfg.training.output_dir = "/home/sun4208/Geo4D/bench_out"
model = cls(cfg).lightning_module_wrapper.to("cuda"); model.eval()
model.model.load_state_dict(torch.load(a.student_ckpt, map_location="cpu")["student"], strict=False)
model.sampler = RenoiseSampler(sigmas_for_steps(3))

cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False; cfg.dataloader.batch_size = 1; cfg.dataloader.num_workers = 0; cfg.dataloader.persistent_workers = False
random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
batches = []
for i, b in enumerate(DataLoader(dataset, **cfg.dataloader)):
    if i >= a.n: break
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True)); ib["num_video_frames"] = b["pointmap"].shape[1]; batches.append(ib)

# ── 타이머
stats = {}
def timed(name, fn):
    s = stats.setdefault(name, [0.0, 0])
    def w(*x, **k):
        torch.cuda.synchronize(); t0 = time.time(); out = fn(*x, **k); torch.cuda.synchronize(); s[0] += time.time() - t0; s[1] += 1
        return out
    return w
unet_fn = model.model.forward
cond_fn = model.conditioner.forward
dec_pm = model.first_stage_pointmap_model.decode
dec_col = model.first_stage_color_model.decode
model.model.forward = timed("UNet", unet_fn)
model.conditioner.forward = timed("conditioner", cond_fn)
model.first_stage_pointmap_model.decode = timed("VAE decode(pointmap)", dec_pm)
model.first_stage_color_model.decode = timed("VAE decode(color)", dec_col)

unnorm = lambda x, mn, mx: torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)

DECODE_COLOR = [True]
def decode(z, ac):
    z = 1.0 / model.scale_factor * z
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
        o1 = model.first_stage_pointmap_model.decode(z[:, :4], timesteps=z.shape[0])
        if DECODE_COLOR[0]:
            o2 = model.first_stage_color_model.decode(z[:, 4:], timesteps=z.shape[0])
        else:
            o2 = torch.zeros_like(o1)   # 깊이만 쓰는 용도(계획용)에서는 색 디코더 생략
    return torch.cat([o1.float(), o2.float()], dim=1)

def infer(batch_old, ac_unet, ac_cond, ac_dec, seed=0):
    batch = {k: v[0:1] for k, v in batch_old.items() if k != "num_video_frames" and torch.is_tensor(v)}
    batch.update({k: v for k, v in batch_old.items() if not torch.is_tensor(v)})
    batch["num_video_frames"] = batch_old["num_video_frames"]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac_cond):
        c = model.conditioner(batch)
    c = {k: (v.float() if torch.is_tensor(v) else v) for k, v in c.items()}
    ami = {"num_video_frames": batch["num_video_frames"], "image_only_indicator": batch["image_only_indicator"]}
    def denoiser(x, sigma, cc):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac_unet):
            return model.denoiser(model.model, x, sigma, cc, **ami).float()
    mv = torch.cat([batch["pointmap"][0], batch["pointmap_right"][0]], dim=0)
    BT, C, H, W = mv.shape
    g = torch.Generator(device="cuda").manual_seed(seed)
    randn = torch.randn((BT, 8, H // 8, W // 8), device="cuda", generator=g)
    z = model.sampler(denoiser, randn, cond=c, uc=None, generator=g)
    x = decode(z, ac_dec)
    vd = {"sampled_video_left": x[:BT // 2], "sampled_video_right": x[BT // 2:],
          "gt_video_left": batch["pointmap"][0], "gt_video_right": batch["pointmap_right"][0],
          "cond_pointmap_left": batch["cond_pointmaps_without_noise"][0], "cond_pointmap_right": batch["cond_pointmaps_without_noise_right"][0],
          "extra": {"cam_extr": batch["cam_extr"], "cam_extr_right": batch["cam_extr_right"]}}
    apply_cond_anchor_per_view(vd)
    return vd

def metrics(vd):
    out = {}
    for v in ["left", "right"]:
        p = vd[f"sampled_video_{v}"]; gt = vd[f"gt_video_{v}"]
        pr, gr = unnorm(p[:, 3:], 0, 1), unnorm(gt[:, 3:], 0, 1)
        out[f"PSNR_{v[0]}"] = (10 * torch.log10(1. / torch.mean((pr - gr) ** 2))).item()
        d1, d2 = unnorm(p[:, 2], -1, 2), unnorm(gt[:, 2], -1, 2)
        m = (d1 > 0) & (d2 > 0)
        if v == "right":
            E1 = vd["extra"]["cam_extr"].reshape(-1, 4, 4)[0].float(); E2 = vd["extra"]["cam_extr_right"].reshape(-1, 4, 4)[0].float()
            t = (torch.linalg.inv(E1) @ E2)[:3, 3].to(d2.device).view(1, 3, 1, 1)
            gxyz = unnorm(gt[:, :3], -1, 2); m = m & ~((gxyz - t).abs().amax(dim=1) < 2e-3)
        out[f"AbsRel_{v[0]}"] = torch.mean(torch.abs(d1[m] - d2[m]) / d2[m]).item()
    return out

VARIANTS = [("A fp32 기준", False, False, False), ("B 디코더 bf16", False, False, True), ("C +UNet bf16", True, False, True),
            ("D +conditioner bf16", True, True, True)]
results = []
def run(label, ac_unet, ac_cond, ac_dec):
    with torch.no_grad(): infer(batches[0], ac_unet, ac_cond, ac_dec)      # 워밍업(compile 포함)
    for v in stats.values(): v[0] = 0.0; v[1] = 0
    torch.cuda.synchronize(); t0 = time.time(); ms = []
    with torch.no_grad():
        for ib in batches: ms.append(metrics(infer(ib, ac_unet, ac_cond, ac_dec)))
    torch.cuda.synchronize(); total = (time.time() - t0) / len(batches)
    comp = {k: v[0] / len(batches) for k, v in stats.items()}
    mm = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    results.append((label, total, comp, mm))
    print(f"[{label:<22}] 총 {total:.2f}초 | UNet {comp['UNet']:.2f} cond {comp['conditioner']:.2f} dec {comp['VAE decode(pointmap)']+comp['VAE decode(color)']:.2f} "
          f"| PSNR L {mm['PSNR_l']:.2f} R {mm['PSNR_r']:.2f} | AbsRel L {mm['AbsRel_l']:.4f} R {mm['AbsRel_r']:.4f}", flush=True)

for v in VARIANTS: run(*v)

if not a.no_compile:
    for label, what in [("E D+compile(디코더)", "dec"), ("F E+compile(UNet)", "unet"), ("G F+compile(cond)", "cond"), ("H G+색 디코더 생략", "nocolor")]:
        try:
            t0 = time.time()
            if what == "dec":
                model.first_stage_pointmap_model.decoder = torch.compile(model.first_stage_pointmap_model.decoder)
                model.first_stage_color_model.decoder = torch.compile(model.first_stage_color_model.decoder)
            elif what == "unet":
                model.model.diffusion_model = torch.compile(model.model.diffusion_model)
                model.model.diffusion_model_2 = model.model.diffusion_model   # 가중치 공유(발견 8)
            elif what == "cond":
                for emb in model.conditioner.embedders:
                    if hasattr(emb, "encoder") and isinstance(emb.encoder, torch.nn.Module):
                        emb.encoder = torch.compile(emb.encoder)
                    elif hasattr(emb, "model") and isinstance(emb.model, torch.nn.Module):
                        emb.model = torch.compile(emb.model)
            else:
                DECODE_COLOR[0] = False
            with torch.no_grad(): infer(batches[0], True, True, True)
            print(f"  ({label} 컴파일+워밍업 {time.time()-t0:.0f}초)", flush=True)
            run(label, True, True, True)
        except Exception as e:
            print(f"[{label}] 실패: {type(e).__name__}: {str(e)[:200]}", flush=True)

L = ["=== Step 7-②: 순수 추론(uc 생략) 속도 변형 — student 재노이징 3스텝, 배치 %d ===" % len(batches), "",
     "변형 | 총(초) | UNet | conditioner | VAE 디코딩 | PSNR L/R | AbsRel L/R (가짜 픽셀 제외)"]
for label, total, comp, mm in results:
    L.append(f"{label:<22} | {total:.2f} | {comp['UNet']:.2f} | {comp['conditioner']:.2f} | {comp['VAE decode(pointmap)']+comp['VAE decode(color)']:.2f} | {mm['PSNR_l']:.2f}/{mm['PSNR_r']:.2f} | {mm['AbsRel_l']:.4f}/{mm['AbsRel_r']:.4f}")
base = results[0]
L += ["", "[A 대비 품질 변화] " + " | ".join(f"{r[0].split()[0]}: PSNR {r[3]['PSNR_l']-base[3]['PSNR_l']:+.2f} AbsRel L {r[3]['AbsRel_l']-base[3]['AbsRel_l']:+.4f} R {r[3]['AbsRel_r']-base[3]['AbsRel_r']:+.4f}" for r in results[1:])]
L += ["판정: 시간이 줄고 PSNR/AbsRel 변화가 ±0.05dB/±0.002 이내면 무손실 가속으로 채택."]
text = "\n".join(L); print(); print(text)
with open("/home/sun4208/Geo4D/bench_out/speed_variants.txt", "w") as f: f.write(text + "\n")
print("\n저장: ~/Geo4D/bench_out/speed_variants.txt")
