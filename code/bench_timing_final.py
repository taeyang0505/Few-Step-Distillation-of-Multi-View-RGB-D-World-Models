"""논문 headline 시간 재측정 — 순수 추론 경로(평가 코드 제외), 컴포넌트별 실측
설정: T25 = teacher EulerEDM 25스텝 + CFG(uc 포함, fp32) / S3b·S4b·S5b = DMD student 재노이징 3·4·5스텝 + per-view anchor + bf16
각 설정: 총 시간과 UNet / conditioner / VAE 디코딩 / anchor 분해 + 같은 시드 생성물의 PSNR·AbsRel(가짜 픽셀 제외)로 무결성 확인
출력: ~/Geo4D/bench_out/timing_final.txt (+ _raw.json)"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, json, time, random
import numpy as np, torch, hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps, apply_cond_anchor_per_view

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--n", type=int, default=5, help="측정 배치 수 (설정당 생성 횟수)")
ap.add_argument("--out", default="/home/sun4208/Geo4D/bench_out/timing_final")
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
euler = model.sampler                                                   # teacher 기본 샘플러(EulerEDM + CFG)
teacher_sd = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
student_sd = torch.load(a.student_ckpt, map_location="cpu")["student"]

cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False; cfg.dataloader.batch_size = 1; cfg.dataloader.num_workers = 0; cfg.dataloader.persistent_workers = False
random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
batches = []
for i, b in enumerate(DataLoader(dataset, **cfg.dataloader)):
    if i >= a.n: break
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True)); ib["num_video_frames"] = b["pointmap"].shape[1]; batches.append(ib)
print(f"[1/3] 배치 {len(batches)}개 준비", flush=True)

# ── 컴포넌트 타이머 (원 함수를 감싸 GPU 동기화 후 누적)
stats = {}
def timed(name, fn):
    s = stats.setdefault(name, [0.0, 0])
    def w(*x, **k):
        torch.cuda.synchronize(); t0 = time.time(); out = fn(*x, **k); torch.cuda.synchronize(); s[0] += time.time() - t0; s[1] += 1
        return out
    return w
model.model.forward = timed("UNet", model.model.forward)
model.conditioner.forward = timed("conditioner", model.conditioner.forward)
model.first_stage_pointmap_model.decode = timed("decode_pm", model.first_stage_pointmap_model.decode)
model.first_stage_color_model.decode = timed("decode_col", model.first_stage_color_model.decode)

unnorm = lambda x, mn, mx: torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)

def decode(z, bf16):
    z = 1.0 / model.scale_factor * z
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        o1 = model.first_stage_pointmap_model.decode(z[:, :4], timesteps=z.shape[0])
        o2 = model.first_stage_color_model.decode(z[:, 4:], timesteps=z.shape[0])
    return torch.cat([o1.float(), o2.float()], dim=1)

def infer(batch_old, use_cfg, bf16, anchor, seed=0):
    """sample_multiview_video와 동일한 순수 추론 경로 (평가 손실·GT 재구성 없음)"""
    batch = {k: v[0:1] for k, v in batch_old.items() if k != "num_video_frames" and torch.is_tensor(v)}
    batch.update({k: v for k, v in batch_old.items() if not torch.is_tensor(v)})
    batch["num_video_frames"] = batch_old["num_video_frames"]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        if use_cfg:
            c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch)   # teacher: c + uc (CFG)
        else:
            c, uc = model.conditioner(batch), None                                            # student: uc 생략
    c = {k: (v.float() if torch.is_tensor(v) else v) for k, v in c.items()}
    if uc is not None:
        uc = {k: (v.float() if torch.is_tensor(v) else v) for k, v in uc.items()}
    ami = {"num_video_frames": batch["num_video_frames"], "image_only_indicator": batch["image_only_indicator"]}
    def denoiser(x, sigma, cc):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            return model.denoiser(model.model, x, sigma, cc, **ami).float()
    mv = torch.cat([batch["pointmap"][0], batch["pointmap_right"][0]], dim=0)
    BT, C, H, W = mv.shape
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    randn = torch.randn((BT, 8, H // 8, W // 8), device=mv.device)
    z = model.sampler(denoiser, randn, cond=c, uc=uc)
    x = decode(z, bf16)
    vd = {"sampled_video_left": x[:BT // 2], "sampled_video_right": x[BT // 2:],
          "gt_video_left": batch["pointmap"][0], "gt_video_right": batch["pointmap_right"][0],
          "cond_pointmap_left": batch["cond_pointmaps_without_noise"][0], "cond_pointmap_right": batch["cond_pointmaps_without_noise_right"][0],
          "extra": {"cam_extr": batch["cam_extr"], "cam_extr_right": batch["cam_extr_right"]}}
    if anchor:
        torch.cuda.synchronize(); t0 = time.time(); apply_cond_anchor_per_view(vd); torch.cuda.synchronize()
        s = stats.setdefault("anchor", [0.0, 0]); s[0] += time.time() - t0; s[1] += 1
    return vd

def metrics(vd):
    out = {}
    for v in ["left", "right"]:
        p, gt = vd[f"sampled_video_{v}"], vd[f"gt_video_{v}"]
        pr, gr = unnorm(p[:, 3:], 0, 1), unnorm(gt[:, 3:], 0, 1)
        out[f"PSNR_{v[0]}"] = (10 * torch.log10(1. / torch.mean((pr - gr) ** 2))).item()
        d1, d2 = unnorm(p[:, 2], -1, 2), unnorm(gt[:, 2], -1, 2)
        m = (d1 > 0) & (d2 > 0)
        if v == "right":                                                   # 가짜 픽셀 제외 (발견 10)
            E1 = vd["extra"]["cam_extr"].reshape(-1, 4, 4)[0].float(); E2 = vd["extra"]["cam_extr_right"].reshape(-1, 4, 4)[0].float()
            t = (torch.linalg.inv(E1) @ E2)[:3, 3].to(d2.device).view(1, 3, 1, 1)
            gxyz = unnorm(gt[:, :3], -1, 2); m = m & ~((gxyz - t).abs().amax(dim=1) < 2e-3)
        out[f"AbsRel_{v[0]}"] = torch.mean(torch.abs(d1[m] - d2[m]) / d2[m]).item()
    return out

# name, who, steps, use_cfg, bf16, anchor
CONFIGS = [("T25 (teacher, Euler 25스텝, CFG, fp32)", "teacher", 25, True,  False, False),
           ("S3b (student, 재노이징 3스텝, anchor, bf16)", "student", 3, False, True,  True),
           ("S4b (student, 재노이징 4스텝, anchor, bf16)", "student", 4, False, True,  True),
           ("S5b (student, 재노이징 5스텝, anchor, bf16)", "student", 5, False, True,  True)]

print("[2/3] 측정 시작 (설정당 워밍업 1회 + 본 측정 %d회)" % len(batches), flush=True)
results, cur = [], None
for name, who, steps, use_cfg, bf16, anchor in CONFIGS:
    if who != cur:
        model.model.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
        cur = who
    if who == "teacher":
        model.sampler = euler; model.sampler.num_steps = steps
    else:
        model.sampler = RenoiseSampler(sigmas_for_steps(steps))
    with torch.no_grad(): infer(batches[0], use_cfg, bf16, anchor)                      # 워밍업
    for v in stats.values(): v[0] = 0.0; v[1] = 0
    torch.cuda.synchronize(); t0 = time.time(); ms = []
    with torch.no_grad():
        for ib in batches: ms.append(metrics(infer(ib, use_cfg, bf16, anchor)))
    torch.cuda.synchronize()
    total = (time.time() - t0) / len(batches)
    comp = {k: v[0] / len(batches) for k, v in stats.items()}
    calls = {k: v[1] / len(batches) for k, v in stats.items()}
    mm = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    dec = comp.get("decode_pm", 0) + comp.get("decode_col", 0)
    results.append({"name": name, "total": total, "UNet": comp.get("UNet", 0), "cond": comp.get("conditioner", 0),
                    "decode": dec, "anchor": comp.get("anchor", 0), "unet_calls": calls.get("UNet", 0),
                    "cond_calls": calls.get("conditioner", 0), **mm})
    print(f"[{name}] 총 {total:.2f}초 | UNet {comp.get('UNet',0):.2f}({calls.get('UNet',0):.0f}회) "
          f"cond {comp.get('conditioner',0):.2f}({calls.get('conditioner',0):.0f}회) dec {dec:.2f} anchor {comp.get('anchor',0):.3f} "
          f"| PSNR L {mm['PSNR_l']:.2f} R {mm['PSNR_r']:.2f} | AbsRel L {mm['AbsRel_l']:.4f} R {mm['AbsRel_r']:.4f}", flush=True)

print("[3/3] 저장", flush=True)
T = results[0]["total"]
L = [f"=== 순수 추론 시간 재측정 (평가 코드 제외, 배치 {len(batches)}개 평균, RTX 5090) ===", "",
     "설정 | 총(초) | 배속 | UNet(호출) | conditioner(호출) | VAE 디코딩 | anchor | PSNR L/R | AbsRel L/R (가짜 픽셀 제외)"]
for r in results:
    L.append(f"{r['name']:<44} | {r['total']:.2f} | {T/r['total']:.1f}x | {r['UNet']:.2f}({r['unet_calls']:.0f}) | "
             f"{r['cond']:.2f}({r['cond_calls']:.0f}) | {r['decode']:.2f} | {r['anchor']:.3f} | "
             f"{r['PSNR_l']:.2f}/{r['PSNR_r']:.2f} | {r['AbsRel_l']:.4f}/{r['AbsRel_r']:.4f}")
s3 = next((r for r in results if r["name"].startswith("S3b")), None)
if s3:
    L += ["", f"고정비(스텝 수와 무관) = conditioner {s3['cond']:.2f} + 디코딩 {s3['decode']:.2f} = {s3['cond']+s3['decode']:.2f}초 "
              f"→ 스텝을 0으로 줄여도 이 시간은 남음 (휴머노이드 0.3-0.5초 목표의 벽)",
          f"teacher UNet 호출당 {results[0]['UNet']/max(results[0]['unet_calls'],1):.3f}초(CFG로 배치 40) vs "
          f"student {s3['UNet']/max(s3['unet_calls'],1):.3f}초(CFG 없음, 배치 20)",
          "H4b(하이브리드 4스텝)는 마지막 호출만 teacher UNet(CFG 없음, 같은 배치)이라 S4b와 사실상 동일한 비용."]
text = "\n".join(L); print(); print(text)
with open(a.out + ".txt", "w") as f: f.write(text + "\n")
with open(a.out + "_raw.json", "w") as f: json.dump(results, f, indent=1)
print(f"\n저장: {a.out}.txt / _raw.json")
