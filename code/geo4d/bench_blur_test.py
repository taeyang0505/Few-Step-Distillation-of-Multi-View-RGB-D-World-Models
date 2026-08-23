"""블러 가설 검증: "적은 스텝 → 조건부 평균(블러) 수렴"의 3중 테스트
① LPIPS (지각 거리 — PSNR과의 해리 확인)  ② Laplacian 선명도 (블러 직접 측정)
③ 시드 4개 반복 생성의 샘플 간 분산 (평균 수렴이면 분산 → 0)"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import time
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
from skimage.metrics import structural_similarity as ssim_fn

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
ckpt_path = f"{output_dir}/4dgen.ckpt"
STEPS_LIST = [25, 16, 8, 4, 2, 1]
SEEDS = [0, 1, 2, 3]
N_BATCHES = 2


def unnormalize(input, min, max):
    return torch.clamp(((input + 1.) / 2.) * (max - min) + min, min, max)


def get_rgb(outputs, key):
    """(T, 3, H, W), [0,1] 범위 RGB 비디오."""
    return unnormalize(outputs["video_dict"][key][:, 3:], 0, 1)


LAP_KERNEL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def sharpness(video):
    """Laplacian 분산 (프레임 평균). video: (T,3,H,W) [0,1]."""
    gray = video.mean(dim=1, keepdim=True)  # (T,1,H,W)
    lap = F.conv2d(gray, LAP_KERNEL.to(video.device))
    return lap.var(dim=(1, 2, 3)).mean().item()


def psnr(pred, gt):
    mse = torch.mean((pred - gt).pow(2))
    return (10 * torch.log10(1. / mse)).item()


def ssim_video(pred, gt):
    """프레임별 SSIM 평균 (CPU, skimage)."""
    p = pred.cpu().numpy().transpose(0, 2, 3, 1)
    g = gt.cpu().numpy().transpose(0, 2, 3, 1)
    return float(np.mean([ssim_fn(g[t], p[t], channel_axis=2, data_range=1.0) for t in range(p.shape[0])]))


print("[1/3] 모델·LPIPS 로드")
cfg = OmegaConf.load(f"{output_dir}/config.yaml")
for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and "desc" in cfg[key]:
        cfg[key] = cfg[key]["value"]
cls = hydra.utils.get_class(cfg._target_)
cfg.logging.mode = "offline"
cfg.model.params.ckpt_path = ckpt_path
cfg.training.seed = 42
cfg.training.output_dir = "/home/sun4208/Geo4D/bench_out"
workspace = cls(cfg)
model = workspace.lightning_module_wrapper.to("cuda")
model.eval()
sampler = model.sampler
lpips_net = lpips_lib.LPIPS(net="alex").cuda().eval()

print(f"[2/3] 데이터 준비 (배치 {N_BATCHES}개)")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
batches = []
for i, b in enumerate(loader):
    if i >= N_BATCHES:
        break
    n = b["pointmap"].shape[1]
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True))
    ib["num_video_frames"] = n
    batches.append(ib)

with torch.no_grad():
    model.log_images(batches[0])  # 워밍업

# GT 기준 선명도 (1회)
with torch.no_grad():
    out0 = model.log_images(batches[0])
gt_sharp = np.mean([sharpness(get_rgb(out0, f"gt_video_{v}")) for v in ["left", "right"]])
print(f"GT 선명도 (Laplacian var): {gt_sharp:.6f}")

print(f"[3/3] 스윕: steps={STEPS_LIST} x seeds={SEEDS} x 배치 {N_BATCHES}개")
rows = []
for steps in STEPS_LIST:
    sampler.num_steps = steps
    m = {k: [] for k in ["PSNR", "SSIM", "LPIPS", "sharp", "div"]}
    t0 = time.time()
    for ib in batches:
        seed_videos = {v: [] for v in ["left", "right"]}
        with torch.no_grad():
            for seed in SEEDS:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                out = model.log_images(ib)
                for v in ["left", "right"]:
                    pred = get_rgb(out, f"sampled_video_{v}")
                    gt = get_rgb(out, f"gt_video_{v}")
                    seed_videos[v].append(pred.cpu())
                    m["PSNR"].append(psnr(pred, gt))
                    m["SSIM"].append(ssim_video(pred, gt))
                    d = lpips_net(pred * 2 - 1, gt * 2 - 1)  # LPIPS는 [-1,1] 입력
                    m["LPIPS"].append(d.mean().item())
                    m["sharp"].append(sharpness(pred))
        # ③ 시드 간 다양성: 같은 입력 4개 샘플의 픽셀 표준편차 평균
        for v in ["left", "right"]:
            stack = torch.stack(seed_videos[v])  # (S, T, 3, H, W)
            m["div"].append(stack.std(dim=0).mean().item())
    row = (steps, np.mean(m["PSNR"]), np.mean(m["SSIM"]), np.mean(m["LPIPS"]),
           np.mean(m["sharp"]), np.mean(m["div"]))
    rows.append(row)
    print(f"[steps={steps:2d}] ({time.time()-t0:.0f}초) PSNR {row[1]:.2f} | SSIM {row[2]:.4f} | LPIPS {row[3]:.4f} | 선명도 {row[4]:.6f} (GT {gt_sharp:.6f}) | 시드간 std {row[5]:.5f}")

report = [f"=== 블러 가설 3중 검증 (배치 {N_BATCHES} x 시드 {len(SEEDS)}, GT 선명도={gt_sharp:.6f}) ===",
          "steps | PSNR^ | SSIM^ | LPIPS v | 선명도(pred) | 시드간 다양성"]
for r in rows:
    report.append(f"{r[0]:>5} | {r[1]:>5.2f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.6f} | {r[5]:.5f}")
base = rows[0]
report.append("--- 25스텝 대비 상대 변화(%) ---")
for r in rows[1:]:
    report.append(f"{r[0]:>5} | PSNR {100*r[1]/base[1]-100:+.1f}% | SSIM {100*r[2]/base[2]-100:+.1f}% | LPIPS {100*r[3]/base[3]-100:+.1f}% | 선명도 {100*r[4]/base[4]-100:+.1f}% | 다양성 {100*r[5]/base[5]-100:+.1f}%")
report.append("판정 기준: 블러 가설이 맞다면 스텝↓에서 (i) LPIPS 악화(+), (ii) 선명도 하락(-), (iii) 다양성 붕괴(-)가 동시에 나타나야 함. PSNR/SSIM은 개선돼도 무방(해리).")
text = "\n".join(report)
print(); print(text)
with open("/home/sun4208/Geo4D/bench_out/blur_test_results.txt", "w") as f:
    f.write(text + "\n")
print()
print("결과 저장: ~/Geo4D/bench_out/blur_test_results.txt")
