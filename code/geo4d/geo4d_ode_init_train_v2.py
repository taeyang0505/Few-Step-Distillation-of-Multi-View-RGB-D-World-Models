"""Step 6-2 v2: 진짜 궤적 기반 ODE 회귀 초기화
v1 대비: ① 입력이 teacher 궤적의 실제 중간 상태 (on-manifold), ② σ≈0 슬롯 제거,
③ σ별 분리 로깅, ④ 기본 1200스텝."""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import argparse
import glob
import json
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

parser = argparse.ArgumentParser()
parser.add_argument("--pairs_dir", type=str, default="/home/sun4208/Geo4D/ode_pairs_v2")
parser.add_argument("--out_ckpt", type=str, default="/home/sun4208/Geo4D/ode_init_geo4d_v2.pt")
parser.add_argument("--max_steps", type=int, default=1200)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--save_every", type=int, default=200)
args_cli = parser.parse_args()

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
print("[1/4] 모델 로드 (teacher 가중치로 student 초기화)")
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

student = model.model
student.requires_grad_(True).train()
model.conditioner.requires_grad_(False)
student.to(torch.bfloat16)
print(f"학습 파라미터: {sum(p.numel() for p in student.parameters() if p.requires_grad)/1e9:.2f}B")

print("[2/4] 궤적 데이터 로드")
with open(os.path.join(args_cli.pairs_dir, "meta.json")) as f:
    meta = json.load(f)
target_sigmas = meta["target_sigmas"]
sigma_keys = [f"{s:.4f}" for s in target_sigmas]
print("학습 σ:", [f"{s:.1f}" for s in target_sigmas], "(σ≈0 제외)")
pair_files = sorted(glob.glob(os.path.join(args_cli.pairs_dir, "pair_*.pt")))
print(f"ODE 쌍: {len(pair_files)}개")

print("[3/4] 데이터셋·optimizer 준비")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit([p for p in student.parameters() if p.requires_grad], lr=args_cli.lr)

cond_cache = {}
def get_cond(idx):
    if idx in cond_cache:
        return cond_cache[idx]
    b_raw = default_collate([dataset[idx]])
    b_raw = dict_apply(b_raw, lambda x: x.to("cuda", non_blocking=True))
    b_raw["num_video_frames"] = b_raw["pointmap"].shape[1]
    batch = {}
    batch.update({k: v[0:1] for (k, v) in b_raw.items()
                  if k != "num_video_frames" and torch.is_tensor(v)})
    batch.update({k: v for k, v in b_raw.items() if not torch.is_tensor(v)})
    batch["num_video_frames"] = b_raw["num_video_frames"]
    with torch.no_grad():
        c, _ = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch)
    ami = {"num_video_frames": batch["num_video_frames"],
           "image_only_indicator": batch["image_only_indicator"]}
    if len(cond_cache) < 40:
        cond_cache[idx] = (c, ami)
    return c, ami

print("[4/4] 학습 시작")
random.seed(0)
t0 = time.time()
step = 0
loss_acc = []
loss_by_sigma = {}
while step < args_cli.max_steps:
    f = random.choice(pair_files)
    d = torch.load(f, map_location="cpu")
    z = d["z"].to("cuda", dtype=torch.float32)
    c, ami = get_cond(d["idx"])

    si = random.randrange(len(target_sigmas))
    entry = d["traj"][sigma_keys[si]]
    x = entry["x"].to("cuda", dtype=torch.float32)          # teacher 궤적의 실제 중간 상태
    if x.shape[0] == 2 * z.shape[0]:
        x = x[: z.shape[0]]  # CFG 이중 배치([x,x] 복제)의 절반만 사용
    sigma_val = entry["sigma_actual"]                        # 캡처 시점의 실제 σ
    sigma = torch.full((z.shape[0],), sigma_val, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model.denoiser(student, x, sigma, c, **ami)
        loss = torch.nn.functional.mse_loss(pred.float(), z)

    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    loss_acc.append(loss.item())
    loss_by_sigma.setdefault(si, []).append(loss.item())

    if step % args_cli.log_every == 0:
        el = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 2**30
        per = " | ".join(f"σ{target_sigmas[k]:.0f}:{sum(v)/len(v):.3f}"
                         for k, v in sorted(loss_by_sigma.items()))
        print(f"[step {step}/{args_cli.max_steps}] loss={sum(loss_acc)/len(loss_acc):.4f} [{per}] "
              f"({el/step:.1f}초/step, VRAM {vram:.1f}GB)", flush=True)
        loss_acc = []
        loss_by_sigma = {}
    if step % args_cli.save_every == 0 or step == args_cli.max_steps:
        torch.save({"student": student.state_dict(), "step": step,
                    "target_sigmas": target_sigmas}, args_cli.out_ckpt)
        print(f"체크포인트 저장: {args_cli.out_ckpt} (step {step})", flush=True)

print("완료")
