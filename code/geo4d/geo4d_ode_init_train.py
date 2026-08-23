"""Step 6-2: Geo4D student의 ODE 회귀 초기화 학습 (5090 1장)
쌍 데이터(노이즈 시드, teacher 최종 latent)로 student를 few-step 예측기로 회귀:
  x_sigma = z + sigma * eps  (consistency식 의사 궤적)
  loss = MSE( D_student(x_sigma, sigma, cond), z )   sigma ~ few-step 스케줄
teacher는 학습 시 불필요 → 2.4B 1벌만 GPU 상주."""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import argparse
import glob
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
parser.add_argument("--pairs_dir", type=str, default="/home/sun4208/Geo4D/ode_pairs")
parser.add_argument("--out_ckpt", type=str, default="/home/sun4208/Geo4D/ode_init_geo4d.pt")
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--num_student_steps", type=int, default=4, help="student few-step 스케줄")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--save_every", type=int, default=100)
args_cli = parser.parse_args()

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
ckpt_path = f"{output_dir}/4dgen.ckpt"

print("[1/4] 모델 로드 (teacher 가중치로 student 초기화)")
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

# student = UNet만 학습, 나머지(조건 인코더·VAE) 동결
student = model.model  # OpenAIWrapper(UNet)
student.requires_grad_(True).train()
model.conditioner.requires_grad_(False)
n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"학습 파라미터: {n_train/1e9:.2f}B")

# bf16 통일 (Step 5 레시피)
student.to(torch.bfloat16)

print("[2/4] few-step 시그마 스케줄")
sigmas_full = model.sampler.discretization(args_cli.num_student_steps, device="cuda")
sigmas = sigmas_full[:-1]  # 마지막 0 제외
print("student sigmas:", [f"{s:.1f}" for s in sigmas.tolist()])

print("[3/4] 데이터 준비")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
pair_files = sorted(glob.glob(os.path.join(args_cli.pairs_dir, "pair_*.pt")))
print(f"ODE 쌍: {len(pair_files)}개, 데이터셋 샘플: {len(dataset)}개")

import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit([p for p in student.parameters() if p.requires_grad], lr=args_cli.lr)

cond_cache = {}
def get_cond(idx):
    """데이터셋 샘플 idx의 조건(c)과 부가 입력을 재구성 (캐시)."""
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
    c = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in c.items()}
    if len(cond_cache) < 40:  # 캐시 상한 (RAM 보호)
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
    idx, seed = d["idx"], d["seed"]
    z = d["z"].to("cuda", dtype=torch.float32)  # (BT, 8, h, w) teacher latent

    c, ami = get_cond(idx)
    g = torch.Generator(device="cuda").manual_seed(seed)
    eps = torch.randn(z.shape, device="cuda", generator=g)

    sigma = sigmas[random.randrange(len(sigmas))].expand(z.shape[0])
    x_sigma = z + sigma.view(-1, 1, 1, 1) * eps

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model.denoiser(student, x_sigma, sigma, c, **ami)
        loss = torch.nn.functional.mse_loss(pred.float(), z)

    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    loss_acc.append(loss.item())
    si = int((sigmas - sigma[0]).abs().argmin().item())
    loss_by_sigma.setdefault(si, []).append(loss.item())

    if step % args_cli.log_every == 0:
        el = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 2**30
        per = " | ".join(f"s{si}({sigmas[si]:.0f}):{sum(v)/len(v):.3f}" for si, v in sorted(loss_by_sigma.items()))
        print(f"[step {step}/{args_cli.max_steps}] loss={sum(loss_acc)/len(loss_acc):.4f} [{per}] "
              f"({el/step:.1f}초/step, peak VRAM {vram:.1f}GB)", flush=True)
        loss_acc = []
        loss_by_sigma = {}
    if step % args_cli.save_every == 0 or step == args_cli.max_steps:
        torch.save({"student": student.state_dict(), "step": step,
                    "num_student_steps": args_cli.num_student_steps}, args_cli.out_ckpt)
        print(f"체크포인트 저장: {args_cli.out_ckpt} (step {step})", flush=True)

print("완료")
