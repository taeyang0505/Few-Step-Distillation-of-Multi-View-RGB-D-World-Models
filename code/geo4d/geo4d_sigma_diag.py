"""6-2 진단: σ별 회귀 loss를 학습 전(teacher 초기화) vs 학습 후(student)로 비교
재학습 없이 forward만으로, 어느 σ 과제가 개선/악화됐는지 확정."""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import argparse
import glob
import os
import torch
import hydra
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

parser = argparse.ArgumentParser()
parser.add_argument("--pairs_dir", type=str, default="/home/sun4208/Geo4D/ode_pairs")
parser.add_argument("--student_ckpt", type=str, default="/home/sun4208/Geo4D/ode_init_geo4d.pt")
parser.add_argument("--n_pairs", type=int, default=40)
args_cli = parser.parse_args()

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
print("[1/3] 모델 로드 (teacher 가중치)")
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
teacher_sd = {k: v.clone() for k, v in model.model.state_dict().items()}

sigmas = model.sampler.discretization(4, device="cuda")[:-1]
print("sigmas:", [f"{s:.1f}" for s in sigmas.tolist()])

print("[2/3] 데이터 준비")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
pair_files = sorted(glob.glob(os.path.join(args_cli.pairs_dir, "pair_*.pt")))[:args_cli.n_pairs]

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


def eval_per_sigma(tag):
    per = {i: [] for i in range(len(sigmas))}
    with torch.no_grad():
        for f in pair_files:
            d = torch.load(f, map_location="cpu")
            z = d["z"].to("cuda", dtype=torch.float32)
            c, ami = get_cond(d["idx"])
            g = torch.Generator(device="cuda").manual_seed(d["seed"])
            eps = torch.randn(z.shape, device="cuda", generator=g)
            for i, s in enumerate(sigmas):
                sigma = s.expand(z.shape[0])
                x = z + sigma.view(-1, 1, 1, 1) * eps
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model.denoiser(model.model, x, sigma, c, **ami)
                per[i].append(torch.nn.functional.mse_loss(pred.float(), z).item())
    print(f"--- {tag} ---")
    for i, s in enumerate(sigmas):
        print(f"  sigma={s.item():7.1f}: loss {np.mean(per[i]):.4f} ± {np.std(per[i]):.4f}")
    return {float(s): float(np.mean(per[i])) for i, s in enumerate(sigmas)}


print(f"[3/3] σ별 loss 비교 ({len(pair_files)}쌍)")
before = eval_per_sigma("학습 전 (teacher 초기화 그대로)")
sd = torch.load(args_cli.student_ckpt, map_location="cpu")
model.model.load_state_dict(sd["student"], strict=False)
after = eval_per_sigma(f"학습 후 (student, step {sd['step']})")

print()
print("=== σ별 변화 (음수 = 개선) ===")
for s in before:
    print(f"  sigma={s:7.1f}: {before[s]:.4f} -> {after[s]:.4f}  ({100*(after[s]-before[s])/before[s]:+.1f}%)")
