"""Step 6-1: Geo4D teacher(25스텝)로 ODE 쌍 생성
각 (데이터셋 샘플, 시드)에 대해 teacher의 최종 latent를 저장.
노이즈는 시드로 재현 가능하므로 (idx, seed, samples_z)만 저장 → 쌍당 ~0.4MB.
출력: ~/Geo4D/ode_pairs/pair_{idx:05d}_s{seed}.pt + meta.json"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import argparse
import json
import os
import time
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

parser = argparse.ArgumentParser()
parser.add_argument("--max_pairs", type=int, default=200)
parser.add_argument("--seeds", type=int, default=2, help="샘플당 노이즈 시드 수")
parser.add_argument("--num_steps", type=int, default=25, help="teacher 샘플링 스텝")
parser.add_argument("--out_dir", type=str, default="/home/sun4208/Geo4D/ode_pairs")
args_cli = parser.parse_args()
os.makedirs(args_cli.out_dir, exist_ok=True)

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
ckpt_path = f"{output_dir}/4dgen.ckpt"

print("[1/3] 모델 로드")
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
model.sampler.num_steps = args_cli.num_steps

print("[2/3] 데이터 준비")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
print(f"데이터셋 샘플 수: {len(dataset)}")

meta = {"teacher_ckpt": ckpt_path, "num_steps": args_cli.num_steps,
        "seeds_per_sample": args_cli.seeds, "sampler": type(model.sampler).__name__,
        "latent_shape": None, "note": "noise = torch.randn(shape, generator=seed) 로 재현"}

print(f"[3/3] ODE 쌍 생성 시작 (최대 {args_cli.max_pairs}쌍, 시드 {args_cli.seeds}개/샘플)")
n_saved, t0 = 0, time.time()
with torch.no_grad():
    for i, batch_old in enumerate(loader):
        if n_saved >= args_cli.max_pairs:
            break
        batch_old = dict_apply(batch_old, lambda x: x.to("cuda", non_blocking=True))
        batch_old["num_video_frames"] = batch_old["pointmap"].shape[1]

        # sample_multiview_video의 전처리를 그대로 복제 (latent만 취득)
        batch = {}
        batch.update({k: v[0:1] for (k, v) in batch_old.items()
                      if k != "num_video_frames" and torch.is_tensor(v)})
        batch.update({k: v for k, v in batch_old.items() if not torch.is_tensor(v)})
        batch["num_video_frames"] = batch_old["num_video_frames"]

        c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch)
        ami = {"num_video_frames": batch["num_video_frames"],
               "image_only_indicator": batch["image_only_indicator"]}

        def denoiser(input, sigma, cond):
            return model.denoiser(model.model, input, sigma, cond, **ami)

        mv = torch.cat([batch["pointmap"][0], batch["pointmap_right"][0]], dim=0)
        BT, C, H, W = mv.shape
        shape = (BT, 8, H // 8, W // 8)
        if meta["latent_shape"] is None:
            meta["latent_shape"] = list(shape)
            with open(os.path.join(args_cli.out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        for seed in range(args_cli.seeds):
            if n_saved >= args_cli.max_pairs:
                break
            path = os.path.join(args_cli.out_dir, f"pair_{i:05d}_s{seed}.pt")
            if os.path.exists(path):
                n_saved += 1
                continue
            g = torch.Generator(device="cuda").manual_seed(seed)
            randn = torch.randn(shape, device="cuda", generator=g)
            z = model.sampler(denoiser, randn.clone(), cond=c, uc=uc)
            torch.save({"idx": i, "seed": seed,
                        "z": z.to(torch.float16).cpu()}, path)
            n_saved += 1
            el = time.time() - t0
            print(f"[{n_saved}/{args_cli.max_pairs}] idx={i} seed={seed} "
                  f"({el/n_saved:.1f}초/쌍, 남은 예상 {(args_cli.max_pairs-n_saved)*el/n_saved/60:.0f}분)", flush=True)

print(f"완료: {n_saved}쌍 저장 → {args_cli.out_dir}")
