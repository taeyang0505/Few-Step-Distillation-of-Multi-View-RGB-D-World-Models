"""Step 6-1 v2: teacher(25스텝) 궤적의 '실제 중간 상태'까지 저장
denoiser 래퍼로 매 스텝의 (sigma, x)를 관찰, student σ(700/70.5/2.3)에 가장 가까운
지점의 x를 그대로 캡처 → v1의 의사 궤적(z+σε) 근사를 제거.
출력: ~/Geo4D/ode_pairs_v2/pair_{idx:05d}_s{seed}.pt  (z + traj 3점, ~1.6MB/쌍)"""
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
parser.add_argument("--seeds", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=25)
parser.add_argument("--out_dir", type=str, default="/home/sun4208/Geo4D/ode_pairs_v2")
args_cli = parser.parse_args()
os.makedirs(args_cli.out_dir, exist_ok=True)

output_dir = os.environ.get("GEO4D_TEACHER_DIR", "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple")   # 태스크 전환: 환경변수
print("[1/3] 모델 로드")
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
model.sampler.num_steps = args_cli.num_steps

# student σ 타깃: 4스텝 스케줄에서 σ≈0 제외한 3개
target_sigmas = [float(s) for s in model.sampler.discretization(4, device="cuda")[:3].tolist()]
print("student σ 타깃:", [f"{s:.1f}" for s in target_sigmas])

print("[2/3] 데이터 준비")
cfg.task = OmegaConf.load(os.environ.get("GEO4D_TASK_YAML", "/home/sun4208/4dgen/config/task/inference.yaml"))
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)

meta = {"teacher_num_steps": args_cli.num_steps, "target_sigmas": target_sigmas,
        "seeds_per_sample": args_cli.seeds,
        "note": "traj[sigma] = teacher 25스텝 경로에서 해당 sigma에 가장 가까운 지점의 실제 x (denoiser 입력)"}
with open(os.path.join(args_cli.out_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("[3/3] 궤적 캡처 생성 시작")
n_saved, t0 = 0, time.time()
with torch.no_grad():
    for i, batch_old in enumerate(loader):
        if n_saved >= args_cli.max_pairs:
            break
        batch_old = dict_apply(batch_old, lambda x: x.to("cuda", non_blocking=True))
        batch_old["num_video_frames"] = batch_old["pointmap"].shape[1]
        batch = {}
        batch.update({k: v[0:1] for (k, v) in batch_old.items()
                      if k != "num_video_frames" and torch.is_tensor(v)})
        batch.update({k: v for k, v in batch_old.items() if not torch.is_tensor(v)})
        batch["num_video_frames"] = batch_old["num_video_frames"]

        c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch)
        ami = {"num_video_frames": batch["num_video_frames"],
               "image_only_indicator": batch["image_only_indicator"]}

        mv = torch.cat([batch["pointmap"][0], batch["pointmap_right"][0]], dim=0)
        BT, C, H, W = mv.shape
        shape = (BT, 8, H // 8, W // 8)

        for seed in range(args_cli.seeds):
            if n_saved >= args_cli.max_pairs:
                break
            path = os.path.join(args_cli.out_dir, f"pair_{i:05d}_s{seed}.pt")
            if os.path.exists(path):
                n_saved += 1
                continue

            # 궤적 관찰용 denoiser 래퍼: 각 타깃 σ에 가장 가까운 스텝의 입력 x를 캡처
            best = {s: (float("inf"), None) for s in target_sigmas}

            def denoiser(input, sigma, cond):
                sv = float(sigma.max().item())
                for ts in target_sigmas:
                    gap = abs(sv - ts) / max(ts, 1e-6)
                    if gap < best[ts][0]:
                        best[ts] = (gap, (sv, input.detach().to(torch.float16).cpu()))
                return model.denoiser(model.model, input, sigma, cond, **ami)

            g = torch.Generator(device="cuda").manual_seed(seed)
            randn = torch.randn(shape, device="cuda", generator=g)
            z = model.sampler(denoiser, randn.clone(), cond=c, uc=uc)

            traj = {f"{ts:.4f}": {"sigma_actual": best[ts][1][0], "x": best[ts][1][1]}
                    for ts in target_sigmas}
            torch.save({"idx": i, "seed": seed,
                        "z": z.to(torch.float16).cpu(), "traj": traj}, path)
            n_saved += 1
            el = time.time() - t0
            print(f"[{n_saved}/{args_cli.max_pairs}] idx={i} seed={seed} "
                  f"({el/n_saved:.1f}초/쌍, 남은 예상 {(args_cli.max_pairs-n_saved)*el/n_saved/60:.0f}분)", flush=True)

print(f"완료: {n_saved}쌍 → {args_cli.out_dir}")
