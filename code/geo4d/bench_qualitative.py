"""정성 비교: GT vs 25/8/1스텝 생성 비디오 저장 (RGB·깊이·양 뷰) + 시드 다양성 그리드 + GIF
출력: ~/Geo4D/bench_out/qualitative/"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import os
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import imageio
from matplotlib import cm
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
ckpt_path = f"{output_dir}/4dgen.ckpt"
OUT = "/home/sun4208/Geo4D/bench_out/qualitative"
os.makedirs(OUT, exist_ok=True)
STEPS_LIST = [25, 8, 1]
DIV_SEEDS = [0, 1, 2, 3]


def unnormalize(x, mn, mx):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)


def to_rgb_np(video):
    """(T,3,H,W)[0,1] → (T,H,W,3) uint8"""
    v = (unnormalize(video[:, 3:], 0, 1) * 255).byte().cpu().numpy()
    return v.transpose(0, 2, 3, 1)


def to_depth_np(video, dmin, dmax):
    """포인트맵 z채널 → viridis 컬러맵 uint8"""
    d = unnormalize(video[:, 2], -1, 2).cpu().numpy()  # (T,H,W)
    d = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    return (cm.viridis(d)[..., :3] * 255).astype(np.uint8)  # (T,H,W,3)


def label(img, text):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    dr.text((4, 2), text, fill=(255, 255, 0))
    return np.array(im)


def grid(rows, row_names, path):
    """rows: list of (T,H,W,3) — 세로로 행, 가로로 프레임을 이어붙인 그리드 저장"""
    strips = []
    for name, r in zip(row_names, rows):
        frames = [label(r[t].copy(), f"{name} t={t}" if t == 0 else f"t={t}") for t in range(r.shape[0])]
        strips.append(np.concatenate(frames, axis=1))
    Image.fromarray(np.concatenate(strips, axis=0)).save(path)
    print("저장:", path)


print("[1/4] 모델 로드")
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

print("[2/4] 데이터 준비 (배치 1개)")
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False
cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
b = next(iter(loader))
n = b["pointmap"].shape[1]
ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True))
ib["num_video_frames"] = n

print("[3/4] 생성: GT + 스텝별(시드 0) + 다양성용(25·1스텝 × 시드 4)")
results = {}   # (steps, seed) -> outputs 텐서 dict (cpu)
def gen(steps, seed):
    sampler.num_steps = steps
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model.log_images(ib)
    return {k: v.cpu() for k, v in out["video_dict"].items() if torch.is_tensor(v)}

for steps in STEPS_LIST:
    results[(steps, 0)] = gen(steps, 0)
    print(f"  steps={steps} seed=0 완료")
for seed in DIV_SEEDS[1:]:
    for steps in [25, 1]:
        results[(steps, seed)] = gen(steps, seed)
    print(f"  다양성 seed={seed} 완료")

print("[4/4] 이미지·GIF 저장")
ref = results[(25, 0)]
for view in ["left", "right"]:
    gt = ref[f"gt_video_{view}"]
    # 깊이 스케일: GT 기준 공통 정규화
    d = unnormalize(gt[:, 2], -1, 2).numpy()
    dmin, dmax = float(d[d > 0].min()) if (d > 0).any() else 0.0, float(d.max())

    rgb_rows = [to_rgb_np(gt)] + [to_rgb_np(results[(s, 0)][f"sampled_video_{view}"]) for s in STEPS_LIST]
    names = ["GT"] + [f"{s}step" for s in STEPS_LIST]
    grid(rgb_rows, names, f"{OUT}/rgb_{view}.png")

    dep_rows = [to_depth_np(gt, dmin, dmax)] + [to_depth_np(results[(s, 0)][f"sampled_video_{view}"], dmin, dmax) for s in STEPS_LIST]
    grid(dep_rows, names, f"{OUT}/depth_{view}.png")

    # GIF: 프레임별로 [GT | 25 | 8 | 1] 가로 결합
    frames = []
    for t in range(gt.shape[0]):
        panels = [label(r[t].copy(), nm) for nm, r in zip(names, rgb_rows)]
        frames.append(np.concatenate(panels, axis=1))
    imageio.mimsave(f"{OUT}/compare_{view}.gif", frames, fps=3, loop=0)
    print("저장:", f"{OUT}/compare_{view}.gif")

# 다양성 그리드: 행 = 25스텝 시드0~3, 1스텝 시드0~3 / 열 = 프레임 0·5·9 (left 뷰 RGB)
sel_t = [0, 5, 9]
div_rows, div_names = [], []
for steps in [25, 1]:
    for seed in DIV_SEEDS:
        v = to_rgb_np(results[(steps, seed)]["sampled_video_left"])
        div_rows.append(v[sel_t])
        div_names.append(f"{steps}step s{seed}")
grid(div_rows, div_names, f"{OUT}/diversity_left.png")

print()
print("완료 — 출력 폴더:", OUT)
print("보는 법: rgb_*.png(행: GT/25/8/1스텝, 열: 프레임) — 1스텝 행의 블러 확인.")
print("        diversity_left.png — 25스텝 4행은 서로 달라야 하고, 1스텝 4행은 거의 동일해야 함(평균 수렴).")
