"""6-2 정성 검증: GT / teacher@25 / student@4 / student@1 그리드 + student@1 시드 4개 다양성
퇴화(블러) 가설: student 행이 뭉개져 있고, 시드 4개가 거의 동일해야 가설 적중.
출력: ~/Geo4D/bench_out/student_qual/"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
import argparse
from common import transformers_pre_import_mods  # isort:skip
import os
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from matplotlib import cm
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace

from geo4d_fewstep import RenoiseSampler, sigmas_for_steps
ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/ode_init_geo4d.pt")
ap.add_argument("--sampler", choices=["euler", "renoise"], default="euler")
ap.add_argument("--out", default="/home/sun4208/Geo4D/bench_out/student_qual")
ap.add_argument("--full_steps", type=int, default=4, help="student full-step 수 (renoise면 3)")
ap.add_argument("--label", default="student", help="그리드 행 이름")
ap.add_argument("--views", nargs="+", default=["left"])
a = ap.parse_args()
OUT = a.out
os.makedirs(OUT, exist_ok=True)
STUDENT_CKPT = a.student_ckpt


def unnormalize(x, mn, mx):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)


def to_rgb_np(video):
    v = (unnormalize(video[:, 3:], 0, 1) * 255).byte().cpu().numpy()
    return v.transpose(0, 2, 3, 1)


def to_depth_np(video, dmin, dmax):
    d = unnormalize(video[:, 2], -1, 2).cpu().numpy()
    d = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    return (cm.viridis(d)[..., :3] * 255).astype(np.uint8)


def label(img, text):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    dr.text((4, 2), text, fill=(255, 255, 0))
    return np.array(im)


def grid(rows, names, path, sel_t=None):
    strips = []
    for name, r in zip(names, rows):
        ts = sel_t if sel_t is not None else range(r.shape[0])
        frames = [label(r[t].copy(), f"{name} t={t}" if j == 0 else f"t={t}")
                  for j, t in enumerate(ts)]
        strips.append(np.concatenate(frames, axis=1))
    Image.fromarray(np.concatenate(strips, axis=0)).save(path)
    print("저장:", path)


print("[1/4] 모델 로드 (teacher)")
output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
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


euler_sampler = model.sampler


def gen(steps, seed, student=False):
    if student and a.sampler == "renoise":
        model.sampler = RenoiseSampler(sigmas_for_steps(steps))
    else:
        model.sampler = euler_sampler
        model.sampler.num_steps = steps
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model.log_images(ib)
    return {k: v.cpu() for k, v in out["video_dict"].items() if torch.is_tensor(v)}


print("[3/4] 생성: teacher@25 → student 가중치 교체 → student@4/@1 + 시드 4개")
res = {"teacher25": gen(25, 0)}
if STUDENT_CKPT != "none":
    sd = torch.load(STUDENT_CKPT, map_location="cpu")
    model.model.load_state_dict(sd["student"], strict=False)
    print("student 가중치 주입 완료 (step", sd["step"], ")")
else:
    print("teacher 가중치 유지 (학습 없는 few-step 비교)")
res["student4"] = gen(a.full_steps, 0, student=True)
res["student1"] = gen(1, 0, student=True)
div = {s: gen(1, s, student=True) for s in [0, 1, 2, 3]}

print("[4/4] 그리드 저장")
ref = res["teacher25"]
for view in a.views:
    gt = ref[f"gt_video_{view}"]
    d = unnormalize(gt[:, 2], -1, 2).numpy()
    dmin, dmax = float(d[d > 0].min()) if (d > 0).any() else 0.0, float(d.max())
    names = ["GT", "teacher25", f"{a.label}{a.full_steps}", f"{a.label}1"]
    rgb_rows = [to_rgb_np(gt)] + [to_rgb_np(res[k][f"sampled_video_{view}"])
                                  for k in ["teacher25", "student4", "student1"]]
    grid(rgb_rows, names, f"{OUT}/rgb_{view}.png")
    dep_rows = [to_depth_np(gt, dmin, dmax)] + [to_depth_np(res[k][f"sampled_video_{view}"], dmin, dmax)
                                                for k in ["teacher25", "student4", "student1"]]
    grid(dep_rows, names, f"{OUT}/depth_{view}.png")
    gtd = unnormalize(gt[:, 2], -1, 2)
    def err_np(video):
        pd = unnormalize(video[:, 2], -1, 2)
        m = (pd > 0) & (gtd > 0)
        e = torch.where(m, (pd - gtd).abs() / gtd.clamp_min(1e-3), torch.zeros_like(pd)).clamp(0, 0.5) / 0.5
        out = (cm.magma(e.numpy())[..., :3] * 255).astype(np.uint8)
        out[~m.numpy()] = 40
        return out
    err_rows = [err_np(res[k][f"sampled_video_{view}"]) for k in ["teacher25", "student4", "student1"]]
    for k, r in zip(["teacher25", "student4", "student1"], err_rows):
        pd = unnormalize(res[k][f"sampled_video_{view}"][:, 2], -1, 2); m = (pd > 0) & (gtd > 0)
        print(f"  [{view}] {k}: AbsRel {((pd - gtd).abs() / gtd.clamp_min(1e-3))[m].mean():.3f} | 프레임별", [f"{((pd[t]-gtd[t]).abs()/gtd[t].clamp_min(1e-3))[m[t]].mean():.2f}" for t in range(pd.shape[0])])
    grid(err_rows, names[1:], f"{OUT}/err_{view}.png")

div_rows = [to_rgb_np(div[s]["sampled_video_left"])[[0, 5, 9]] for s in [0, 1, 2, 3]]
grid(div_rows, [f"{a.label}1 s{s}" for s in [0, 1, 2, 3]], f"{OUT}/diversity_{a.label}1.png", sel_t=None)
print("완료 — 출력:", OUT)
