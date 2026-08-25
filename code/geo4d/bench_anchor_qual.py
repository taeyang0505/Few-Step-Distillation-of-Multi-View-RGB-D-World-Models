"""Figure 5 확장: 앵커 적용 전/후 깊이 오차맵 정성 비교 (양 뷰)
행: teacher@25 / student@3 앵커 없음 / student@3 앵커 / student@1 앵커 없음 / student@1 앵커
앵커는 평가와 동일한 코드 경로(enable_cond_anchor(per_view=True))로 적용하고, 같은 배치·같은 시드로 생성해
"앵커 전후"가 순수하게 앵커의 효과만 보이도록 함. 행별 AbsRel도 출력해 수치로 검증.
출력: ~/Geo4D/bench_out/anchor_qual/{err,depth,rgb}_{left,right}.png"""
import sys; sys.path.insert(0, "/home/sun4208"); sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import argparse, os
import numpy as np, torch, hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from matplotlib import cm
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps, enable_cond_anchor, disable_cond_anchor

ap = argparse.ArgumentParser()
ap.add_argument("--student_ckpt", default="/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt")
ap.add_argument("--out", default="/home/sun4208/Geo4D/bench_out/anchor_qual")
ap.add_argument("--views", nargs="+", default=["left", "right"])
ap.add_argument("--batch_idx", type=int, default=0, help="기존 Figure 5와 같은 배치를 쓰려면 0")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

unnormalize = lambda x, mn, mx: torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)
to_rgb_np = lambda v: (unnormalize(v[:, 3:], 0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)

def to_depth_np(video, dmin, dmax):
    d = unnormalize(video[:, 2], -1, 2).cpu().numpy()
    d = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    return (cm.viridis(d)[..., :3] * 255).astype(np.uint8)

def label(img, text):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0)); dr.text((4, 2), text, fill=(255, 255, 0))
    return np.array(im)

def grid(rows, names, path):
    strips = []
    for name, r in zip(names, rows):
        frames = [label(r[t].copy(), f"{name} t={t}" if t == 0 else f"t={t}") for t in range(r.shape[0])]
        strips.append(np.concatenate(frames, axis=1))
    Image.fromarray(np.concatenate(strips, axis=0)).save(path); print("저장:", path, flush=True)

print("[1/4] 모델 로드", flush=True)
output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
cfg = OmegaConf.load(f"{output_dir}/config.yaml")
for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and "desc" in cfg[key]:
        cfg[key] = cfg[key]["value"]
cls = hydra.utils.get_class(cfg._target_)
cfg.logging.mode = "offline"; cfg.model.params.ckpt_path = f"{output_dir}/4dgen.ckpt"
cfg.training.seed = 42; cfg.training.output_dir = "/home/sun4208/Geo4D/bench_out"
model = cls(cfg).lightning_module_wrapper.to("cuda"); model.eval()
euler = model.sampler
teacher_sd = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}

print("[2/4] 데이터 준비", flush=True)
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False; cfg.dataloader.batch_size = 1
loader = DataLoader(dataset, **cfg.dataloader)
for i, b in enumerate(loader):
    if i == a.batch_idx: break
ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True)); ib["num_video_frames"] = b["pointmap"].shape[1]

def gen(steps, renoise, anchor, seed=0):
    model.sampler = RenoiseSampler(sigmas_for_steps(steps)) if renoise else euler
    if not renoise: model.sampler.num_steps = steps
    enable_cond_anchor(model, per_view=True) if anchor else disable_cond_anchor(model)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model.log_images(ib)
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in out["video_dict"].items()}

print("[3/4] 생성 (teacher 25 → student 3/1, 앵커 off/on, 모두 시드 0)", flush=True)
res = {"teacher25": gen(25, False, False)}
model.model.load_state_dict(torch.load(a.student_ckpt, map_location="cpu")["student"], strict=False)
res["s3_raw"] = gen(3, True, False)
res["s3_anchor"] = gen(3, True, True)
res["s1_raw"] = gen(1, True, False)
res["s1_anchor"] = gen(1, True, True)
disable_cond_anchor(model)
ORDER = ["teacher25", "s3_raw", "s3_anchor", "s1_raw", "s1_anchor"]
NAMES = ["teacher25", "student3 raw", "student3 anchor", "student1 raw", "student1 anchor"]

print("[4/4] 그리드 저장 + 행별 AbsRel 검증", flush=True)
summary = []
for view in a.views:
    gt = res["teacher25"][f"gt_video_{view}"]
    gtd = unnormalize(gt[:, 2], -1, 2)
    d = gtd.numpy(); dmin = float(d[d > 0].min()) if (d > 0).any() else 0.0; dmax = float(d.max())

    def absrel(video):
        pd = unnormalize(video[:, 2], -1, 2)
        m = (pd > 0) & (gtd > 0)
        if view == "right":                                        # 가짜 픽셀 제외 (발견 10)
            ex = res["teacher25"].get("extra", {})
            if "cam_extr" in ex and "cam_extr_right" in ex:
                E1 = ex["cam_extr"].reshape(-1, 4, 4)[0].float(); E2 = ex["cam_extr_right"].reshape(-1, 4, 4)[0].float()
                t = (torch.linalg.inv(E1) @ E2)[:3, 3].view(1, 3, 1, 1)
                gxyz = unnormalize(gt[:, :3], -1, 2)
                m = m & ~((gxyz - t).abs().amax(dim=1) < 2e-3)
        return float(((pd - gtd).abs() / gtd.clamp_min(1e-3))[m].mean()), m

    def err_np(video, m):
        pd = unnormalize(video[:, 2], -1, 2)
        e = torch.where(m, (pd - gtd).abs() / gtd.clamp_min(1e-3), torch.zeros_like(pd)).clamp(0, 0.5) / 0.5
        out = (cm.magma(e.numpy())[..., :3] * 255).astype(np.uint8); out[~m.numpy()] = 40
        return out

    err_rows, dep_rows, rgb_rows = [], [], []
    for k in ORDER:
        v = res[k][f"sampled_video_{view}"]
        ar, m = absrel(v)
        summary.append((view, k, ar))
        print(f"  [{view}] {k}: AbsRel {ar:.4f}", flush=True)
        err_rows.append(err_np(v, m)); dep_rows.append(to_depth_np(v, dmin, dmax)); rgb_rows.append(to_rgb_np(v))
    grid(err_rows, NAMES, f"{a.out}/err_{view}.png")
    grid([to_depth_np(gt, dmin, dmax)] + dep_rows, ["GT"] + NAMES, f"{a.out}/depth_{view}.png")
    grid([to_rgb_np(gt)] + rgb_rows, ["GT"] + NAMES, f"{a.out}/rgb_{view}.png")

with open(f"{a.out}/absrel_rows.txt", "w") as f:
    f.write("view\trow\tAbsRel (this single sample; 20-sample means differ)\n")
    for v, k, ar in summary: f.write(f"{v}\t{k}\t{ar:.4f}\n")
print("\n검증 기준: student3 raw >> student3 anchor 이어야 하고 anchor 행은 teacher 근처여야 함 "
      "(20샘플 평균 기준 0.175 -> 0.082 vs teacher 0.072). RGB는 앵커가 건드리지 않으므로 raw/anchor 행이 동일해야 정상.")
print("ANCHOR_QUAL_DONE")
