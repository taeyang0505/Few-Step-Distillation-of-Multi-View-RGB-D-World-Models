"""student(DMD step1600, 재노이징 3/1스텝) 추론 시간 컴포넌트 실측 — 총시간 분해 검증용
타이머: UNet forward(호출 수 포함) / conditioner / VAE 디코딩 / VAE 인코딩 / shared_step(평가용 loss) / log_images 총합"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen"); sys.path.insert(0, "/home/sun4208/4dgen/notebooks")
from common import transformers_pre_import_mods  # isort:skip
import time, random, numpy as np, torch, hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from video_common.pytorch_util import dict_apply
from dataset.spartan_video_dataset import SpartanVideoMultiViewDataset
from workspace.base_workspace import BaseWorkspace
from geo4d_fewstep import RenoiseSampler, sigmas_for_steps

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
student_sd = torch.load("/home/sun4208/Geo4D/dmd_6a/dmd_gen_step1600.pt", map_location="cpu")["student"]

cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
dataset = hydra.utils.instantiate(cfg.task.dataset)
cfg.dataloader.shuffle = False; cfg.dataloader.batch_size = 1; cfg.dataloader.num_workers = 0; cfg.dataloader.persistent_workers = False
random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
batches = []
for i, b in enumerate(DataLoader(dataset, **cfg.dataloader)):
    if i >= 3: break
    ib = dict_apply(b, lambda x: x.to("cuda", non_blocking=True)); ib["num_video_frames"] = b["pointmap"].shape[1]; batches.append(ib)

stats = {}
def timed(name, fn):
    s = stats.setdefault(name, [0.0, 0])
    def w(*a, **k):
        torch.cuda.synchronize(); t0 = time.time()
        out = fn(*a, **k)
        torch.cuda.synchronize(); s[0] += time.time() - t0; s[1] += 1
        return out
    return w
model.model.forward = timed("UNet forward", model.model.forward)
model.conditioner.forward = timed("conditioner", model.conditioner.forward)
model.decode_first_stage = timed("VAE decode", model.decode_first_stage)
model.encode_first_stage = timed("VAE encode", model.encode_first_stage)
model.shared_step = timed("shared_step(평가용 loss)", model.shared_step)

def run(label, who, sampler, n=3):
    model.model.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
    model.sampler = sampler
    with torch.no_grad(): model.log_images(batches[0])          # 워밍업
    for v in stats.values(): v[0] = 0.0; v[1] = 0
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for ib in batches[:n]: model.log_images(ib)
    torch.cuda.synchronize(); total = (time.time() - t0) / n
    comp = {k: (v[0] / n, v[1] / n) for k, v in stats.items()}
    print(f"\n[{label}] 총 {total:.2f}초/생성")
    for k, (t, c) in comp.items():
        print(f"   {k:<24} {t:6.2f}초  (호출 {c:.0f}회, 호출당 {t/max(c,1e-9):.3f}초)")
    known = comp["UNet forward"][0] + comp["conditioner"][0] + comp["VAE decode"][0] + comp["VAE encode"][0]
    print(f"   합(UNet+cond+VAE) {known:.2f}초 | 기타(데이터 이동·지표·메트릭 등) {total-known:.2f}초 | ※ shared_step 안의 UNet 1회·VAE 인코딩은 위 항목에 이미 포함")

run("teacher Euler 25스텝", "teacher", (setattr(euler, "num_steps", 25) or euler))
run("student 재노이징 3스텝", "student", RenoiseSampler(sigmas_for_steps(3)))
run("student 재노이징 1스텝", "student", RenoiseSampler(sigmas_for_steps(1)))


# ───────── 순수 추론 경로 실측: conditioner(c,uc) 1회 → 샘플러 → 생성물 VAE 디코딩만 (평가 코드 제거) ─────────
def pure_infer(batch_old):
    batch = {k: v[0:1] for k, v in batch_old.items() if k != "num_video_frames" and torch.is_tensor(v)}
    batch.update({k: v for k, v in batch_old.items() if not torch.is_tensor(v)})
    batch["num_video_frames"] = batch_old["num_video_frames"]
    c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch)
    ami = {"num_video_frames": batch["num_video_frames"], "image_only_indicator": batch["image_only_indicator"]}
    def denoiser(x, sigma, cc): return model.denoiser(model.model, x, sigma, cc, **ami)
    mv = torch.cat([batch["pointmap"][0], batch["pointmap_right"][0]], dim=0)
    BT, C, H, W = mv.shape
    randn = torch.randn((BT, 8, H // 8, W // 8), device="cuda")
    z = model.sampler(denoiser, randn, cond=c, uc=uc)
    return model.decode_first_stage(z)

def run_pure(label, who, sampler, n=3):
    model.model.load_state_dict(teacher_sd if who == "teacher" else student_sd, strict=False)
    model.sampler = sampler
    with torch.no_grad(): pure_infer(batches[0])
    for v in stats.values(): v[0] = 0.0; v[1] = 0
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for ib in batches[:n]: pure_infer(ib)
    torch.cuda.synchronize(); total = (time.time() - t0) / n
    print(f"\n[순수 추론: {label}] 총 {total:.2f}초/생성 (10프레임×2뷰 RGB-D)")
    for k, v in stats.items():
        if v[1] > 0: print(f"   {k:<24} {v[0]/n:6.2f}초  (호출 {v[1]/n:.0f}회)")

print("\n================ 평가 코드 제거한 순수 추론 ================")
run_pure("teacher Euler 25스텝", "teacher", (setattr(euler, "num_steps", 25) or euler))
run_pure("teacher Euler 4스텝", "teacher", (setattr(euler, "num_steps", 4) or euler))
run_pure("student 재노이징 3스텝", "student", RenoiseSampler(sigmas_for_steps(3)))
run_pure("student 재노이징 1스텝", "student", RenoiseSampler(sigmas_for_steps(1)))


# ───────── Step 7-①: student는 CFG 미사용 → uc 계산 생략 (conditioner 1회) 실측 ─────────
def pure_infer_cond_only(batch_old):
    batch = {k: v[0:1] for k, v in batch_old.items() if k != "num_video_frames" and torch.is_tensor(v)}
    batch.update({k: v for k, v in batch_old.items() if not torch.is_tensor(v)})
    batch["num_video_frames"] = batch_old["num_video_frames"]
    c = model.conditioner(batch)                                   # uc 없이 c만
    ami = {"num_video_frames": batch["num_video_frames"], "image_only_indicator": batch["image_only_indicator"]}
    def denoiser(x, sigma, cc): return model.denoiser(model.model, x, sigma, cc, **ami)
    mv = torch.cat([batch["pointmap"][0], batch["pointmap_right"][0]], dim=0)
    BT, C, H, W = mv.shape
    randn = torch.randn((BT, 8, H // 8, W // 8), device="cuda")
    z = model.sampler(denoiser, randn, cond=c, uc=None)
    return model.decode_first_stage(z)

def run_pure2(label, sampler, n=3):
    model.model.load_state_dict(student_sd, strict=False); model.sampler = sampler
    with torch.no_grad(): pure_infer_cond_only(batches[0])
    for v in stats.values(): v[0] = 0.0; v[1] = 0
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for ib in batches[:n]: pure_infer_cond_only(ib)
    torch.cuda.synchronize(); total = (time.time() - t0) / n
    print(f"\n[순수 추론, uc 생략: {label}] 총 {total:.2f}초/생성")
    for k, v in stats.items():
        if v[1] > 0: print(f"   {k:<24} {v[0]/n:6.2f}초  (호출 {v[1]/n:.0f}회)")

print("\n================ Step 7-①: uc 생략 ================")
run_pure2("student 재노이징 3스텝", RenoiseSampler(sigmas_for_steps(3)))
run_pure2("student 재노이징 1스텝", RenoiseSampler(sigmas_for_steps(1)))
