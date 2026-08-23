"""오른쪽 조건 포인트맵을 GT 오른쪽 프레임0(참조 좌표계)에 맞추는 올바른 변환 찾기 (CPU, 데이터만)"""
import sys; sys.path.insert(0, "/home/sun4208/4dgen")
from common import transformers_pre_import_mods  # isort:skip
import random, numpy as np, torch, hydra
from omegaconf import OmegaConf
output_dir = "/home/sun4208/Geo4D/checkpoints/checkpoints/outputs/apple"
cfg = OmegaConf.load(f"{output_dir}/config.yaml")
for key in cfg:
    if OmegaConf.is_dict(cfg[key]) and "desc" in cfg[key]:
        cfg[key] = cfg[key]["value"]
cfg.task = OmegaConf.load("/home/sun4208/4dgen/config/task/inference.yaml")
ds = hydra.utils.instantiate(cfg.task.dataset)
random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
un = lambda x: torch.clamp(((x + 1.) / 2.) * 3.0 - 1.0, -1.0, 2.0)
def tf(pm, T):
    C, H, W = pm.shape; pts = pm.reshape(3, -1); hom = torch.cat([pts, torch.ones(1, H*W)], 0)
    return (T @ hom)[:3].reshape(3, H, W)
def absrel(p, g, m): return (torch.abs(p[m]-g[m])/g[m]).mean().item()
def stats(cond_raw, gt0, T, name, extra=""):
    valid = cond_raw[2] > 0
    c = tf(cond_raw, T) if T is not None else cond_raw
    m = valid & (c[2] > 0) & (gt0[2] > 0)
    xyz = (c[:, m] - gt0[:, m]).abs().mean(1)
    print(f"   {name:<26} depth AbsRel {absrel(c[2], gt0[2], m):.4f} | |Δx|,|Δy|,|Δz| 평균 {[round(v,3) for v in xyz.tolist()]} | 유효픽셀 {m.float().mean():.2f}{extra}")
for i in range(3):
    b = ds[i]
    E1 = b["cam_extr"].reshape(-1,4,4)[0].float(); E2 = b["cam_extr_right"].reshape(-1,4,4)[0].float()
    gtL0 = un(b["pointmap"][0, :3]); gtR0 = un(b["pointmap_right"][0, :3])
    cL = un(b["cond_pointmaps_without_noise"].reshape(-1, *b["cond_pointmaps_without_noise"].shape[-3:])[-1][:3])
    cR = un(b["cond_pointmaps_without_noise_right"].reshape(-1, *b["cond_pointmaps_without_noise_right"].shape[-3:])[-1][:3])
    print(f"샘플 {i}:")
    stats(cL, gtL0, None, "왼쪽 조건 vs GT0 (기준)")
    stats(cR, gtR0, None, "오른쪽: 변환 없음")
    stats(cR, gtR0, torch.linalg.inv(E1) @ E2, "오른쪽: inv(E1)@E2 (현재)")
    stats(cR, gtR0, E1 @ torch.linalg.inv(E2), "오른쪽: E1@inv(E2)")
    stats(cR, gtR0, torch.linalg.inv(E1), "오른쪽: inv(E1) (raw=world?)")
    stats(cR, gtR0, E1, "오른쪽: E1")
    # 조건 프레임 시간차 확인: 오른쪽 GT 프레임0 vs 프레임1 차이
    gtR1 = un(b["pointmap_right"][1, :3]); m = (gtR0[2] > 0) & (gtR1[2] > 0)
    print(f"   (참고) GT 오른쪽 프레임0 vs 프레임1 depth AbsRel {absrel(gtR1[2], gtR0[2], m):.4f}  | GT 오른쪽 z 범위 {gtR0[2][gtR0[2]>0].min():.2f}~{gtR0[2].max():.2f}, z≈이동벡터 픽셀 비율 {((gtR0[2]-(torch.linalg.inv(E1)@E2)[2,3]).abs()<1e-3).float().mean():.3f}")
