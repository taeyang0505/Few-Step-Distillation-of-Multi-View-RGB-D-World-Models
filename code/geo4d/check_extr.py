"""외부 파라미터(cam_extr, cam_extr_right)가 프레임별로 변하는지 검사 (CPU, ~1분)
변하면: 오른쪽 앵커 변환에 프레임 대응 파라미터를 써야 함 (inv(E1[1]) @ E2[0] 등)"""
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
for i in range(3):
    b = ds[i]
    E1 = b["cam_extr"].reshape(-1, 4, 4); E2 = b["cam_extr_right"].reshape(-1, 4, 4)
    d1 = (E1 - E1[0:1]).abs().amax(dim=(1, 2)); d2 = (E2 - E2[0:1]).abs().amax(dim=(1, 2))
    print(f"샘플 {i}: 프레임 수 {E1.shape[0]} | 왼쪽 카메라 프레임별 최대 변화 {[round(x,4) for x in d1.tolist()]}")
    print(f"         오른쪽 카메라 프레임별 최대 변화 {[round(x,4) for x in d2.tolist()]}")
    T0 = torch.linalg.inv(E1[0]) @ E2[0]; T1 = torch.linalg.inv(E1[1]) @ E2[0]
    print(f"         inv(E1[0])@E2[0] 이동(m) {[round(x,3) for x in T0[:3,3].tolist()]} | inv(E1[1])@E2[0] 이동 {[round(x,3) for x in T1[:3,3].tolist()]}")
print("판정: 변화가 전부 0.0이면 카메라 고정(앵커 변환 문제 아님) / 0이 아니면 프레임 대응 파라미터 필요 → 2번 명령 실행")
