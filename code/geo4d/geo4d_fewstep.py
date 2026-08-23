"""Geo4D few-step 공용 유틸 (6-3 DMD 트레이너와 bench_student_sweep가 공유)

RenoiseSampler: DMD/CausVid 계열 student의 추론 방식.
  각 스텝에서 x0를 예측하고, 다음 σ로 '새 노이즈'를 더해 재노이징 (Euler 적분 아님).
  학습(backward simulation)과 동일한 절차여야 하므로 모듈로 분리.
"""
import math
import torch

# 4스텝 EDM 스케줄(σ_max 700, ρ=7)에서 σ≈0 슬롯을 제외한 3개 — ode_pairs_v2/meta.json과 동일
DEFAULT_SIGMAS = [700.0001220703125, 70.54086303710938, 2.2691192626953125]


def sigmas_for_steps(n_steps, base=DEFAULT_SIGMAS):
    """n_steps개 σ 스케줄. 3→[700,70.5,2.3], 2→[700,2.3], 1→[700]"""
    if n_steps >= len(base):
        return list(base)
    if n_steps == 1:
        return [base[0]]
    return [base[0], base[-1]]


class RenoiseSampler:
    """sgm sampler와 같은 호출 규약: sampler(denoiser, randn, cond=c, uc=uc) -> x0 latent
    denoiser(input, sigma, c) 클로저를 받음. uc는 무시(student는 CFG 없이 cond-only)."""

    def __init__(self, sigmas, device="cuda"):
        self.sigmas = [float(s) for s in sigmas]
        self.num_steps = len(self.sigmas)
        self.device = device

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, generator=None):
        x = x * math.sqrt(1.0 + self.sigmas[0] ** 2)  # sgm prepare_sampling_loop와 동일한 초기 스케일
        x0 = None
        for i, s in enumerate(self.sigmas):
            sigma = x.new_full((x.shape[0],), s)
            x0 = denoiser(x, sigma, cond)
            if i < self.num_steps - 1:
                noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
                x = x0 + self.sigmas[i + 1] * noise
        return x0


# ───────────── 6-4 (a): 조건 프레임 앵커 스케일 보정 (추론 시, 학습·GT 불필요) ─────────────
# 근거(08-22 실험): 왼쪽 조건 포인트맵 ≈ GT 프레임 0(AbsRel 0.013, 참조 좌표계). student는 깊이를 일관되게 ~9% 멀게
# 예측하므로 프레임 0을 조건에 맞춘 스칼라 s 하나를 전 프레임·양 뷰(모두 참조 좌표계)의 xyz에 적용.
PM_MIN, PM_MAX = -1.0, 2.0   # 포인트맵 정규화 범위 (unnormalize(-1, 2))


def _unnorm(x, mn=PM_MIN, mx=PM_MAX):
    return torch.clamp(((x + 1.) / 2.) * (mx - mn) + mn, mn, mx)


def _renorm(x, mn=PM_MIN, mx=PM_MAX):
    return torch.clamp((x - mn) / (mx - mn) * 2. - 1., -1., 1.)


def cond_anchor_scale(video_dict, ref_view="left"):
    """s = median(cond_depth / pred_depth[frame 0]) — ref_view(참조 좌표계) 기준"""
    pred = _unnorm(video_dict[f"sampled_video_{ref_view}"][0, 2])          # (H, W) 프레임 0 깊이
    cond = video_dict[f"cond_pointmap_{ref_view}"]
    cond = cond.reshape(-1, *cond.shape[-3:])[-1]                           # 마지막 조건 프레임 (3, H, W)
    cd = _unnorm(cond[2])
    if cd.shape != pred.shape:
        cd = torch.nn.functional.interpolate(cd[None, None], size=pred.shape, mode="nearest")[0, 0]
    m = (pred > 0) & (cd > 0)
    if m.sum() < 100:
        return 1.0
    return torch.median(cd[m] / pred[m]).item()


def apply_cond_anchor(video_dict, ref_view="left", views=("left", "right")):
    """양 뷰 포인트맵(xyz) 전 프레임에 s 적용. 반환: s"""
    s = cond_anchor_scale(video_dict, ref_view)
    for v in views:
        vid = video_dict[f"sampled_video_{v}"]
        xyz = _unnorm(vid[:, :3]) * s
        vid = vid.clone()
        vid[:, :3] = _renorm(xyz)
        video_dict[f"sampled_video_{v}"] = vid
    video_dict["anchor_scale"] = s
    return s


def enable_cond_anchor(model):
    """model.sample_multiview_video를 감싸 출력에 앵커 보정 적용 (한 번만 호출)"""
    if getattr(model, "_cond_anchor_enabled", False):
        return
    orig = model.sample_multiview_video

    def wrapped(batch):
        vd = orig(batch)
        apply_cond_anchor(vd)
        return vd
    model.sample_multiview_video = wrapped
    model._cond_anchor_enabled = True


def disable_cond_anchor(model):
    if getattr(model, "_cond_anchor_enabled", False):
        model.sample_multiview_video = type(model).sample_multiview_video.__get__(model, type(model))
        model._cond_anchor_enabled = False


# ───────────── 6-4 (c′): 뷰별 입력 앵커 — 오른쪽 뷰는 외부 파라미터로 조건 포인트맵을 참조 프레임으로 변환 후 앵커 ─────────────
# 데이터셋(spartan_video_dataset.py:1206)이 GT right를 만드는 변환과 동일: T = inv(cam_extr_ref) @ cam_extr_right, p_ref = T·p_right
def _transform_pointmap(pm_metric, T):
    """pm_metric: (3,H,W) 미터 단위 포인트맵, T: (4,4) → (3,H,W)"""
    C, H, W = pm_metric.shape
    pts = pm_metric.reshape(3, -1)                                   # (3, HW)
    hom = torch.cat([pts, torch.ones(1, H * W, device=pts.device, dtype=pts.dtype)], 0)
    out = (T.to(pts.dtype) @ hom)[:3]
    return out.reshape(3, H, W)


EXTR_IDX = (0, 0)   # (왼쪽 참조 프레임 idx, 오른쪽 조건 프레임 idx); 조건=클립 프레임0, 예측 프레임0=클립 프레임1 → 카메라가 움직이면 (1,0)


def set_extr_idx(ref_idx, cond_idx):
    global EXTR_IDX
    EXTR_IDX = (int(ref_idx), int(cond_idx))


def cond_anchor_scale_right(video_dict):
    """오른쪽 뷰: 조건 포인트맵(자기 카메라 프레임)을 참조 프레임으로 변환한 뒤 예측 프레임0 깊이와 median 스케일.
    반환 (s, cond_ref_depth). extrinsics 없으면 (None, None)"""
    ex = video_dict.get("extra", {})
    if "cam_extr" not in ex or "cam_extr_right" not in ex:
        return None, None
    pred = _unnorm(video_dict["sampled_video_right"][0, 2])
    E1s = ex["cam_extr"].reshape(-1, 4, 4); E2s = ex["cam_extr_right"].reshape(-1, 4, 4)   # 프레임별 (T+1,4,4)
    E1 = E1s[min(EXTR_IDX[0], E1s.shape[0] - 1)].to(pred.device).float()   # 참조(왼쪽) 프레임 인덱스
    E2 = E2s[min(EXTR_IDX[1], E2s.shape[0] - 1)].to(pred.device).float()   # 조건(오른쪽) 프레임 인덱스
    T = torch.linalg.inv(E1) @ E2
    cond = video_dict["cond_pointmap_right"]
    cond = cond.reshape(-1, *cond.shape[-3:])[-1]                    # (3,H,W) 정규화
    valid = _unnorm(cond[2]) > 0
    cond_ref = _transform_pointmap(_unnorm(cond), T)
    cd = cond_ref[2]
    if cd.shape != pred.shape:
        cd = torch.nn.functional.interpolate(cd[None, None], size=pred.shape, mode="nearest")[0, 0]
        valid = torch.nn.functional.interpolate(valid[None, None].float(), size=pred.shape, mode="nearest")[0, 0] > 0.5
    m = valid & (pred > 0) & (cd > 0)
    if m.sum() < 100:
        return None, cd
    return torch.median(cd[m] / pred[m]).item(), cd


def apply_cond_anchor_per_view(video_dict):
    """왼쪽: 기존 앵커 s_L / 오른쪽: 변환된 조건으로 s_R (없으면 s_L 대체). 반환 (s_L, s_R)"""
    s_l = cond_anchor_scale(video_dict, "left")
    s_r, _ = cond_anchor_scale_right(video_dict)
    if s_r is None:
        s_r = s_l
    for v, s in (("left", s_l), ("right", s_r)):
        vid = video_dict[f"sampled_video_{v}"].clone()
        vid[:, :3] = _renorm(_unnorm(vid[:, :3]) * s)
        video_dict[f"sampled_video_{v}"] = vid
    video_dict["anchor_scale"] = s_l
    video_dict["anchor_scale_right"] = s_r
    return s_l, s_r


def enable_cond_anchor(model, per_view=False):
    """model.sample_multiview_video를 감싸 출력에 앵커 보정 적용. per_view=True면 (c′) 뷰별 앵커"""
    disable_cond_anchor(model)
    orig = model.sample_multiview_video

    def wrapped(batch):
        vd = orig(batch)
        (apply_cond_anchor_per_view if per_view else apply_cond_anchor)(vd)
        return vd
    model.sample_multiview_video = wrapped
    model._cond_anchor_enabled = True


# ───────────── 6-4 (a″): robust affine 앵커 — 스케일 a + 오프셋 b (뷰별) ─────────────
# 근거(08-23 오른쪽 뷰 오차맵): 화면 전체 균일 오차 = 전역 편향. 스케일만으로는 46%만 잡힘(왼쪽 진단) → 오프셋까지.
def fit_robust_affine(pred, cd, mask):
    """cd ≈ a·pred + b. 1차 median 스케일 → 잔차 MAD 기준 이상치 제거 → 최소제곱. 비정상이면 (s, 0)로 폴백."""
    p, g = pred[mask], cd[mask]
    if p.numel() < 100:
        return 1.0, 0.0
    s = torch.median(g / p.clamp_min(1e-3)).item()
    res = g - s * p
    mad = torch.median((res - res.median()).abs()).item() + 1e-6
    keep = (res - res.median()).abs() < 2.5 * 1.4826 * mad
    if keep.sum() < 100:
        return s, 0.0
    A = torch.stack([p[keep], torch.ones_like(p[keep])], 1)
    sol = torch.linalg.lstsq(A, g[keep].unsqueeze(1)).solution.squeeze(1)
    a, b = sol[0].item(), sol[1].item()
    if not (0.5 < a < 2.0) or abs(b) > 1.0:
        return s, 0.0
    return a, b


def _affine_params_left(video_dict):
    pred = _unnorm(video_dict["sampled_video_left"][0, 2])
    cond = video_dict["cond_pointmap_left"]
    cond = cond.reshape(-1, *cond.shape[-3:])[-1]
    cd = _unnorm(cond[2])
    if cd.shape != pred.shape:
        cd = torch.nn.functional.interpolate(cd[None, None], size=pred.shape, mode="nearest")[0, 0]
    return fit_robust_affine(pred, cd, (pred > 0) & (cd > 0))


def _affine_params_right(video_dict, fallback):
    _, cd = cond_anchor_scale_right(video_dict)
    if cd is None:
        return fallback
    pred = _unnorm(video_dict["sampled_video_right"][0, 2])
    cond = video_dict["cond_pointmap_right"]; cond = cond.reshape(-1, *cond.shape[-3:])[-1]
    valid = _unnorm(cond[2]) > 0
    if valid.shape != pred.shape:
        valid = torch.nn.functional.interpolate(valid[None, None].float(), size=pred.shape, mode="nearest")[0, 0] > 0.5
    return fit_robust_affine(pred, cd, valid & (pred > 0) & (cd > 0))


def apply_cond_anchor_affine(video_dict):
    """뷰별 (a,b): xyz에 a 곱, z에 b 더함. 반환 {(view): (a,b)}"""
    aL, bL = _affine_params_left(video_dict)
    aR, bR = _affine_params_right(video_dict, (aL, bL))
    for v, (a_, b_) in (("left", (aL, bL)), ("right", (aR, bR))):
        vid = video_dict[f"sampled_video_{v}"].clone()
        xyz = _unnorm(vid[:, :3]) * a_
        xyz[:, 2] = xyz[:, 2] + b_
        vid[:, :3] = _renorm(xyz)
        video_dict[f"sampled_video_{v}"] = vid
    video_dict["anchor_scale"] = aL
    video_dict["anchor_affine"] = {"left": (aL, bL), "right": (aR, bR)}
    return video_dict["anchor_affine"]


def enable_cond_anchor(model, per_view=False, affine=False):
    """a: 왼쪽 스케일 양 뷰 / b(per_view): 뷰별 스케일 / c(affine): 뷰별 robust affine"""
    disable_cond_anchor(model)
    orig = model.sample_multiview_video
    fn = apply_cond_anchor_affine if affine else (apply_cond_anchor_per_view if per_view else apply_cond_anchor)

    def wrapped(batch):
        vd = orig(batch)
        fn(vd)
        return vd
    model.sample_multiview_video = wrapped
    model._cond_anchor_enabled = True
