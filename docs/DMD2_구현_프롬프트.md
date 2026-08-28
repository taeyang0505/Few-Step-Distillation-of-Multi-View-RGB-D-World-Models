# DMD2 적대 손실(GAN) 구현 요청

## 0. 한 줄 요약

이미 동작하는 Geo4D DMD 증류 트레이너에 **DMD2의 적대 손실(판별자)** 을 추가해 주세요.
현재 우리 구현은 DMD2의 backward simulation만 가져왔고 **GAN 항이 빠져 있습니다.** 이것이 LPIPS가
teacher 수준으로 내려가지 않는(어떤 설정에서도 +0.011이 바닥) 유력한 원인입니다.

## 1. 현재 상황 (측정된 사실)

**대상 모델**: Geo4D (arXiv:2507.01099) — 두 카메라의 현재 RGB-D 1프레임을 받아 미래 10프레임의
RGB + pointmap을 두 뷰 모두 생성하는 latent video diffusion. teacher 샘플러는 EulerEDM 25스텝 + CFG(1.0→2.5).
latent shape (2뷰 × 10프레임 = 20, 8채널, 32 × 40).

**우리가 만든 것**: 25스텝 teacher를 3스텝 student로 증류(DMD). σ 스케줄 [700, 70.5, 2.3] 고정,
x₀ 예측 후 다음 σ로 재노이징하는 샘플러, CFG 없음, bf16. 순수 추론 21.49초 → 1.64초 (13.1배).

**측정된 성능 (apple 태스크, 60샘플 = 120 뷰샘플, paired Wilcoxon)**

| 설정 | AbsRel ↓ | LPIPS ↓ | 선명도 | 다양성 |
|---|---|---|---|---|
| teacher 25스텝 | 0.0725 | **0.1227** | 0.01268 | 0.01846 |
| 학습 없이 3스텝 | 0.0697 | 0.1375 | 0.01128 | 0.00977 (−47%) |
| **DMD 3스텝 (현재)** | 0.0858 | **0.1407 (+0.0180)** | 0.01295 | 0.01931 (+4%) |
| DMD 5스텝 | 0.0773 | 0.1378 (+0.0151) | 0.01296 | 0.02027 |

**문제**: LPIPS가 teacher보다 항상 +0.011 ~ +0.018 나쁩니다. 스텝을 8까지 늘려도, 학습을 4000스텝까지
늘려도, 체크포인트를 바꿔도 이 바닥을 뚫지 못했습니다. 영역별로 나눠 봐도 전 영역에서 균일하게 나쁩니다
(전체 +15%, 배경 +15%, 움직임 +15%, 그리퍼 +18%) — 특정 영역 문제가 아니라 전역 텍스처 차이입니다.

**이미 실패한 시도 (다시 하지 말 것)**: 뷰 간 깊이비 손실 3회, 자기 앵커 손실 2회 — 전부 **기하 손실**이었고
전부 실패했습니다. **지각 품질을 겨냥한 손실은 한 번도 시도하지 않았습니다.**

## 2. 목표

DMD2 논문(arXiv:2405.14867)의 적대 손실을 추가해 **LPIPS를 teacher 수준(0.1227)에 가깝게** 낮춥니다.
성공 기준: LPIPS 격차가 +0.018 → **+0.010 이하**, 동시에 AbsRel·다양성·선명도가 현재보다 나빠지지 않을 것.

## 3. 참고 자료

- **공식 코드**: https://github.com/tianweiy/DMD2 (NeurIPS 2024 Oral). 핵심 파일 `main/train_sd.py`,
  `main/sd_guidance.py`. 관련 플래그: `cls_on_clean_image`, `gen_cls_loss`, `gen_cls_loss_weight`,
  `guidance_cls_loss_weight`, `dfake_gen_update_ratio`.
- DMD2의 판별자는 **별도 네트워크가 아니라 fake-score(guidance) 네트워크에 붙은 분류 헤드**입니다.
  `cls_on_clean_image=True`면 노이즈 입력이 아니라 예측된 clean image(x₀)에서 판별합니다.

## 4. 수정할 파일과 통합 지점

**파일**: `~/4dgen/notebooks/geo4d_dmd_train.py` (485줄). 서버 `inha5090`, conda 환경 `video_policy`.

기존 구조 (그대로 유지할 것):

| 이름 | 역할 |
|---|---|
| `gen` | student UNet (학습 대상) |
| `fake` | teacher 사본, fake score 네트워크 (학습 대상) — **여기에 판별자 헤드를 붙일 것** |
| `teacher` | 동결된 teacher (real score, CFG 가이더 포함) |
| `D(unet, x, sigma_val, c, ami)` | x_σ → x₀ 예측 (cond-only) |
| `D_teacher_cfg(x, sigma_val, c, uc, ami)` | teacher + CFG 가이더 |
| `run_generator(c, ami, k, grad)` | backward simulation: k스텝까지 no_grad, k번째 x₀만 grad |
| `dmd_loss(x0, c, uc, ami)` | 현재의 분포 매칭 손실 |
| `critic_loss(c, ami)` | fake score 학습 손실 (EDM 가중) |
| `opt_g`, `opt_c` | AdamW8bit, lr 2e-6 / 4e-7, betas (0.0, 0.999) |
| 학습 루프 | `args.gen_every`(기본 5)마다 generator 1회, 나머지는 critic |

**"진짜" 샘플을 어디서 가져오나**: `--pairs_dir`에 teacher가 25스텝으로 만든 (noise, 최종 latent) 쌍이
저장돼 있습니다(`pair_files`, 파일명에 idx 포함). **판별자의 real = 이 teacher 최종 latent**,
**fake = student가 만든 x₀** 로 두는 것이 자연스럽습니다. 파일 구조는 코드에서 확인하세요.

## 5. 구현 요구사항

1. **판별자 헤드**: `fake` UNet의 중간 특징(bottleneck 권장)에 forward hook을 걸어 특징을 받고,
   작은 conv + pooling + linear로 스칼라 로짓을 냅니다. **파라미터를 작게 유지하세요** (아래 메모리 제약).
2. **두 개의 손실**:
   - `guidance_cls_loss`: 판별자 학습. real(teacher latent) vs fake(student x₀). non-saturating 또는 hinge.
   - `gen_cls_loss`: generator가 판별자를 속이는 항. `dmd_loss`에 더해집니다.
3. **가중치 균형 — 가장 중요**: 고정 가중치를 쓰지 마세요. 우리는 이전에 고정 λ=1로 보조 손실을 넣었다가
   그 그래디언트가 DMD의 **500배**가 되어 DMD 신호를 지운 적이 있습니다. 기존 코드에 있는 방식
   (`λ = β · |g_DMD| / |g_aux|`, x₀ 그래디언트 크기 기준 상대 강도)을 그대로 따르세요.
   β를 `--gan_weight` 인자로 노출하고 기본값 0(꺼짐)으로 두세요.
4. **플래그**: `--gan_weight`(기본 0), `--gan_start_step`(기본 0, 워밍업 후 켜기용),
   `--cls_on_clean_image`(기본 True). 0이면 기존 동작과 **비트 단위로 동일**해야 합니다.
5. **로깅**: 기존 `acc` 딕셔너리에 판별자 손실·generator 적대 손실·실제 적용된 λ를 추가하고
   `--log_every`마다 출력하세요. 기존 DIAG(std비) 출력은 그대로 유지하세요.
6. **체크포인트**: `save()`가 판별자 헤드 가중치도 저장하고 `--resume`이 복원하도록 하세요.
   단 **추론 시에는 판별자가 필요 없으므로** `dmd_gen.pt`의 `student` 키는 지금 형식 그대로 두세요
   (평가 스크립트들이 이 키를 읽습니다).

## 6. 제약

- **GPU 메모리 32GB, 현재 피크 24.4GB.** 여유가 7.6GB뿐입니다. UNet 3개(gen/fake/teacher)가 bf16으로
  9.2GB를 쓰고 AdamW8bit를 씁니다. 판별자는 작게, 필요하면 gradient checkpointing을 쓰세요.
- **공유 GPU입니다.** OOM으로 죽으면 기존 `--resume` 경로로 복구돼야 합니다.
- 학습 2000스텝이 약 1시간입니다. 판별자 추가로 20% 이상 느려지면 알려주세요.
- Python 3.10, torch 2.11+cu128, **xformers 사용 금지**(SDPA 패치가 적용돼 있음).

## 7. 검증 방법

1. `--gan_weight 0`으로 200스텝 돌려 기존과 손실 곡선이 같은지 확인 (회귀 없음)
2. `--gan_weight 0.1`로 2000스텝 학습 (약 1시간)
3. 평가:
   ```
   python notebooks/bench_precise_6a.py --student_ckpt <새_ckpt> \
     --configs T25 S3b S5b --n_batches 20 --n_div 3 --fast --tag _gan
   ```
   **판정**: LPIPS 격차가 +0.018보다 작아졌는가. AbsRel·다양성·선명도가 나빠지지 않았는가.
4. std비 진단(DIAG)이 1.0 근처를 유지하는지 확인 — 1.1을 넘으면 과선명 구간입니다.

## 8. 주의할 함정 (우리가 겪은 것)

- 보조 손실이 DMD를 압도하는 문제 → 반드시 그래디언트 크기 기준 균형
- 학습이 길수록 **과선명**이 생겨 선명도가 GT를 넘고 깊이가 나빠집니다 (다른 태스크에서 GT의 113%까지 감).
  적대 손실이 이 경향을 가속할 수 있으니 200스텝마다 저장하고 여러 체크포인트를 평가하세요.
- `optimizer` 모멘트가 재시작되면 실질 학습량이 줄어 결과가 달라집니다 (구간 재개와 연속 학습이 다름).
