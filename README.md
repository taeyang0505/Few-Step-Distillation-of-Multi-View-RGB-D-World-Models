# Few-Step Distillation of Multi-View RGB-D World Models

> Geo4D(Geometry-aware 4D Video Generation, ICLR 2026)의 추론 병목(10프레임×2뷰에 ~22–30초)을 **분포 매칭 증류(DMD) + 입력 앵커 보정 + bf16**으로 **1.64초(13.3배)** 까지 줄이면서, 선명도·시드 다양성·왼쪽 뷰 기하를 teacher 수준으로 유지하는지 검증한 연구 기록. 오른쪽 뷰 깊이(AbsRel +0.022)와 LPIPS(+0.018)의 작지만 유의한 손실은 그대로 보고한다.

- 기간: 2026-08-20 ~ 2026-08-23 (진행 중) · 장비: RTX 5090 32GB 1장 (공유 서버)
- 상세 연구 로그(Notion 내보내기): [`docs/research_log.md`](docs/research_log.md)
- 모든 수치는 자체 실측. 논문 인용은 arXiv 번호·섹션을 함께 표기.

---

## 1. 한눈에 보기

| 축 | 지표 (20샘플 = 뷰 40, 고정 시드) | Geo4D teacher (25스텝) | 우리 (3스텝 + 앵커 + bf16) | 판정 |
|---|---|---|---|---|
| 속도 | 순수 추론, 10프레임×2뷰 RGB-D | 21.80초 | **1.64초 (13.3배)** / +torch.compile 1.18초† | ✅ 준정적 조작 목표 2초 달성 |
| 품질 | 선명도 (Laplacian var) / 시드 다양성 | 0.0134 / 0.0227 | **0.0136 / 0.0224** | ✅ 동급 (p=0.95 / 0.84) |
| 기하 | 왼쪽 뷰 AbsRel / 뷰 간 정합 CV-Chamfer | 0.067 / 0.169 | **0.076 / 0.137** | ✅ +0.008 / CV 개선 (p=0.006) |
| 기하 | 오른쪽 뷰 AbsRel | 0.066 | 0.088 | 🔶 +0.022 (p<0.001) 유의 손실 |
| 품질 | PSNR / LPIPS | 20.62 / 0.118 | 20.43 / 0.136 | 🔶 −0.19 (p=0.026) / +0.018 (p<0.001) |
| 1스텝 | 위 전부 | — | 1.65초, AbsRel 0.116, LPIPS 0.177, 다양성 0.0117 | ❌ 다양성 절반 — 품질 미달 |

† compile 설정(1.18초)의 품질은 5샘플에서만 확인(AbsRel ±0.005). bf16 설정(1.64초)은 20샘플에서 fp32 대비 무손실 확인.

**정직한 해석**: 학습 없이 스텝만 줄인 teacher(3스텝 재노이징)는 깊이 정확도(AbsRel 0.066)는 유지하지만 **시드 다양성이 −47%** 로 붕괴하고 선명도도 떨어진다(조건부 평균 수렴). DMD student는 다양성·선명도를 teacher 수준으로 복원하는 대신 AbsRel +0.015·LPIPS +0.018을 치른다 — **트레이드오프이지 무료 점심이 아니다.** 월드모델 용도(여러 미래 샘플링)에서 다양성이 필수라는 것이 우리 주장의 근거가 되어야 한다.

---

## 2. 메인 표

데이터 시드 1234, 배치 20(=뷰 40), 오른쪽 GT의 가짜 유효 픽셀(§5 발견 10) 제외, 다양성은 3배치×시드 4. 원본: [`results/quantitative/precise_6a_main20.txt`](results/quantitative/precise_6a_main20.txt), [`precise_6a_main20_fast.txt`](results/quantitative/precise_6a_main20_fast.txt)

| 방법 | 학습 | 스텝 | 순수 추론 | PSNR↑ | AbsRel↓ (왼/오) | LPIPS↓ | 선명도 | 시드 다양성 | CV↓ |
|---|---|---|---|---|---|---|---|---|---|
| Geo4D teacher (Euler) | — | 25 | 21.80초 | 20.62 | 0.066 (.067/.066) | 0.118 | 0.0134 | 0.0227 | 0.169 |
| teacher, 스텝만 축소 (Euler) | 없음 | 4 | 4.84초 | 21.22 | 0.064 | 0.132 | 0.0121 | 0.0131 (−42%) | 0.165 |
| teacher, 스텝만 축소 (재노이징) | 없음 | 3 | 2.81초 | 21.30 | 0.066 | 0.132 | 0.0119 | 0.0120 (−47%) | 0.167 |
| ODE 회귀 초기화 (v2) | 회귀 | 4 | 7.9초* | 14.42 | 0.547 | — | 안개 | 시드 동일 | 0.137 |
| **DMD student + 뷰별 앵커 (bf16)** | DMD | 3 | **1.64초** | 20.43 | 0.082 (.076/.088) | 0.136 | **0.0136** | **0.0224** | 0.137 |
| DMD student + 뷰별 앵커 | DMD | 1 | 1.65초* | 20.56 | 0.116 | 0.177 | 0.0107 | 0.0117 | 0.165 |

\* 평가 코드 포함 시간(v2) / fp32 시간(1스텝). Paired Wilcoxon(n=40) vs teacher — 우리(3스텝): PSNR −0.19 (p=0.026), AbsRel +0.015 (p<0.001; 왼 +0.008, 오 +0.022), LPIPS +0.018 (p<0.001), 선명도 +0.0001 (p=0.95), 다양성 −0.0004 (p=0.84), CV −0.034 (p=0.006).

### 속도 분해 (순수 추론, 평가 코드 제거)

원본: [`speed_variants.txt`](results/quantitative/speed_variants.txt), [`logs/speed_variants.log`](results/quantitative/logs/speed_variants.log)

| 경로 | UNet | conditioner | VAE 디코딩 | 총 |
|---|---|---|---|---|
| teacher Euler 25스텝 (CFG, 배치 40) | 20.13 (25회) | 0.73 | 0.89 | 21.75초 |
| teacher Euler 4스텝 | 3.22 | 0.73 | 0.89 | 4.84초 |
| student 재노이징 3스텝, fp32 | 1.19 (3회×0.40) | 0.73 | 0.88 | 2.81초 |
| + uc 생략 (student는 CFG 미사용) | 1.18 | 0.30 | 0.90 | 2.38초 |
| + VAE 디코더 bf16 | 1.18 | 0.30 | 0.68 | 2.16초 |
| + UNet·conditioner bf16 | 0.66 | 0.30 | 0.68 | **1.64초** |
| + torch.compile(디코더) | 0.66 | 0.30 | 0.33 | 1.29초 |
| + torch.compile(UNet) | 0.55 | 0.30 | 0.33 | **1.18초** |

교훈: 지금까지 보고되던 "25.4초 / 5.8초"는 평가 스캐폴딩(loss용 UNet 1회·GT 재구성 디코딩·GT 인코딩·conditioner 중복) 2.9초를 포함한 값이었다(발견 11). 추론 UNet이 fp32로 돌고 있었던 것이 가장 큰 낭비.

---

## 3. 정성 결과

`results/qualitative/` — 행 순서는 파일명 라벨 참조 (GT / teacher 25스텝 / student 3스텝 / student 1스텝 등).

| 무엇 | 파일 | 보는 포인트 |
|---|---|---|
| 학습 없는 스텝 축소 (Step 1) | [`qualitative/rgb_left.png`](results/qualitative/qualitative/rgb_left.png), [`diversity_left.png`](results/qualitative/qualitative/diversity_left.png), [`compare_left.gif`](results/qualitative/qualitative/compare_left.gif) | 1스텝에서 움직이는 팔만 반투명 얼룩, 시드 4개 동일 |
| ODE 회귀 초기화 (v1) | [`student_qual/rgb_left.png`](results/qualitative/student_qual/rgb_left.png) | 전역 안개 (평균 붕괴) |
| DMD student (step1600) | [`dmd6a_qual/rgb_left.png`](results/qualitative/dmd6a_qual/rgb_left.png), [`depth_left.png`](results/qualitative/dmd6a_qual/depth_left.png), [`diversity_student1.png`](results/qualitative/dmd6a_qual/diversity_student1.png) | 3스텝은 선명, 1스텝은 팔이 뿌옇고 시드 유사 |
| DMD student 오른쪽 뷰 + 오차맵 | [`dmd6a_qual_lr/rgb_right.png`](results/qualitative/dmd6a_qual_lr/rgb_right.png), [`depth_right.png`](results/qualitative/dmd6a_qual_lr/depth_right.png), [`err_right.png`](results/qualitative/dmd6a_qual_lr/err_right.png) | 오차가 화면 전체에 균일 = 전역 스케일 편향 (앵커로 해결) |
| 학습 없는 teacher 3스텝 (T3r) | [`t3r_qual/rgb_left.png`](results/qualitative/t3r_qual/rgb_left.png), [`diversity_T3r1.png`](results/qualitative/t3r_qual/diversity_T3r1.png) | 팔 번짐, 시드 4개 동일 — 다양성 붕괴의 육안 확인 |
| Self-Forcing / Causal-Forcing 재현 비디오 | [`videos/`](results/qualitative/videos) | 4스텝 증류 모델의 생성물 (Step 3·4 재현) |

![DMD student 3스텝 (왼쪽 뷰): GT / teacher25 / student3 / student1](results/qualitative/dmd6a_qual/rgb_left.png)

![오른쪽 뷰 상대 깊이 오차맵: teacher25 / student3 / student1 (밝을수록 오차 큼)](results/qualitative/dmd6a_qual_lr/err_right.png)

---

## 4. 무엇을 했나 — 단계별 기록

| 단계 | 내용 | 결과 (원본 파일) |
|---|---|---|
| 0 | Geo4D 병목 실측·원인 분해 | 10프레임×2뷰 25.4초(평가 포함)/21.8초(순수), UNet 25회가 93% — [`bench_profile.py`](code/geo4d/bench_profile.py) |
| 1 | 학습 없는 스텝 축소 스윕 + 블러 가설 3중 검증 + 정성 | 8스텝까지 무손실, 1–2스텝 붕괴. **해리**: PSNR↑인데 LPIPS·다양성·기하↓ — [`full_sweep_results.txt`](results/quantitative/full_sweep_results.txt), [`blur_test_results.txt`](results/quantitative/blur_test_results.txt) |
| 2 | 평가 파이프라인 고정 | `bench_*.py` 8종, 고정 데이터 시드 |
| 3 | Self-Forcing 추론 재현 (5090) | 8.2 FPS, 증류 모델은 평균 붕괴 없음 — [`code/self_forcing/inference_timed.py`](code/self_forcing/inference_timed.py) |
| 4 | Causal-Forcing 3단계 체크포인트 해부 | 속도=구조(0.45초/청크), 품질=학습 — [`videos/stage*`](results/qualitative/videos) |
| 5 | DMD 학습 스모크 (Wan 1.3B, 32GB 레시피) | bf16 3벌 + AdamW8bit + T5 GPU 셔틀 + 메모리 프로브 — [`code/self_forcing/self_forcing_dmd_5090.yaml`](code/self_forcing/self_forcing_dmd_5090.yaml), [`geo4d_patches/self_forcing.patch`](code/geo4d_patches/self_forcing.patch) |
| 6-1 | ODE 쌍 생성 (teacher 25스텝 궤적, 284쌍) | [`geo4d_ode_gen_v2.py`](code/geo4d/geo4d_ode_gen_v2.py), [`logs/ode_gen_1000.log`](results/quantitative/logs/ode_gen_1000.log) |
| 6-2 | ODE 회귀 초기화 v1·v2 | 둘 다 안개 — MSE 회귀의 평균 붕괴 확정 → DMD 필연 — [`geo4d_ode_init_train_v2.py`](code/geo4d/geo4d_ode_init_train_v2.py), [`geo4d_sigma_diag.py`](code/geo4d/geo4d_sigma_diag.py) |
| 6-3 | Geo4D 바닐라 DMD (2000스텝, 1시간, 피크 24.4GB) | 선명도·다양성 복원, 최적 ckpt step1600 — [`geo4d_dmd_train.py`](code/geo4d/geo4d_dmd_train.py), [`logs/dmd_6a.log`](results/quantitative/logs/dmd_6a.log) |
| 6-4(a) | 입력 앵커 보정 (조건 프레임 + 카메라 외부 파라미터, 학습 없음) | student AbsRel 0.175→0.086; teacher도 0.072→0.060 — [`geo4d_fewstep.py`](code/geo4d/geo4d_fewstep.py), [`eval_6x_maskfix.txt`](results/quantitative/eval_6x_maskfix.txt) |
| 6-4(c) | GT 감독 뷰 간 깊이 비 loss (β 1/3/10) | 3회 모두 DMD와 충돌해 PSNR −3.5dB 붕괴 — [`logs/train_6b*.log`](results/quantitative/logs) |
| 7 | 고정비 공격: uc 생략 → bf16 → compile | 2.81 → 1.64 → 1.18초, bf16 무손실(20샘플) — [`bench_speed_variants.py`](code/geo4d/bench_speed_variants.py) |
| 8 | 기능 평가 (정책 루프) | **미수행** |

### Geo4D 논문의 한계(Sec. 5) 대비

| Geo4D 한계 | 상태 |
|---|---|
| ① 추론 속도 (~30초/10프레임, closed-loop planning 어려움) | 🔶 속도는 13.3배 단축(1.64초). 단 "closed-loop에서 동작한다"는 정책 루프 평가(Step 8) 전까지 주장 불가 |
| ② 멀티뷰 RGB-D 데이터·캘리브레이션 요구 | ❌ 범위 밖 |
| ③ 실세계 깊이 취득 품질 | ❌ 범위 밖 |

---

## 5. 발견 (전부 자체 실측, † = 철회)

1. **해리(dissociation)** — 스텝을 줄이면 PSNR은 "개선"(+4.8%)되는데 LPIPS +21%·CV-Chamfer +10.5% 악화. 픽셀 지표 단독 평가의 오판 위험.
2. **지표 붕괴 순서** — 시드 다양성(−77%) → LPIPS → 기하 → PSNR(끝까지 오판). 지표 민감도 위계.
3. **CV-Chamfer 함정** — 두 뷰가 닮은 오답을 내면 정합 지표가 좋아짐(v1: CV −31%인데 AbsRel 3배 악화). 절대 거리라 스케일에도 비례. 기하 평가는 {정합 + 뷰별 정확도 + 지각} 세트 필수.
4. **MSE 회귀의 평균 붕괴** — 의사 궤적(v1)·진짜 궤적(v2) 모두 σ700 loss ≈0.40 정체 + 전역 안개. 궤적 품질이 아니라 목적함수 본질 → 분포 매칭 필요.
5. **DMD가 평균 붕괴를 꺾는다** — 400회 업데이트로 선명도·다양성 teacher 수준. 학습 중 DIAG(std(x0)/std(z_teacher))가 최적 ckpt 예측(≈1.0 최고, 1.1 과상승 악화).
6. † "teacher 오른쪽 뷰 붕괴" — 철회 (발견 10의 평가 오염).
7. **GT 감독 기하 loss는 DMD와 충돌** — DMD는 teacher 분포를 통째로 증류하므로 그에 모순되는 감독은 품질을 희생시킴 (3회 재현; 단 표적이 오염된 GT였으므로 깨끗한 GT로도 충돌하는지는 미확인).
8. **Geo4D의 두 뷰는 가중치 100% 공유** — `wrappers.py`의 뷰별 디코더 deepcopy가 얕은 복사의 `_modules` 공유 때문에 무효 (ckpt `output_blocks` 795키 전부 동일). 뷰2를 구분하는 것은 `spatial_context`와 조건 latent뿐.
9. **학습 없는 few-step teacher는 샘플러와 무관하게 붕괴** — 재노이징 3스텝 다양성 −47%·선명도 −12% (1스텝 −78%). 논문 표의 필수 대조군.
10. **오른쪽 뷰 GT의 53%가 가짜 유효 픽셀** — 데이터셋이 무효(xyz=0)까지 참조 프레임으로 변환해 xyz=이동벡터로 채움. 제외 전 teacher 오른쪽 AbsRel 0.418, 제외 후 0.064. 오른쪽 뷰 관련 이전 결론(구조 오류·뷰 간 불일치) 전부 철회. [`check_cond_transform.py`](code/geo4d/check_cond_transform.py)
11. **시간 측정에 평가 스캐폴딩 2.9초 포함** — 순수 추론은 teacher 21.75초, student 3스텝 2.81초(→1.64초). [`bench_profile_student.py`](code/geo4d/bench_profile_student.py)
12. **입력 앵커는 teacher도 개선** — robust affine 앵커로 teacher AbsRel 0.072→0.060 (10/10 샘플). 학습 없는 기하 개선.

---

## 6. 레포 구조

```
code/
  geo4d/                 Geo4D(~/4dgen) 위에서 돌리는 스크립트 (notebooks/ 에 두고 실행)
    geo4d_fewstep.py       RenoiseSampler(재노이징 few-step 샘플러) + 입력 앵커 보정(a/b/c)
    geo4d_dmd_train.py     DMD 트레이너 (--cv_weight 0 = 바닐라, >0 = 뷰 간 깊이 비 loss 실험)
    geo4d_ode_gen*.py      ODE 쌍 생성 (v2 = teacher 궤적 중간 상태 캡처)
    geo4d_ode_init_train*.py  ODE 회귀 초기화 (실패 기록용)
    bench_*.py             평가·진단 스크립트 (아래 표)
    launchers/             서버에서 쓴 nohup 런처 (인자 조합 기록)
  geo4d_patches/         sm_120(Blackwell)용 패치: xformers→SDPA, inference.yaml 경로, Self-Forcing 32GB 레시피
  self_forcing/          self_forcing_dmd_5090.yaml (32GB 레시피), inference_timed.py
  setup/                 conda 환경·체크포인트 다운로드 스크립트
results/
  quantitative/          모든 평가 결과 txt + 샘플별 raw json, logs/ 에 학습·평가 로그
  qualitative/           정성 그리드 png/gif, videos/ 에 Self-/Causal-Forcing 재현 mp4
docs/research_log.md     Notion 연구 로그 전체 내보내기 (Step 0~7 상세, 발견, 철회, 다음 액션)
```

| 스크립트 | 용도 |
|---|---|
| `bench_eval_6x.py` | 표준 평가: 고정 시드, teacher/student 설정(`T25 T4 S3 S3b S3c ...`) paired Wilcoxon, `--fix_mask`(발견 10), 앵커 접미사 a/b/c |
| `bench_precise_6a.py` | LPIPS·선명도·시드 다양성 포함 정밀 평가 (`--fast` bf16, `--compile`) — 메인 표 |
| `bench_speed_variants.py`, `bench_profile_student.py` | 순수 추론 시간 분해·가속 변형 |
| `bench_scale_diag.py`, `bench_cond_calib.py`, `bench_frame_diag.py` | 깊이 편향 진단 (오라클 affine, 조건 앵커, 프레임별) |
| `bench_student_qual.py`, `bench_qualitative.py` | 정성 그리드 (RGB/깊이/오차맵/시드 다양성) |
| `check_extr.py`, `check_cond_transform.py` | 데이터셋 외부 파라미터·좌표계·가짜 픽셀 검사 |

---

## 7. 재현

환경 (RTX 5090, sm_120): torch 2.11 + cu128, **xformers 제거 + SDPA 패치**(`code/geo4d_patches/geo4d_4dgen.patch`), flash-attn 설치 불가. Geo4D 코드(`~/4dgen`)·체크포인트(apple 태스크)·데이터는 원 저자 배포본을 따른다 ([`code/setup/`](code/setup) 참고). 스크립트는 `~/4dgen/notebooks/`에 두고 `~/4dgen`에서 실행한다. 경로(`/home/sun4208/...`)는 환경에 맞게 수정.

```bash
# 1) teacher 궤적 ODE 쌍 (데이터셋 142샘플 × 시드 2 = 284쌍, ~2h)
python notebooks/geo4d_ode_gen_v2.py --max_pairs 1000
# 2) ODE 회귀 초기화 (v2, 1200스텝 ~1.5h) — DMD의 초기 가중치
python notebooks/geo4d_ode_init_train_v2.py
# 3) 바닐라 DMD (2000스텝 ~1h, 피크 24.4GB). 최적 ckpt = step1600 (DIAG std비 ≈1.0)
python notebooks/geo4d_dmd_train.py --max_steps 2000 --out_dir ~/Geo4D/dmd_6a
# 4) 메인 표 (20샘플, bf16, ~20분)
python notebooks/bench_precise_6a.py --configs T25 T4 T3r S3b S1b --n_batches 20 --fast --tag _main20
# 5) 순수 추론 시간 분해 / 가속 변형 (~5분)
python notebooks/bench_speed_variants.py --n 5
```

---

## 8. 한계와 다음 단계

- 단일 태스크(apple)·고정 시드 20샘플·단일 GPU. 다른 에피소드/태스크 일반화 미검증.
- 오른쪽 뷰 AbsRel +0.022, LPIPS +0.018의 유의 손실. 1스텝은 품질 미달.
- 휴머노이드 목표(0.3–0.5초)는 고정비(conditioner 0.30 + 디코딩 0.33초)로 미달.
- Step 8(정책 루프에서의 기능 평가) 미수행 — Geo4D 한계 ①을 "해결"했다고 말하려면 필요.
- 방법론 novelty는 약함(DMD·재노이징·bf16은 기존 기법). 기여는 (i) few-step이 멀티뷰 기하를 보존하는지의 첫 검증, (ii) 평가 발견(해리·붕괴 순서·CV 함정·GT 아티팩트), (iii) 입력 앵커 보정, (iv) 부정적 결과(회귀 초기화·GT 감독 loss). 다음: teacher와 충돌하지 않는 기하 보존 학습(자기 앵커 loss 등) 1개 성공이 방법 논문으로 가는 조건.

## 참고 문헌

Geo4D arXiv:2507.01099 · DMD arXiv:2311.18828 · DMD2 arXiv:2405.14867 · CausVid arXiv:2412.07772 · Self-Forcing arXiv:2506.08009 · Causal Forcing arXiv:2602.02214 · perception-distortion tradeoff arXiv:1711.06077 · LPIPS arXiv:1801.03924

코드 패치는 각 원 저장소(Geo4D/4dgen, Self-Forcing, Causal-Forcing)의 라이선스를 따릅니다.
