# 연구 로그 (Notion 페이지 내보내기, 2026-08-23)

> 원본: Notion 「🎯 연구 계획: Geometry-Preserving Fast Inference」. 표는 HTML 그대로이며 GitHub에서 렌더링됩니다. 첨부 이미지는 `results/qualitative/`에 있습니다.


비디오 생성 월드모델의 추론 병목을 해소하되, **기하 일관성과 비디오 품질을 검증·보존하는 것을 차별점**으로 하는 연구 계획. 모든 수치·인용은 arXiv 원문/README에서 직접 확인 (2026-08-21 기준). 자체 실측은 inha5090 서버(RTX 5090)에서 수행.
> 	**한눈에 보기 (2026-08-23)** — DMD 3스텝 + 입력 앵커 + bf16 → Geo4D 순수 추론 **21.8초 → 1.64초 (13.3배)**. **선명도·시드 다양성·왼쪽 깊이는 teacher 동급**, **오른쪽 깊이 AbsRel +0.022·LPIPS +0.018은 작지만 유의한 손실**, **1스텝(1.65초)은 다양성 절반으로 미달**. 학습 없이 스텝만 줄인 teacher는 다양성 −47%로 붕괴 — DMD의 기여는 여기.
	색 범례: **초록 = 동급/달성/확정 발견** · **노랑 = 유의 손실/주의** · **빨강 = 미달/철회** · † = 철회된 결론
	바로가기: 섹션 5 → 현황판 / 발견 1–12 / 목표 대비 표 / 논문에 쓸 것 / 다음 액션. 상세 로그(Step 0–8)는 재현·디버깅용.


# 1. Purpose와 목표
**메인 purpose**: 4D 비디오 월드모델(Geo4D류)의 추론 시간을 도메인 요구 수준 이내로 단축.
**부가 purpose (novelty의 핵심)**: 단축된 추론 시간에서도 **기하 일관성(cross-view 포인트맵 정합)과 비디오 품질이 유지되는가**를 검증하고 보존하는 방법 제시.
## 도메인별 목표 추론 시간
<table header-row="true">
<tr>
<td>도메인</td>
<td>목표</td>
<td>근거</td>
<td>판정</td>
</tr>
<tr>
<td>준정적 조작 (집기·놓기·수납)</td>
<td>**≤ 2초**</td>
<td>재계획 주기 — 정지 물체는 open-loop 가능, 교란 시 1–2초 내 재계획이면 실용적</td>
<td>증류 1개 축으로 도달 가능 (Self-Forcing 0.69초 선례)</td>
</tr>
<tr>
<td>휴머노이드 움직임 (전신 보행·조작)</td>
<td>**≤ 0.3–0.5초**</td>
<td>보행 주기(0.5–1초)의 절반 내 다음 스텝 계획 필요. 하위 관절 제어는 별도 제어기 담당</td>
<td>증류 + 캐싱/양자화 결합 필요. frame-wise 0.45초가 경계선</td>
</tr>
<tr>
<td>동적 조작 (받기·따르기·미끄러짐 대응)</td>
<td>**≤ 0.1초**</td>
<td>접촉 역학이 수십 ms 단위로 변함</td>
<td>스트레치 골 — 단일 GPU 최고 기록의 5배 격차, 미답 영역</td>
</tr>
</table>
## 현재 위치 (자체 실측, RTX 5090 1장)
*[이미지:  — Notion 첨부, 레포의 results/qualitative 참조]*
*[이미지:  — Notion 첨부, 레포의 results/qualitative 참조]*
- Geo4D 공개 체크포인트(apple 태스크): 10프레임×2뷰 생성 **25.4초**(평가 코드 포함) / **순수 추론 21.75초** (논문 "4090에서 ~30초" 재현), 피크 VRAM 13.2GB
- **현재 도달점 (08-23)**: DMD student 3스텝 + 입력 앵커 = 순수 추론 **2.81초 (7.7배)**, 선명도·다양성·왼쪽 기하·PSNR teacher 동급, 오른쪽 AbsRel +0.022·LPIPS +0.020 유의 손실 (→ 섹션 5 목표 대비 표)
- 원인 분해 (컴포넌트별 프로파일링): **UNet denoising 21.14초 = 83.8%** (25스텝 + loss 1회 = 26호출, 호출당 813ms) / conditioner 1.81초(7.2%) / VAE 디코딩 1.75초(7.0%) / VAE 인코딩 0.48초(1.9%)
- 4스텝 증류 시 예상: **25.2 → 7.3초** (3.5배). 이후 conditioner+VAE 고정비 4.1초가 새 병목 → 준정적 목표(2초)까지 가려면 고정비까지 공격 필요
# 2. 4계열 가속 연구의 달성 기록 (출처 확인)
## ① 증류 (distillation) 계열 — 가장 강력한 지렛대
<table header-row="true">
<tr>
<td>논문</td>
<td>달성</td>
<td>조건</td>
<td>출처</td>
</tr>
<tr>
<td>**Self-Forcing**</td>
<td>첫 프레임 **0.45초**(frame-wise)/0.69초(chunk), **17 FPS**. 4090에서 ~10 FPS</td>
<td>H100 1장, 832×480, Wan2.1-1.3B, 50→4스텝</td>
<td>[arXiv:2506.08009](https://arxiv.org/abs/2506.08009) Table 1</td>
</tr>
<tr>
<td>Causal Forcing</td>
<td>0.69초/17 FPS (같은 속도, 품질 개선이 기여) + 1/2-step 변형 공개</td>
<td>동일</td>
<td>[arXiv:2602.02214](https://arxiv.org/abs/2602.02214)</td>
</tr>
<tr>
<td>CausVid</td>
<td>첫 프레임 1.3초, 9.4 FPS — teacher(219.2초) 대비 160배</td>
<td>H100 1장, 640×352</td>
<td>[arXiv:2412.07772](https://arxiv.org/abs/2412.07772)</td>
</tr>
<tr>
<td>Seaweed-APT</td>
<td>**1-step**. 720p 2초 분량 latent를 H100 1장에서 2초(전체 파이프라인 6.03초), 8×H100이면 24fps 실시간</td>
<td>8B DiT</td>
<td>[arXiv:2501.08316](https://arxiv.org/abs/2501.08316) Table 8</td>
</tr>
<tr>
<td>AccVideo</td>
<td>오프라인 배치 7.7–9.6배 (HunyuanVideo 3234→380초 등)</td>
<td>A100</td>
<td>[arXiv:2503.19462](https://arxiv.org/abs/2503.19462)</td>
</tr>
</table>
## ② AR/스트리밍 계열 — 첫 프레임 지연 구조를 바꿈
<table header-row="true">
<tr>
<td>논문</td>
<td>달성</td>
<td>조건</td>
<td>출처</td>
</tr>
<tr>
<td>MAGI-1</td>
<td>24 FPS 실시간 (청크당 ≤1초)</td>
<td>480p, 16스텝, **H100×24장**, 24B</td>
<td>[arXiv:2505.13211](https://arxiv.org/abs/2505.13211) §4.2.1</td>
</tr>
<tr>
<td>StreamDiT</td>
<td>16 FPS 스트리밍 (스텝당 482ms/8프레임)</td>
<td>H100 1장, 512p, 4B, 8스텝 증류</td>
<td>[arXiv:2507.03745](https://arxiv.org/abs/2507.03745) §5.4</td>
</tr>
<tr>
<td>Genie 3</td>
<td>720p 24 FPS 인터랙티브</td>
<td>**하드웨어 비공개** (블로그 주장)</td>
<td>DeepMind 블로그</td>
</tr>
<tr>
<td>NOVA / Pyramid Flow / RIVER</td>
<td>2.75 / ~2.1 / ~1.6 FPS — 효율화이지 실시간 아님</td>
<td>A100·3090급</td>
<td>arXiv:2412.14169 / 2410.05954 / 2211.14575</td>
</tr>
<tr>
<td>Diffusion Forcing</td>
<td>속도 수치 미보고 (기여는 무한 rollout 안정성)</td>
<td>—</td>
<td>[arXiv:2407.01392](https://arxiv.org/abs/2407.01392)</td>
</tr>
</table>
**핵심 확인: 단일 GPU에서 24 FPS를 넘긴 공개·검증 사례는 아직 없음** (최고 16–17 FPS).
## ③ 학습 불필요(training-free) 가속 — 곱셈 항, 상한 뚜렷
<table header-row="true">
<tr>
<td>기법</td>
<td>배수</td>
<td>비고</td>
<td>출처</td>
</tr>
<tr>
<td>TeaCache / MagCache / FasterCache</td>
<td>~2× (안전권; TeaCache-fast 6.83×는 손실 큼)</td>
<td>캐싱끼리 상호 배타(같은 슬롯). **증류 후 저스텝에서는 이득 급감** — 증류와 완전히 곱해지지 않음</td>
<td>arXiv:2411.19108 / 2506.09045 / 2410.19355</td>
</tr>
<tr>
<td>SVDQuant (4-bit)</td>
<td>커널 ~3×</td>
<td>비디오 모델 공개본 미확인 (이미지 실증)</td>
<td>[arXiv:2411.05007](https://arxiv.org/abs/2411.05007)</td>
</tr>
<tr>
<td>xDiT 병렬 / ParaDiGMS</td>
<td>GPU 수 비례 지연 단축 / 2–4×</td>
<td>비용 증가·캐싱과 결합 제한</td>
<td>arXiv:2405.14430 / 2305.16317</td>
</tr>
</table>
**현실적 누적 상한: 단일 GPU 5–8배 (낙관 ~10배)** — 그 이상은 증류 없이 불가.
## ④ 로봇 도메인 적용 — 가속 자체는 선점됨, 기하 결합은 공백
<table header-row="true">
<tr>
<td>논문</td>
<td>달성</td>
<td>비고</td>
</tr>
<tr>
<td>**DreamDojo** (NVIDIA, ICML 2026)</td>
<td>**10.81 FPS**, 640×480, 증류 파이프라인</td>
<td>"로봇 월드모델 + 가속"은 이미 논문화 — [arXiv:2602.06949](https://arxiv.org/abs/2602.06949)</td>
</tr>
<tr>
<td>**RoboWorld** (KAIST)</td>
<td>**15.3 FPS**, "Step Forcing" 4-step, 실기 상관 Pearson 0.989</td>
<td>정책 평가 용도 — [arXiv:2607.01060](https://arxiv.org/abs/2607.01060)</td>
</tr>
<tr>
<td>**MotionWAM**</td>
<td>휴머노이드 G1 실기 4.9Hz</td>
<td>완전 rollout이 아닌 중간 feature 우회 — [arXiv:2606.09215](https://arxiv.org/abs/2606.09215)</td>
</tr>
<tr>
<td>기하 일관성 × few-step 결합</td>
<td>**미발견 (공백 확인)**</td>
<td>Geo4D·RoboTransfer·PAIWorld는 증류 없음, 증류 논문들은 기하 없음. 최근접 VideoScene(arXiv:2504.01956)은 3D 씬 재구성 용도</td>
</tr>
</table>
# 3. Novelty 정의
> **"멀티뷰 기하 일관성(포인트맵 정합)을 보존함을 검증한 최초의 few-step 로봇 월드모델"**
- "로봇 월드모델을 빠르게"만으로는 DreamDojo·RoboWorld와 정면 충돌 → 살아남는 축은 **기하 보존 검증 + 보존 방법**
- 4계열 조사 전체에서 교차 확인된 공백: 증류 8편 전부 기하 지표 미검증, 로봇 적용 조사에서도 결합 미발견
- 방법 기여 후보: 기하 제약(포인트맵 consistency loss)을 증류 목적함수(DMD)에 통합하는 *geometry-aware distillation*; 증류 후 새 병목이 되는 VAE·conditioner 고정비 공격(CausVid가 미해결로 남긴 "VAE 청킹 지연 하한"과 연결)
# 4. 품질 평가 체계
## 기존 증류 연구들의 품질 검증 방식 (원문 확인)
<table header-row="true">
<tr>
<td>논문</td>
<td>자동 지표</td>
<td>인간 평가</td>
<td>teacher 대비 주장</td>
</tr>
<tr>
<td>CausVid</td>
<td>VBench competition 3축 (MovieGen 128 프롬프트) + VBench-Long 946 프롬프트 (Total 84.27, 리더보드 1위)</td>
<td>Prolific, 29 프롬프트×3명=쌍당 87건 쌍대비교</td>
<td>"comparable to teacher" — 이겼다고 주장 안 함</td>
</tr>
<tr>
<td>Self-Forcing</td>
<td>VBench Total 84.31 vs teacher 84.26</td>
<td>MovieGenBench 1,003 프롬프트×1명 쌍대비교</td>
<td>+0.05인데 분해 시 **Quality는 하락(85.07<85.30)**, Semantic이 견인</td>
</tr>
<tr>
<td>Causal Forcing</td>
<td>VBench + VisionReward(6.326) + Dynamic Degree + **자체 큐레이션 100 프롬프트** ("many VBench prompts involve minimal motion")</td>
<td>10명×10 프롬프트 전 방법 동시 순위 (1.64 vs 2.87)</td>
<td>같은 속도에서 품질만 개선</td>
</tr>
<tr>
<td>Seaweed-APT</td>
<td>FID/CLIP/VBench를 보조로 격하 — "automatic metrics to be less accurate than user studies"</td>
<td>**50,328 쌍대비교×3명**, 3축: Visual Fidelity / Structural Integrity / Text Alignment</td>
<td>축별 정직 보고: fidelity +10.4%, **구조 −38.5%**</td>
</tr>
</table>
**공통 패턴 4가지 (전부 원문 확인):**
1. 전부 무참조(no-reference) 지각 지표 + 인간 선호 — **GT 대비 픽셀 지표(PSNR/SSIM/LPIPS)와 FVD는 4편 모두 미사용** (T2V라 GT가 없음)
2. **VBench 16개 차원(**[**arXiv:2311.17982**](https://arxiv.org/abs/2311.17982)**) 어디에도 3D/기하 일관성 차원 없음** — Subject Consistency는 DINO 2D 외형 유사도일 뿐
3. **Seaweed-APT의 자기 반증**: VBench는 1-step(82.00) vs 25-step(82.15) 거의 무손실인데 인간 평가 Structural Integrity는 −38.5% 붕괴 → **VBench가 구조 붕괴를 못 잡는다는 실증**
4. **Causal Forcing의 지적**: "the collapse that pathologically inflates the motion metric" — 붕괴가 Dynamic Degree를 부풀리고, 움직임을 죽이면 consistency 점수가 높아짐. 자동 지표가 양방향으로 속음
## 우리의 평가 체계 (제안)
우리 셋업의 구조적 이점: **시뮬레이션 RGB-D GT가 있어** T2V 논문들이 못 쓴 참조 기반 지표를 전부 쓸 수 있고, Geo4D Table 1이 이미 그 프로토콜을 제공 → **"증류 전후로 Geo4D Table 1 지표가 유지되는가"라는 자연스러운 실험 설계**.
<table header-row="true">
<tr>
<td>층위</td>
<td>지표</td>
<td>역할</td>
<td>GT 필요</td>
</tr>
<tr>
<td>**기하 (핵심 기여)**</td>
<td>AbsRel, δ₁ (깊이), cross-view mIoU, 포인트맵 정합 오차</td>
<td>증류 후에도 3D가 맞는가 — Geo4D Table 1과 직접 비교</td>
<td>✅</td>
</tr>
<tr>
<td>픽셀/참조</td>
<td>PSNR, LPIPS, FVD</td>
<td>기존 증류 논문이 못 쓴 GT 기반 검증</td>
<td>✅</td>
</tr>
<tr>
<td>지각/무참조</td>
<td>VBench 일부 차원 + VisionReward</td>
<td>기존 연구와의 비교 가능성 (표준어)</td>
<td>❌</td>
</tr>
<tr>
<td>**기능 (최종 심판)**</td>
<td>FoundationPose 궤적 추출 오차, 정책 성공률</td>
<td>빨라진 비디오로 로봇이 실제로 일하는가</td>
<td>환경</td>
</tr>
</table>
**정성 평가**: Seaweed-APT 3축 프로토콜 차용 + Structural Integrity를 도메인 확장 — "물체/그리퍼 구조 보존 + **뷰 간 일치**(두 뷰가 같은 3D 사건을 보여주는가)"를 별도 축으로 side-by-side 순위 평가(Causal Forcing 방식). 스텝 수별 기하 붕괴 실패 사례 갤러리 병행.
**예상 스토리**: "25→4스텝에서 VBench는 유지되는데 AbsRel·mIoU가 무너진다"를 보이는 것 자체가 1차 발견 (Seaweed-APT의 −38.5%가 강하게 시사) → 기하 손실을 막는 증류 목적함수가 방법론 기여로 연결.
# 5. 실행 로드맵
**순서 원칙**: 가설 검증 먼저 → 싼 실험(추론)으로 리스크 제거 → 비싼 실험(학습).
## 📊 진행 현황판 (2026-08-23 기준)
<table header-row="true">
<tr>
<td>단계</td>
<td>내용</td>
<td>상태</td>
<td>핵심 결과 한 줄</td>
</tr>
<tr>
<td>0</td>
<td>병목 실측 + 원인 분해</td>
<td>✅</td>
<td>평가 포함 25.4초 / 순수 추론 21.75초, UNet 25회가 93%. 실제 고정비 1.6초(cond 0.73+디코딩 0.88) — "4.6초"은 평가 오버헤드 포함이었음(발견 11)</td>
</tr>
<tr>
<td>1</td>
<td>스텝 축소 스윕 + 블러 검증 + 정성</td>
<td>✅</td>
<td>**해리 발견** (PSNR↑ vs 기하·지각↓), 무손실 한계 = 8스텝(2.3배 단축)</td>
</tr>
<tr>
<td>2</td>
<td>평가 파이프라인</td>
<td>✅</td>
<td>bench 스크립트 7종 — 스텝/ckpt만 바꾸면 지표표 출력</td>
</tr>
<tr>
<td>3</td>
<td>Self-Forcing 추론 재현</td>
<td>✅</td>
<td>같은 5090에서 8.3 FPS — 증류는 평균 붕괴를 회피함을 확인</td>
</tr>
<tr>
<td>4</td>
<td>Causal-Forcing 3단계 해부</td>
<td>✅</td>
<td>첫 청크 0.45초/12FPS — **속도=구조, 품질=학습**의 분리 확인</td>
</tr>
<tr>
<td>5</td>
<td>DMD 학습 스모크 (Wan 1.3B)</td>
<td>✅</td>
<td>5090 1장에서 루프 안정 가동 — 32GB 레시피 7종 확립</td>
</tr>
<tr>
<td>6-1</td>
<td>ODE 쌍 생성</td>
<td>✅</td>
<td>200쌍 ×2버전(v2=진짜 궤적 중간상태 포함), 1000쌍 확장 예정</td>
</tr>
<tr>
<td>6-2</td>
<td>ODE 회귀 초기화 v1·v2</td>
<td>✅ 실패 규명</td>
<td>둘 다 안개 바닥 — **MSE 평균 붕괴 확정 → DMD 필연성 입증**</td>
</tr>
<tr>
<td>**6-3**</td>
<td>**Geo4D DMD (바닐라 증류)**</td>
<td>✅ (08-22)</td>
<td>2000스텝 1시간, 피크 24.4GB. **안개 해소(PSNR 14.4→19.9, 선명도·다양성 teacher 수준)**. 깊이는 전역 스케일 편향(앵커 전 0.175 vs teacher 0.072, 수정 마스크) — 6-4(a)의 입력 앵커로 해소. 최적 ckpt step1600(DIAG std비 ≈1.0)</td>
</tr>
<tr>
<td>6-4</td>
<td>geometry-aware loss</td>
<td>🔶 (a) 완료</td>
<td>입력 앵커(조건 프레임+외부 파라미터, 학습 없음)로 **student AbsRel 0.175→0.086 (teacher 0.072)** — 왼쪽 동급(p=0.43), 오른쪽 +0.022(p=0.01). 학습 loss(6b×3)는 DMD와 충돌해 실패. 평가 오염(발견 10) 수정 후 확정</td>
</tr>
<tr>
<td>7</td>
<td>도메인 목표 검증 + 고정비 공격</td>
<td>🔶 준정적 ✅</td>
<td>✅ 준정적 2초 달성: uc 생략 2.44 → bf16 1.64 → +compile **1.18초** (Step 7-②, 품질 드리프트 노이즈 수준). 휴머노이드 0.3–0.5초는 미달(남은 고정비 cond 0.30+디코딩 0.33)</td>
</tr>
<tr>
<td>8</td>
<td>기능 평가 + 논문화</td>
<td>예정</td>
<td>FoundationPose 궤적 오차·정책 성공률</td>
</tr>
</table>
## ⭐ 확보된 발견 (전부 자체 실측 — 1~4는 08-21, 5~11은 아래 목록)
<table header-row="true">
<tr>
<td>#</td>
<td>발견</td>
<td>핵심 수치</td>
<td>의미</td>
</tr>
<tr>
<td>1</td>
<td>**해리(dissociation)**</td>
<td>1스텝에서 PSNR +4.8% "개선" vs LPIPS +21%·CV-Chamfer +10.5% 악화</td>
<td>픽셀 지표 단독 평가의 오판 위험 실증 — Figure 1 후보</td>
</tr>
<tr>
<td>2</td>
<td>**지표 붕괴 순서**</td>
<td>다양성(−77%) → LPIPS → 기하 → PSNR(끝까지 오판)</td>
<td>지표 민감도 위계 자체가 분석 기여</td>
</tr>
<tr>
<td>3</td>
<td>**CV-Chamfer의 함정**</td>
<td>v1 student: CV −31% "개선"인데 AbsRel 3배 악화</td>
<td>두 뷰가 닮은 오답을 내면 정합 지표가 속음 → 기하 평가는 세트 필수</td>
</tr>
<tr>
<td>4</td>
<td>**MSE 평균 붕괴**</td>
<td>v1(의사 궤적)·v2(진짜 궤적) 모두 σ700 loss ≈ 0.40 정체 + 전역 안개</td>
<td>궤적 품질 문제가 아니라 목적함수 본질 → 분포 매칭(DMD)이 필연</td>
</tr>
</table>
**⭐ 발견 5–11 (08-22~23)** — † = 철회/수정된 항목
- **발견 5 — DMD가 평균 붕괴를 꺾는다**: v1·v2 ODE 회귀는 1200스텝 내내 안개, DMD는 400회 업데이트로 선명도 0.0125·다양성 0.0223 = teacher(0.0122/0.0227) 수준. 중간 검증용 DIAG(std(x0)/std(z_teacher))가 최적 ckpt 예측(≈1.0인 step1600 최고, 1.1 과상승 step2000 악화) 
- **발견 6** † "teacher 오른쪽 뷰 붕괴" — 철회. 발견 10의 평가 오염이었음 (수정 후 teacher 오른쪽 0.064) 
- **발견 7 — GT 감독 기하 loss는 DMD와 충돌**: 뷰 간 깊이 비 loss(β 1/3/10, 표적 teacher/GT) 3회 모두 PSNR −3.5dB·LPIPS 0.36으로 붕괴 — DMD는 teacher 분포를 통째로 증류하므로 그에 모순되는 감독은 품질을 희생시킴. (단 표적이 오염된 GT였으므로 "깨끗한 GT로도 충돌"은 미확인) 
- **발견 8 — Geo4D 두 뷰는 가중치 100% 공유**: [wrappers.py](http://wrappers.py)의 뷰별 디코더 deepcopy가 얕은 복사의 _modules 공유 때문에 무효(ckpt output_blocks 795키 전부 동일). 코드 발견으로 보고 가치 
- **발견 9 — 학습 없는 few-step teacher는 샘플러와 무관하게 붕괴**: 재노이징 3스텝 다양성 −47%·선명도 −12%(1스텝 −78%) — 논문 표의 필수 대조군. PSNR/AbsRel은 오히려 좋아 보임(해리 재확인) 
- **발견 10 — 오른쪽 뷰 GT의 53%가 가짜 유효 픽셀**: 데이터셋이 무효(xyz=0)까지 참조 프레임으로 변환해 xyz=이동벡터로 채움 → 제외 전 teacher 오른쪽 0.418, 제외 후 0.064. 오른쪽 뷰 관련 이전 결론(구조 오류·뷰 간 불일치) 전부 철회. Geo4D 논문 평가의 처리 방식 확인 필요 
- **발견 11 — 시간 측정에 평가 스캐폴딩 2.9초 포함**: 순수 추론은 teacher 21.75초, student 3스텝 2.81초, 1스텝 2.01초 (발견 11 표). 실제 고정비 1.6초 
- **발견 12 — 입력 앵커는 teacher도 개선**: robust affine 앵커로 teacher AbsRel 0.072→0.060 (10/10 샘플) — 학습 없는 기하 개선 
## 📌 목표 대비 현재 위치 (2026-08-23)
**목표 = 추론 시간 단축 + Geo4D의 장점(기하 일관성·비디오 품질) 유지**
<table fit-page-width="true" header-row="true">
<tr>
<td>축</td>
<td>지표 (20샘플, bf16)</td>
<td>teacher 25스텝</td>
<td>우리 3스텝 + 앵커</td>
<td>판정</td>
</tr>
<tr>
<td>속도</td>
<td>순수 추론 (10프레임×2뷰)</td>
<td>21.80초</td>
<td>**1.64초 (13.3배)** / +compile 1.18초†</td>
<td>✅ 준정적 2초 달성. fp32 2.81 → uc 생략 2.44 → bf16 1.64 (20샘플 무손실) → compile 1.18 († 5샘플만 확인). 휴머노이드 0.3–0.5초는 미달</td>
</tr>
<tr>
<td>품질</td>
<td>선명도 / 시드 다양성</td>
<td>0.0134 / 0.0227</td>
<td>**0.0136 / 0.0224**</td>
<td>✅ 동급 (p=0.95 / 0.84). 학습 없는 3스텝은 0.0119 / 0.0120 (−47%)로 붕괴</td>
</tr>
<tr>
<td>기하</td>
<td>왼쪽 AbsRel / 뷰 간 정합 CV</td>
<td>0.067 / 0.169</td>
<td>**0.076 / 0.137**</td>
<td>✅ 왼쪽 +0.008, CV 개선 (p=0.006)</td>
</tr>
<tr>
<td>기하</td>
<td>오른쪽 AbsRel (가짜 픽셀 제외)</td>
<td>0.066</td>
<td>0.088</td>
<td>🔶 +0.022 (p<0.001) — 작지만 유의. 앵커 없이는 0.205</td>
</tr>
<tr>
<td>품질</td>
<td>PSNR / LPIPS</td>
<td>20.62 / 0.118</td>
<td>20.43 / 0.136</td>
<td>🔶 PSNR −0.19 (p=0.026), LPIPS +0.018 (p<0.001, 상대 +15%) — 학습 없는 3스텝도 LPIPS +0.014</td>
</tr>
<tr>
<td>1스텝</td>
<td>위 전부</td>
<td>—</td>
<td>1.65초, AbsRel 0.116, LPIPS 0.177, 다양성 0.0117</td>
<td>❌ 품질 미달 (다양성 절반, 팔 영역 블러)</td>
</tr>
</table>
**↓ 아래 표는 08-23 오전의 구 판(10샘플·fp32, 평가 코드 포함 시간) — 기록용으로 남김**
<table header-row="true">
<tr>
<td>축</td>
<td>지표</td>
<td>teacher (25스텝)</td>
<td>(구 표) 우리 3스텝, 10샘플 fp32</td>
<td>판정</td>
</tr>
<tr>
<td>속도</td>
<td>10프레임×2뷰 생성 시간</td>
<td>24.7초</td>
<td>**3스텝 bf16 1.64초 (13.3배), +compile 1.18초 (18.5배†)** — 경로: fp32 2.81 → uc 생략 2.44 → bf16 1.64 → compile 1.18. 1스텝 1.65초(fp32). 평가 코드 포함 시 5.69/4.90</td>
<td>✅ **준정적 2초 달성 (08-23, Step 7-②)**: 3스텝 순수 추론 2.38 → bf16(디코더+UNet+cond) **1.64초** → +torch.compile **1.18초 (teacher 대비 18.5배)**. **bf16은 20샘플에서 무손실 확인**(AbsRel 0.0813→0.0819, LPIPS 0.1360→0.1357, 다양성 동일). compile(1.18초)은 5샘플만(PSNR +0.03~+0.12, AbsRel ±0.005) — 20샘플 확인 대기. 1스텝 2.01초는 다양성 절반·LPIPS 0.186으로 품질 미달이라 목표 달성으로 세지 않음. teacher 순수 추론 21.75초, 실제 고정비 cond 0.73+디코딩 0.88=1.6초 (발견 11)</td>
</tr>
<tr>
<td>기하</td>
<td>AbsRel 왼/오 (가짜 픽셀 제외)</td>
<td>0.080 / 0.064</td>
<td>**0.086 / 0.086** (+0.006 p=0.43 / +0.022 p=0.01)</td>
<td>🔶 왼쪽 동급(p=0.43) / 오른쪽 소폭 유의 손실(+0.022, 상대 +34%, p=0.01). 앵커 없이는 0.146/0.205</td>
</tr>
<tr>
<td>기하</td>
<td>뷰 간 정합 CV-Chamfer / 뷰비오차</td>
<td>0.183 / 0.036</td>
<td>0.161 / 0.029</td>
<td>✅ 유지</td>
</tr>
<tr>
<td>품질</td>
<td>선명도 / 시드 다양성</td>
<td>0.0122 / 0.0227</td>
<td>**0.0125 / 0.0223**</td>
<td>✅ 유지 (학습 없는 3스텝은 0.0107 / 0.0120으로 붕괴)</td>
</tr>
<tr>
<td>품질</td>
<td>PSNR / LPIPS</td>
<td>20.09 / 0.125</td>
<td>19.91 (p=0.08) / 0.146 (+0.020, p<0.001)</td>
<td>🔶 PSNR 동급(p=0.08), LPIPS 유의 손실(+0.020, 상대 +16%, p<0.001)</td>
</tr>
<tr>
<td>1스텝</td>
<td>위 전부</td>
<td>—</td>
<td>AbsRel 0.122, LPIPS 0.186, 다양성 절반</td>
<td>❌ 미달 (팔 영역 잔여 블러)</td>
</tr>
</table>
**어디까지 왔나 (정직한 문장)**: 3스텝 student는 순수 추론 7.7배 빠르면서 **선명도·다양성·왼쪽 뷰 기하·PSNR은 teacher와 동급, 오른쪽 뷰 기하(+0.022)와 LPIPS(+0.020)는 작지만 유의한 손실** — "무손실"은 아님. 학습 없는 4스텝 축소(T4: LPIPS +0.013에 다양성 −53%)과 비교하면 다양성을 지키면서 비슷한 LPIPS 비용을 치른 셋. "시간 단축"은 4.3배까지 왔고 2초 목표는 스텝이 아닌 고정비를 줄여야 하는 별도 과제(Step 7). **5.8초 분해**: UNet 3회 ≈ 1.2초(호출당 ~0.4초 — student는 CFG가 없어 배치 20, teacher는 CFG로 배치 40이라 0.81초) + 고정비 ≈ 4.6초(conditioner 1.8 + VAE 디코딩 1.75 + 인코딩 0.5 + 평가용 loss UNet 1회 0.8). 1스텝 5.0초 = 0.4 + 4.6. 즉 모델 계산은 21%뿐, 79%가 고정비.
**무엇이 효과가 있었나**: ① DMD(분포 매칭) → 선명도·다양성 복원 ② 입력 앵커(조건 프레임+외부 파라미터, 학습 없음) → 깊이 스케일 복원 (teacher도 0.072→0.060 개선). 효과 없었던 것: ODE 회귀 초기화(평균 붕괴), GT 감독 기하 loss(DMD와 충돌).
**논문 메인 표 — 최종 수치 (08-23, data_seed 1234, 배치 20 = 뷰 40, 오른쪽 GT 가짜 픽셀 제외, 다양성은 3배치×시드 4)**:
<table header-row="true">
<tr>
<td>방법</td>
<td>학습</td>
<td>스텝</td>
<td>순수 추론</td>
<td>PSNR↑</td>
<td>AbsRel↓ (L/R)</td>
<td>LPIPS↓</td>
<td>선명도 (GT 0.0214)</td>
<td>다양성</td>
<td>CV↓</td>
</tr>
<tr>
<td>Geo4D teacher (Euler)</td>
<td>—</td>
<td>25</td>
<td>21.80초</td>
<td>20.62</td>
<td>0.066 (.067/.066)</td>
<td>0.118</td>
<td>0.0134</td>
<td>0.0227</td>
<td>0.169</td>
</tr>
<tr>
<td>teacher 스텝만 축소 (Euler)</td>
<td>없음</td>
<td>4</td>
<td>4.84초</td>
<td>21.22</td>
<td>0.064</td>
<td>0.132</td>
<td>0.0121</td>
<td>0.0131 (−42%)</td>
<td>0.165</td>
</tr>
<tr>
<td>teacher 스텝만 축소 (재노이징)</td>
<td>없음</td>
<td>3</td>
<td>2.81초</td>
<td>21.30</td>
<td>0.066</td>
<td>0.132</td>
<td>0.0119</td>
<td>0.0120 (−47%)</td>
<td>0.167</td>
</tr>
<tr>
<td>**DMD student + 뷰별 앵커 (우리)**</td>
<td>DMD</td>
<td>3</td>
<td>**1.64초** (bf16; +compile 1.18초†)</td>
<td>20.43</td>
<td>0.082 (.076/.088)</td>
<td>0.136</td>
<td>**0.0136**</td>
<td>**0.0224**</td>
<td>0.136</td>
</tr>
<tr>
<td>DMD student + 뷰별 앵커</td>
<td>DMD</td>
<td>1</td>
<td>1.65초 (uc 생략 fp32; bf16 미실측)</td>
<td>20.55</td>
<td>0.115</td>
<td>0.178</td>
<td>0.0105</td>
<td>0.0115</td>
<td>0.164</td>
</tr>
</table>
**paired vs teacher (n=40, Wilcoxon)** — 우리(S3b): PSNR −0.19 (p=0.026) / AbsRel +0.015 (p<0.001, 왼 +0.008, 오 +0.022) / LPIPS +0.018 (p<0.001) / **선명도 +0.0001 (p=0.95, 동급) / 다양성 −0.0004 (p=0.84, 동급)** / CV −0.034 (p=0.006, 개선). 학습 없는 3스텝(T3r): PSNR +0.68 / AbsRel −0.001 (p=0.87, 동급) / LPIPS +0.014 / 선명도 −0.0015 / **다양성 −0.011 (−47%)**.
**정직한 해석**: 학습 없는 few-step은 깊이 정확도(AbsRel)는 유지하지만 다양성을 절반 잃고 선명도도 떨어짐. DMD student는 다양성·선명도를 teacher 수준으로 복원하는 대신 AbsRel +0.015·LPIPS +0.018을 치른다 — **트레이드오프이지 무료 점심이 아님**. 월드모델 용도(여러 미래 샘플링)에서는 다양성이 필수라는 것이 우리 주장의 근거가 되어야 함.
**bf16 품질 확인 (20샘플, precise_6a_main20_fast.txt)**: S3b fp32→bf16 PSNR 20.43→20.43 / AbsRel 0.0813→0.0819 / LPIPS 0.1360→0.1357 / 선명도 0.01352→0.01362 / 다양성 0.02233→0.02242; S1b 0.1153→0.1161, LPIPS 0.1776→0.1766 — **무손실**. 위 우리 행은 bf16 수치이고 시간 1.64초도 bf16 설정 실측(시간·품질 동일 설정). † compile 1.18초의 품질은 5샘플만 확인(AbsRel ±0.005), 20샘플 확인 대기 중. (fp32 이전 수치: 2.81/2.44초, AbsRel 0.081, 선명도 0.0135, 다양성 0.0223) 이전 초안(배치 10):
<table header-row="true">
<tr>
<td>방법</td>
<td>학습</td>
<td>스텝</td>
<td>시간</td>
<td>PSNR↑</td>
<td>LPIPS↓</td>
<td>AbsRel↓</td>
<td>선명도</td>
<td>다양성</td>
</tr>
<tr>
<td>Geo4D teacher</td>
<td>—</td>
<td>25</td>
<td>24.7초</td>
<td>20.09</td>
<td>0.125</td>
<td>0.072</td>
<td>0.0122</td>
<td>0.0227</td>
</tr>
<tr>
<td>teacher, 스텝만 축소 (재노이징)</td>
<td>없음</td>
<td>3</td>
<td>5.8초</td>
<td>20.84</td>
<td>0.137</td>
<td>(재측)</td>
<td>0.0107</td>
<td>0.0120</td>
</tr>
<tr>
<td>ODE 회귀 초기화 (v2)</td>
<td>회귀</td>
<td>4</td>
<td>7.9초</td>
<td>14.42</td>
<td>—</td>
<td>0.547</td>
<td>안개</td>
<td>시드 동일</td>
</tr>
<tr>
<td>**DMD student + 입력 앵커 (우리)**</td>
<td>DMD</td>
<td>3</td>
<td>**5.8초**</td>
<td>19.91</td>
<td>0.146</td>
<td>**0.086**</td>
<td>0.0125</td>
<td>0.0223</td>
</tr>
<tr>
<td>DMD student + 입력 앵커</td>
<td>DMD</td>
<td>1</td>
<td>5.0초</td>
<td>20.35</td>
<td>0.186</td>
<td>0.122</td>
<td>0.0088</td>
<td>0.0115</td>
</tr>
</table>
## 📝 논문에 쓸 것 (2026-08-23 기준)
> 	**주장 (현재 수치)**: 분포 매칭 증류(DMD) + 입력 앵커 + bf16으로 Geo4D를 25→3스텝(순수 추론 21.8→1.64초, **13.3배**)으로 줄이면서 선명도·시드 다양성·왼쪽 기하를 teacher 수준으로 유지. 학습 없는 few-step은 다양성 −47%로 붕괴하는 반면 DMD는 이를 복원 — 대신 오른쪽 AbsRel +0.022·LPIPS +0.018의 작지만 유의한 손실(트레이드오프)을 정직히 보고.

**표/그림 후보**: ① 메인 표(위 초안, T3r AbsRel 재측 필요) ② Figure 1 = 해리(스텝↓ PSNR↑ vs LPIPS·다양성·기하↓) ③ 지표 붕괴 순서 그래프 ④ 정성 그리드(GT/teacher/T3r/student, 시드 다양성) ⑤ 시간 분해 표(발견 11) ⑥ ablation: ODE 회귀 초기화 · GT 감독 loss · 앵커 유무(a/b/c)
**평가 기여**: 해리(발견 1), 붕괴 순서(2), CV-Chamfer 함정(3: 퇴화 정합·스케일 의존), 오른쪽 GT 가짜 픽셀(10), 평가 스캐폴딩 시간(11) — "픽셀 지표만 보면 증류가 허사로 보인다"가 핵심 메시지
**부정적 결과(그대로 실을 것)**: ODE 회귀 초기화의 평균 붕괴(4), GT 감독 기하 loss와 DMD의 충돌(7), 학습 없는 few-step의 다양성 붕괴(9)
**코드/데이터 발견**: 두 뷰 가중치 공유(8), 오른쪽 GT 아티팩트(10) — Geo4D 저자 평가 방식 확인 후 기술
**한계(명시)**: 단일 태스크(apple)·고정 시드 10샘플·단일 GPU, 1스텝 미달(다양성 절반), 오른쪽 뷰·LPIPS 잔여 격차, 2초 미달(2.81초), 기능 평가(Step 8) 미수행
## ⏭️ 다음 액션
1. ~~메인 표 완성~~ ✅ (08-23, 20샘플, precise_6a_main20.txt)
2. ~~2초 달성 (Step 7)~~ ✅ bf16 1.64초, 20샘플 품질 무손실 확인 (08-23). 선택: compile 1.18초의 20샘플 품질 확인(15분) — 논문에 1.18초를 쓰려면 필요
3. **잔여 격차 축소 (학습)**: 오른쪽 AbsRel +0.022 · LPIPS +0.020 — 더 긴 DMD(4000스텝) · lr 조정 · EMA, 각 1시간. 기하 loss는 다시 안 함(발견 7)
4. **검증 확장**: 다른 태스크/에피소드로 일반화, Geo4D 저자 평가의 가짜 픽셀 처리 확인
5. Step 8 기능 평가는 위 1~3 후
---
## 📝 단계별 상세 실험 로그 (시간순 원본 — 재현·디버깅용 기록)
### ✅ Step 0 — 병목 실측 + 원인 분해 (08-21)
**한 줄 요약**: 논문의 "10프레임에 30초" 주장을 우리 GPU에서 재현하고, 그 시간이 정확히 어디에 쓰이는지 분해했다.
**무엇을 했나**: 공개 체크포인트(apple 태스크)로 추론 3회 반복 측정(bench_[30s.py](http://30s.py)) → 모델 내부 4곳에 타이머를 심어 컴포넌트별 시간 분해(bench_[profile.py](http://profile.py)).
<table header-row="true">
<tr>
<td>컴포넌트</td>
<td>시간</td>
<td>비중</td>
<td>비고</td>
</tr>
<tr>
<td>**UNet denoising**</td>
<td>**21.14초**</td>
<td>**83.8%**</td>
<td>26회 호출(25스텝 + 평가용 loss 1회), 호출당 813ms</td>
</tr>
<tr>
<td>conditioner (CLIP·VAE 인코딩)</td>
<td>1.81초</td>
<td>7.2%</td>
<td>조건 준비 고정비</td>
</tr>
<tr>
<td>VAE 디코딩</td>
<td>1.75초</td>
<td>7.0%</td>
<td>latent→픽셀 (877ms×2)</td>
</tr>
<tr>
<td>VAE 인코딩</td>
<td>0.48초</td>
<td>1.9%</td>
<td></td>
</tr>
</table>
- 총 **25.4초** 실측 (논문 "4090에서 ~30초" 재현 성공, 5090이 한 세대 빠른 분만큼 단축), 피크 VRAM 13.2GB
**의미 (쉽게)**: 느린 이유는 모델이 커서가 아니라 **같은 계산(UNet)을 25번 반복**하기 때문. 메모리는 13GB로 여유 → 병목은 순수하게 "반복 횟수". 시간 공식으로 쓰면 **고정비 4.6초 + 스텝당 0.84초** — 1차 공격은 스텝 수(증류), 그다음이 고정비. **[08-23 정정]** 이 측정은 log_images 경로라 평가 스캐폴딩(loss용 UNet 1회·GT 재구성 디코딩·GT 인코딩·conditioner 중복) 2.9초를 포함 — 순수 추론은 21.75초, 실제 고정비 1.6초 (발견 11).
### ✅ Step 1b — cross-view 정합 스윕 (08-21)
**한 줄 요약**: "두 뷰의 3D가 서로 맞는가"를 재는 우리만의 지표(CV-Chamfer)를 설계해 스텝 수별로 측정했다.
**지표 설계**: 모델이 두 뷰의 포인트맵을 같은 참조 좌표계에서 예측한다는 사실(sgm/models/[diffusion.py:52](http://diffusion.py:52)로 확인)을 이용 — 두 예측 점군 사이 대칭 chamfer 거리를 재고, 장면 겹침이 원래 부분적이므로 GT 점군끼리의 chamfer를 기준선으로 ratio(pred/GT)를 계산.
**결과**:
- **발견 A**: 스텝 수와 무관하게 ratio ≈ 2.0 (25스텝 2.004 / 4스텝 2.067) — 뷰 간 정합 갭은 샘플링이 아니라 **모델 자체**에 있다
- 단, 숨은 결함은 아님: 논문 Table 1의 자기 보고(mIoU 0.56–0.70, AbsRel 0.03–0.11)와 우리 실측(AbsRel 0.0699, δ₁ 0.935)이 일치하는 공개된 성능 수준의 재확인. 뷰별 깊이 오차(~7%)가 두 뷰에서 독립 발생해 합성되는 효과도 포함
- **발견 B**: 25→4스텝의 기하 손상은 +3.1%뿐 (16스텝은 오히려 −5.8% — 배치 3이라 노이즈 범위)
- **시간 모델 확정**: 1스텝이어도 ~5.5초 → **2초 목표의 진짜 병목은 고정비 4.6초**(VAE 디코딩·conditioner·평가용 loss) — CausVid가 자인한 VAE 지연 하한(arXiv:2412.07772 Sec. 6)과 같은 지점
**의미 (쉽게)**: 기하 정합은 원래도 완벽하지 않다 → "기하 유지"에서 한발 더 나아가 "기하 **개선**"(mIoU 0.6↑)까지 novelty가 될 수 있다.
### ⭐ Step 1c — 해리(dissociation) 발견: 확장 스윕 (스텝 6지점 × 배치 10, 08-21)
**한 줄 요약**: 스텝을 극단(1·2)까지 줄이면 픽셀 지표는 좋아지는 척하는데 기하는 단조로 무너진다 — 두 지표가 서로 반대로 움직이는 "해리"를 발견했다.
<table header-row="true">
<tr>
<td>스텝</td>
<td>시간/배치</td>
<td>PSNR↑</td>
<td>CV-Chamfer↓</td>
<td>판정</td>
</tr>
<tr>
<td>25</td>
<td>25.8초</td>
<td>19.75</td>
<td>0.1758</td>
<td>기준</td>
</tr>
<tr>
<td>8</td>
<td>11.4초</td>
<td>20.06</td>
<td>0.1692</td>
<td>**무손실 + 2.3배 단축**</td>
</tr>
<tr>
<td>4</td>
<td>7.9초</td>
<td>20.35</td>
<td>0.1796</td>
<td>경계</td>
</tr>
<tr>
<td>2</td>
<td>6.4초</td>
<td>20.66</td>
<td>0.1919 (+9.2%)</td>
<td>해리 시작</td>
</tr>
<tr>
<td>1</td>
<td>5.4초</td>
<td>20.64</td>
<td>0.1943 (+10.5%)</td>
<td>해리 확정 (4.8배 단축)</td>
</tr>
</table>
- GT 기준선: CV-Chamfer 0.1105. ratio로 보면 1.59→1.76으로 단조 악화하는데 PSNR은 +4.5% "개선"
**해석 (쉽게)**: 스텝이 적으면 모델이 "가능한 미래들의 평균"(=흐릿한 그림)을 내놓는다. 평균은 픽셀 단위 오차(PSNR)에는 유리하지만 실제 3D 구조는 망가진다. 근거 문헌: Causal Forcing §3.2 "learning a conditional mean averages over frames, which manifests as blurred visual results"(arXiv:2602.02214) + perception-distortion tradeoff(Blau & Michaeli, CVPR 2018, arXiv:1711.06077).
**함의 4가지**:
1. 픽셀 지표 단독 평가는 오판 — PSNR만 보면 "1스텝이 최고"로 보임 (평가 기여의 직접 근거, Figure 1 후보)
2. 학습 없는 축소 한계선 확정 = 8스텝(안전) / 4스텝(경계)
3. 기하 보존 증류의 정량 표적 = 1–2스텝에서 CV-Chamfer를 8스텝 수준(0.169)으로 회복
4. 1스텝 5.4초 중 스텝 비용은 0.8초뿐 → 2초 목표는 고정비 공격으로. **[08-23 정정]** 이 "고정비 4.6초"의 2.9초는 평가 스캐폴딩이었고 실제 고정비는 1.6초 (발견 11)
**주의**: 배치 10의 AbsRel 절대값(0.147±0.204)은 어려운 샘플이 섞여 3배치 때(0.070)보다 크다 — 절대값보다 같은 배치 내 스텝 간 추세를 믿을 것. 결과 파일: bench_out/full_sweep_results.txt
### ⭐ Step 1d — 블러 가설 3중 검증 (6지점 × 시드 4 × 배치 2, 08-21)
**한 줄 요약**: "적은 스텝 = 평균(블러) 수렴"이라는 해석을 문헌 인용이 아니라 세 가지 독립 실험으로 직접 증명했다.
<table header-row="true">
<tr>
<td>테스트</td>
<td>원리</td>
<td>결과 (25→1스텝)</td>
</tr>
<tr>
<td>① LPIPS (지각 거리)</td>
<td>딥 feature 기반이라 블러에 민감</td>
<td>**+20.7% 악화** (0.1222→0.1474) — PSNR은 +4.8% "개선"과 정반대</td>
</tr>
<tr>
<td>② 선명도 (Laplacian 분산)</td>
<td>고주파 에너지 직접 측정 = 블러의 물리적 정의</td>
<td>**−12.5% 단조 하락**. 25스텝조차 GT의 70%(0.0139 vs 0.0197)</td>
</tr>
<tr>
<td>③ 시드 간 다양성</td>
<td>같은 입력을 시드 4개로 생성 — "평균 수렴"이면 샘플들이 같아져야 함</td>
<td>**−77% 붕괴** (0.0188→0.0043) = 사실상 결정론적 평균 출력</td>
</tr>
</table>
**추가 발견 — 지표 붕괴 순서**: 다양성(가장 민감, 8스텝부터 −25%) → LPIPS(4스텝 +11.4%) → 기하 CV-Chamfer(2스텝 +9.2%) → PSNR/SSIM(끝까지 안 무너지고 오히려 개선) — 이 민감도 서열 자체가 분석 기여 후보. 지각 기준 무손실 경계는 16스텝(+0.5%), 8스텝은 경미(+3.9%). SSIM도 PSNR처럼 오판 쪽(+0.5%).
**근거 문헌**: Causal Forcing §3.2(arXiv:2602.02214) / perception-distortion tradeoff(arXiv:1711.06077) / LPIPS(arXiv:1801.03924). 결과 파일: bench_out/blur_test_results.txt
### ✅ Step 1e — 정성(육안) 확인으로 Step 1 완결 (08-21)
**무엇을 했나**: GT/25/8/1스텝 비교 그리드(RGB·깊이, 행=설정, 열=프레임)와 "같은 입력 × 시드 4개" 다양성 그리드를 생성해 직접 확인 (bench_out/qualitative/).
**확인된 것**:
- 1스텝: **움직이는 팔만 반투명 얼룩이고 배경은 선명** — 불확실한(여러 미래가 가능한) 움직임 영역만 평균돼 뭉개지는 "조건부 평균의 국소 서명". 픽셀 대부분이 배경이라 전역 PSNR이 오히려 오르는 이유까지 이 그림 하나로 설명된다
- 다양성 그리드: 25스텝은 시드마다 팔 자세가 다름(여러 미래를 샘플링하는 정상 동작) vs 1스텝은 4시드가 사실상 동일한 그림(평균 하나만 출력) — 정량 다양성 −77%의 시각화
**결론 (Step 1 완결)**: 스텝 축소의 실패 모드 = 움직임 영역의 조건부 평균 붕괴. PSNR/SSIM은 이를 놓치고(오히려 좋게 평가), LPIPS·다양성·CV-Chamfer·육안이 잡는다. rgb_left.png·diversity_left.png는 논문 Figure 후보.
### Step 1a — 픽셀 지표 1차 스윕 (배치 3, 08-21 — 시간순으로는 맨 처음 실험)
**무엇을 했나**: 학습 없이 샘플링 스텝만 25→16→8→4로 줄이며(config `num_steps` 변경만으로) PSNR·MSE·AbsRel 측정.
**결과**: 시간은 예측대로 감소 — 4스텝 7.9초/배치(−69%), **프로파일링 예상 7.3초와 일치해 Step 0의 시간 모델이 검증됨**. 그런데 픽셀 지표는 무너지기는커녕 "개선" (4스텝 PSNR +2.5%, AbsRel −6.1%).
**의미 (쉽게)**: 두 해석이 경합했다 — (a) 조건이 강한 예측이라 정말 4스텝이면 충분하다 vs (b) 픽셀 지표가 블러를 좋게 평가해 붕괴를 못 본다. 이 경합이 1b~1e의 검증 실험들을 촉발했고, 최종 판정은 (b)였다.
### ✅ Step 2 — 평가 파이프라인 고정
**목적**: 이후 모든 단계(예비실험·증류 전후·최종 검증)가 **같은 자(尺)로 측정**돼야 비교 가능하므로, 스텝 수·체크포인트만 바꾸면 동일 프로토콜로 지표표가 나오는 스크립트군을 확립.
**구성 (서버 ~/4dgen/notebooks/)**:
- bench_30s(추론 시간) / bench_profile(컴포넌트 분해) / bench_step_sweep(픽셀 지표) / bench_crossview_sweep(기하) / bench_full_sweep(6지점 통합+표준편차)
- bench_blur_test(LPIPS·SSIM·선명도·시드 다양성) / bench_qualitative(그리드·GIF) / bench_student_sweep(학생 평가, teacher 기준선 자동 비교) / geo4d_sigma_diag(σ별 진단)
### ✅ Step 3 — Self-Forcing 추론 재현 (08-21)
**한 줄 요약**: "증류가 만드는 속도"를 같은 5090에서 실측해 기준점을 확보하고, 증류 출력에는 블러 붕괴가 없다는 것을 확인했다.
**결과**:
- 81프레임(832×480) 생성 **9.8초 = 8.14~8.28 FPS** (프롬프트 3개에서 매우 안정) — README 주장 "4090에서 ~10 FPS"와 정합 (SDPA 폴백 + torch.compile 미사용 상태임을 감안하면 합리적 수준)
- 로그의 current_timestep 값(1000→937.5→833→625)으로 청크당 정확히 4스텝만 도는 것을 실시간 확인
- 같은 GPU에서 Geo4D 0.39 FPS 대비 약 21배 — 단, 태스크·모델이 다르므로(단일뷰 T2V 1.3B vs 멀티뷰 RGB-D 2.4B) 통제된 비교가 아니라 참고치. 통제 비교는 Step 6에서 수행
**핵심 관찰**: 생성 비디오에 Geo4D 1스텝 같은 평균 붕괴 얼룩이 없음 → **DMD 분포 매칭은 "평균"이 아니라 "분포에서 그럴듯한 한 샘플"을 배우므로 평균 수렴을 원리적으로 회피**한다는 주장의 시각적 확인
**연구 서사 연결**: 단순 축소는 붕괴(Step 1 실측) → 증류는 회피(Step 3 확인) → 그럼 증류가 **기하 일관성**도 지키는가? (미답 = 우리의 기여 지점)
**재현 메모**: 전용 conda 환경 self_forcing(torch 2.11+cu128, video_policy와 분리) / **--use_ema 필수**(이 ckpt에는 generator_ema 키만 존재) / torchvision 0.26이 write_video 제거 → imageio로 대체 / sm_120에 flash-attn 설치 불가 → flash_attention 함수 자체에 SDPA 폴백 삽입(직접 호출 6곳 커버)
### ✅ Step 4 — Causal-Forcing 3단계 해부 (08-21)
**한 줄 요약**: 공개된 학습 중간 체크포인트 3개를 같은 조건으로 돌려 "무엇이 속도를 만들고 무엇이 품질을 만드는지"를 분리했다.
**무엇을 했나**: chunkwise 3단계 ckpt(causal_ode → causal_cd → causal_forcing, 각 5.3GB)를 동일 프롬프트·동일 4스텝·--report_timing으로 실행.
**결과**:
- 세 스테이지 모두 첫 청크 **0.45초 / 12.0 FPS로 소수점까지 동일** → 속도는 순전히 아키텍처(4스텝+KV캐시)가 결정하고, 3단계 학습 파이프라인은 **오직 품질만** 바꾼다 — 6-3 설계에서 속도(구조)와 품질(학습)을 독립 최적화할 수 있다는 지침
- 첫 청크 0.45초(5090 실측) — 휴머노이드 목표(0.3–0.5초)가 소비자 GPU의 청크 지연 수준에서 이미 닿는다는 실측 근거
- 스테이지별 품질 비교 비디오 확보 (output/stage2a_ode · 2b_cd · 3_final — 논문 주장대로면 2a에 아티팩트, 3으로 갈수록 정리)
**재현 메모**: wan_models 심볼릭 링크 필요 / --use_ema는 causal_cd만(ckpt 키가 스테이지마다 다름: 나머지는 generator 키) / SDPA·write_video 패치 동일 적용 / Stage 1(ar_diffusion, 다중스텝 teacher)은 미수행(옵션) / 남은 해부: trainer 코드 리딩으로 injectivity(Lemma 3.2) 처리 확인 — Geo4D 멀티뷰 cross-attention에서 같은 함정 회피용
### ✅ Step 5 — DMD 학습 스모크 (Wan 1.3B, 08-21)
**한 줄 요약**: 64×H100용 증류 학습을 5090 1장에 욱여넣는 진단 실험 — 다섯 번의 실패를 하나씩 규명해 학습 루프를 안정 가동시켰다.
**결과**: step 9+ 진행, 스텝당 ~4초, 메모리가 매 스텝 16.87→17.33GB로 정확히 반복(누적 0), 피크 28.4GB < 31.4GB — 지속 학습 가능 확인. 검증 범위는 "루프가 돈다 + 메모리 안정"까지이고 수렴 품질은 미검증(wandb를 꺼서 loss 추이 미관찰).
**32GB 레시피 — 실패와 해결의 순서대로**:
<table header-row="true">
<tr>
<td>막힌 지점</td>
<td>원인</td>
<td>해결</td>
</tr>
<tr>
<td>체크포인트 로드 실패</td>
<td>real_score가 14B 체크포인트 요구</td>
<td>1.3B로 교체 (스모크 목적에는 유효)</td>
</tr>
<tr>
<td>RAM 부족 SIGKILL ×2</td>
<td>T5가 fp32(22GB)로 구성 + 로드 시 이중 버퍼</td>
<td>T5 bf16 구성 + mmap 로드 (22→11GB)</td>
</tr>
<tr>
<td>rollout 크래시</td>
<td>최소 21프레임 하드코딩 vs 우리 3프레임</td>
<td>min_num_frames = min(21, max)로 완화</td>
</tr>
<tr>
<td>CUDA OOM (backward)</td>
<td>모델 3벌의 fp32 마스터 + optimizer 20.8GB</td>
<td>전 모델 순수 bf16(mixed_precision 해제, −10GB) + AdamW8bit(optim→5GB)</td>
</tr>
<tr>
<td>NCCL 할당 실패</td>
<td>VRAM 만재 상태에서 NCCL 워크스페이스 요구</td>
<td>단일 GPU라 gloo 백엔드로 우회</td>
</tr>
<tr>
<td>정체불명 GPU 잔류 11GB</td>
<td>**FSDP CPUOffload가 동결 T5를 forward 후 GPU에 방치** (backward가 없어 반환 계기가 안 돔)</td>
<td>메모리 프로브 3줄로 진단 → 인코딩 순간만 GPU로 올렸다 내리는 명시적 셔틀 패치</td>
</tr>
<tr>
<td>잔여 부족</td>
<td>활성화 메모리</td>
<td>rollout 3프레임 · 해상도 축소 · EMA off · num_workers 0 · 단계 간 empty_cache</td>
</tr>
</table>
**교훈 (Step 6-3에 그대로 적용)**: 메모리 예산은 산수가 아니라 **프로브 실측**으로 잡을 것 — 결정타(11GB)는 어떤 계산에도 없던 라이브러리 내부 동작이었고, 3줄짜리 메모리 프로브가 한 번에 잡았다. 동결 모듈(conditioner·VAE)의 GPU 잔류가 최대 복병.
**자산**: configs/self_forcing_dmd_5090.yaml + 패치 4파일(trainer/distillation · utils/wan_wrapper · utils/distributed · model/base)
### 🔶 Step 6-1 · 6-2 v1 — ODE 쌍 생성 + 초기화 1차 (08-21)
**6-1 (데이터 생성)**: teacher(25스텝)로 (노이즈 시드, 최종 latent) 쌍 200개 생성 — 노이즈는 시드 번호만 있으면 재현되므로 최종 latent만 저장해 쌍당 0.4MB, 총 80MB. 21.7초/쌍 ≈ 72분 소요.
**6-2 v1 (학습)**: student(UNet 1.54B만 학습, 조건 인코더·VAE 동결, bf16 + 8bit AdamW, VRAM 19.5GB, 4초/스텝)를 의사 궤적 x_σ = z+σε 회귀로 300스텝 학습 — loss 0.45→0.22로 순조롭게 감소.
**평가 — "성공처럼 보이는 실패"**:
<table header-row="true">
<tr>
<td>지표 (1스텝)</td>
<td>student v1</td>
<td>teacher 기준선</td>
<td>판정</td>
</tr>
<tr>
<td>CV-Chamfer (뷰 간 정합)</td>
<td>0.1338 (−31% "개선", GT 0.1105 근접)</td>
<td>0.1943</td>
<td>액면상 대폭 개선</td>
</tr>
<tr>
<td>AbsRel (뷰별 깊이 정확도)</td>
<td>0.3867 (**3배 악화**)</td>
<td>0.1321</td>
<td>붕괴</td>
</tr>
<tr>
<td>PSNR</td>
<td>16.4 (−4.2dB)</td>
<td>20.6</td>
<td>붕괴</td>
</tr>
</table>
**발견 3 (쉽게)**: 두 뷰가 서로는 잘 맞는데 둘 다 정답에서 멀다 = **서로 닮은 오답**. 두 뷰가 같은 안개를 그리면 정합 지표는 좋아져 버린다 → **CV-Chamfer 단독은 퇴화 정합에 속는다. 기하 평가는 {뷰 간 정합 + 뷰별 정확도 + 지각} 세트여야 한다** (Seaweed-APT·Causal Forcing이 보고한 지표 함정과 같은 계열을 자체 지표에서 확인 — 평가 기여 강화)
**후속 진단 (재학습 없이 forward 비교 + 정성)**:
- σ별 진단: 모든 σ에서 회귀 loss가 큰 폭 감소(σ700 −38% / σ70.5 −38% / σ2.3 −64%)했는데 품질은 붕괴 = **학습 목적함수 수준의 loss-품질 해리**. 이유: σ=700(거의 순수 노이즈) 입력에서 MSE 회귀의 최적해는 "가능한 출력들의 평균" = 안개. loss가 내려갈수록 평균에 가까워질 뿐이다
- 부수적 발견: 스케줄에 σ≈0 슬롯이 포함돼 학습 스텝의 25%가 "입력을 그대로 내보내기" 항등 과제에 낭비 (유효 학습 ~225스텝)
- 정성 확인: student@4·@1 모두 전역 안개 + 시드 4개 동일 (조건부 평균 확정, bench_out/student_qual/)
- student@4조차 안개인 이유: 중간 σ를 의사 궤적(z+σε)으로 배워서, 추론 때는 자기 출력(안개)이 입력으로 들어오는데 이건 학습 때 본 분포 밖이라 다듬기 스텝이 무력 — **CausVid가 진짜 ODE 궤적으로 초기화하는 이유를 우리 손으로 재발견** → v2로 이어짐
### ✅ Step 6-2 v2 — 진짜 궤적으로 재시도, 결론 확정 (08-22)
**무엇을 했나**: v1의 원인 후보를 전부 제거한 재실험 — ① 6-1 확장판으로 teacher 25스텝 경로의 **실제 중간 상태**(σ=700/70.5/2.3에 가장 가까운 지점의 x)를 denoiser 래퍼로 캡처해 200쌍 재생성(쌍당 1.6MB), ② σ≈0 슬롯 제거, ③ 학습량 4배(1200스텝), ④ σ별 분리 로깅. 구현 이슈: teacher 추론이 CFG로 배치를 2배([x,x])로 불려 캡처 텐서가 40프레임 — 절반만 사용하는 패치로 해결.
**결과**:
- σ별 로그: σ2.3(거의 깨끗한 입력)은 0.158→0.05~0.08로 순조롭게 감소 / **σ700·σ71은 ~0.39–0.40에서 정체** — v1의 바닥 0.44와 사실상 같은 지점
- 평가: 4스텝 PSNR 14.42 / AbsRel 0.547 / CV 0.137 — v1과 동일한 안개 바닥 (근소 악화)
**확정 결론 — 발견 4 (쉽게)**: 진짜 궤적을 줘도 소용없었다. 첫 스텝(σ700)의 출력이 흐릿하면 그 다음 스텝의 입력은 어차피 궤적 밖이라 on-manifold 학습 효과가 무력화된다. 근본 원인은 궤적의 품질이 아니라 **"노이즈→샘플을 MSE 회귀로 한 방에" 과제 자체의 평균 붕괴**.
**문헌 대조로 기대 재보정**: ODE init이 부드러운 출력을 내는 건 레시피상 원래 정상 — Causal Forcing App. C.2가 "중간 단계에 teacher 아티팩트가 남고 최종 품질은 DMD 정제에 의존한다"고 자인한 것과 합치(arXiv:2602.02214). 즉 **선명함을 복원하는 것은 분포 매칭(DMD) 단계의 역할이고, v2는 그 출발점으로서는 유효**.
**자산**: v2 ckpt(ode_init_geo4d_[v2.pt](http://v2.pt)) = 6-3 DMD의 초기화 / 스크립트 geo4d_ode_gen_v2(궤적 캡처) · geo4d_ode_init_train_v2(σ별 로깅) · bench_student_sweep(teacher 기준선 자동 비교)
### ✅ Step 6-3 — 바닐라 DMD(6a) 실행·결과 (08-22)
**한 줄 요약**: Step 5 레시피를 Geo4D에 이식한 DMD가 한 번에 돌았고(추가 디버깅 0회), v1·v2가 1200스텝 내내 못 벗어난 안개를 400회의 generator 업데이트로 걷어냈다 — 단, 깊이 정확도는 돌아오지 않았다.
**실행**: `geo4d_dmd_train.py --max_steps 2000` (ODE 쌍 생성 종료 후 자동 체인). 1.7초/스텝 → 1시간, 피크 24.4GB(예상 28GB보다 여유). generator 업데이트 5스텝당 1회 = 총 400회. ODE 쌍은 데이터셋 한계로 284개(142샘플×시드 2)에서 자연 종료 — DMD는 조건만 쓰므로 무관.
**학습 중 DIAG (동일 시드 생성의 std 비율, 1.0=teacher)**: step 0 0.79/0.70(full/1step) → 400 0.87/0.73 → 1000 **0.99**/0.86 → 1600 1.06/0.97 → 2000 1.11/1.05 (과상승). MSE(x0, z_teacher)는 0.44→0.87로 증가 — 평균 하나가 아니라 분포의 샘플을 내기 시작했다는 정상 신호.
**정량 결과 (배치 10, student는 재노이징 샘플러, teacher는 EulerEDM)** — ⚠ 이 표의 AbsRel은 발견 10 이전(오른쪽 GT 가짜 픽셀 포함)이고 시간은 평가 코드 포함값. 확정 수치는 6-4 블럭의 "마스크 수정 후 재평가" 표와 발견 11 참조:
<table header-row="true">
<tr>
<td>모델</td>
<td>스텝</td>
<td>시간/배치</td>
<td>PSNR↑</td>
<td>AbsRel↓</td>
<td>CV-Chamfer↓</td>
</tr>
<tr>
<td>teacher</td>
<td>25</td>
<td>25.8초</td>
<td>19.75</td>
<td>0.147</td>
<td>0.1758</td>
</tr>
<tr>
<td>teacher (학습 없는 축소)</td>
<td>4 / 1</td>
<td>7.9 / 5.4초</td>
<td>20.35 / 20.64</td>
<td>0.126 / 0.132</td>
<td>0.1796 / 0.1943</td>
</tr>
<tr>
<td>v2 ODE-init</td>
<td>4</td>
<td>7.9초</td>
<td>14.42</td>
<td>0.547</td>
<td>0.137 (퇴화)</td>
</tr>
<tr>
<td>**DMD step 1600**</td>
<td>**3**</td>
<td>**5.9초**</td>
<td>**19.63**</td>
<td>**0.201**</td>
<td>**0.167**</td>
</tr>
<tr>
<td>DMD step 1600</td>
<td>2</td>
<td>5.4초</td>
<td>19.74</td>
<td>0.220</td>
<td>0.169</td>
</tr>
<tr>
<td>DMD step 1600</td>
<td>1</td>
<td>5.0초</td>
<td>20.03</td>
<td>0.213</td>
<td>0.190</td>
</tr>
<tr>
<td>DMD step 1000</td>
<td>3 / 1</td>
<td>—</td>
<td>19.55 / 18.39</td>
<td>0.212 / 0.240</td>
<td>0.184 / 0.165</td>
</tr>
<tr>
<td>DMD step 2000</td>
<td>3 / 1</td>
<td>—</td>
<td>19.21 / 19.85</td>
<td>0.227 / 0.252</td>
<td>0.176 / 0.181</td>
</tr>
</table>
**해석 (쉽게)**:
- **안개 해소 확정**: PSNR 14.4→19.6(teacher 25스텝과 동급), 육안으로도 3스텝 student는 배경·로봇 모두 선명 — 발견 4("복원은 DMD의 역할") 실증
- **기하는 미달 = 6a의 기하 손실**: AbsRel 0.20 vs teacher 0.13–0.15. CV-Chamfer 0.167은 teacher보다 좋아 보이지만 AbsRel이 나빠 발견 3(CV 단독 함정)이 또 적용됨 — 세트로 봐야 "픽셀은 되고 깊이는 안 된다"가 보임. 이게 6-4가 메울 갭
- **체크포인트 선택**: step 1600이 최고, 2000은 과상승(std 1.1)으로 악화 — DIAG std 비율이 ≈1.0일 때가 최적이라는 값싼 모니터 확보 (발견 5)
- **1스텝은 미해결**: 정량은 양호하나 육안으로 움직이는 팔에 잔여 블러, 시드 4개가 유사 — 3스텝이 현재 납품 수준
- **속도**: 5.0–5.9초 = teacher 대비 4.4–5.2배. Step 0 예측대로 고정비(4.6초)가 남은 병목
**정밀 분석 (08-22, bench_precise_**[**6a.py**](http://6a.py)** — 샘플별 paired + LPIPS·선명도·다양성, 배치 10 = 뷰 샘플 20, 다양성은 3배치×시드 4)**:
<table header-row="true">
<tr>
<td>설정</td>
<td>s/생성</td>
<td>PSNR↑</td>
<td>AbsRel↓</td>
<td>LPIPS↓</td>
<td>선명도 (GT 0.0197)</td>
<td>CV↓</td>
<td>시드 다양성</td>
</tr>
<tr>
<td>T25 teacher</td>
<td>26.0</td>
<td>20.21</td>
<td>0.146</td>
<td>0.123</td>
<td>0.0128</td>
<td>0.183</td>
<td>0.0188</td>
</tr>
<tr>
<td>T4 (학습 없는 축소)</td>
<td>8.1</td>
<td>20.77</td>
<td>0.135</td>
<td>0.138</td>
<td>0.0116</td>
<td>0.173</td>
<td>0.0107</td>
</tr>
<tr>
<td>T1</td>
<td>5.5</td>
<td>21.18</td>
<td>0.137</td>
<td>0.149</td>
<td>0.0112</td>
<td>0.166</td>
<td>0.0042</td>
</tr>
<tr>
<td>**S3 DMD step1600**</td>
<td>5.9</td>
<td>20.07</td>
<td>**0.266**</td>
<td>0.142</td>
<td>**0.0132**</td>
<td>**0.160**</td>
<td>**0.0204**</td>
</tr>
<tr>
<td>S1 DMD step1600</td>
<td>5.1</td>
<td>20.13</td>
<td>0.269</td>
<td>0.182</td>
<td>0.0101</td>
<td>0.169</td>
<td>0.0098</td>
</tr>
</table>
**paired (S3 − T25, Wilcoxon)**: PSNR −0.13 (p=0.15, 동급) / **선명도 +3% (p=0.024, 오히려 우수)** / 다양성 +9% (83% 승, p=0.06, 동급↑) / CV −0.023 (p=0.38) / LPIPS +0.019 (p<0.001, 악화 — 학습 없는 T4의 +0.015와 같은 급) / **AbsRel +0.120 — 20샘플 중 0개 승, p<0.001, 전 샘플에서 약 2배 악화** (예: 0.057→0.226, 0.036→0.151)
**해석 (쉽게)**: S3는 **선명하고 다양하고 두 뷰가 서로 맞는데, GT 깊이에서는 일관되게 빗나간다**. 평균 붕괴는 완전히 해소(다양성·선명도가 teacher 이상). 전 샘플에서 균일하게 ≈2배 나빠다는 패턴은 구조 파괴보다 **깊이 스케일/오프셋 편향** 가능성을 시사(확인 필요: 스케일 정렬 AbsRel). 부수 관찰: T4·T1의 AbsRel이 T25보다 좋음 → **AbsRel도 PSNR처럼 평균을 보상하는 distortion 지표**라 해리의 일부는 perception-distortion 트레이드오프 — 단 2배는 그걸로 설명되기에 너무 큼. S1은 LPIPS +0.059·다양성 절반·선명도 하락 = 부분 붕괴 잔존.
**결과 파일**: bench_out/precise_6a.txt (+ _raw.json, 샘플별 전수)
**⭐ 스케일 진단 (08-22, bench_scale_**[**diag.py**](http://diag.py)**) — AbsRel 악화는 구조 오류가 아니라 깊이 스케일·오프셋 편향**:
<table header-row="true">
<tr>
<td>정렬</td>
<td>T25 AbsRel</td>
<td>S3 AbsRel</td>
<td>S3−T25 (paired)</td>
<td>격차 잔존율</td>
<td>S3 우수 샘플</td>
</tr>
<tr>
<td>없음 (raw)</td>
<td>0.137</td>
<td>0.201</td>
<td>+0.064</td>
<td>100%</td>
<td>5%</td>
</tr>
<tr>
<td>스케일 1개 (median)</td>
<td>0.110</td>
<td>0.139</td>
<td>+0.030</td>
<td>46%</td>
<td>25%</td>
</tr>
<tr>
<td>**스케일+오프셋 (affine)**</td>
<td>0.113</td>
<td>0.115</td>
<td>**+0.002**</td>
<td>**3%**</td>
<td>**45%**</td>
</tr>
</table>
- 피팅된 스케일 s=median(gt/pred): teacher 0.979 vs **student 0.909** — student는 깊이를 일관되게 약 9% 멀게 예측(pred/gt 평균비 1.088), 20샘플 중 19개에서 s<1. 로봇/물체 마스크 영역과 배경에서 동일하게 악화(fg 0.18→0.25, bg 0.13→0.20) → 국소 구조가 아니라 전역 편향
- **결론 (쉽게)**: DMD student의 3D 구조는 teacher와 동급이고, "자를 잘못 들고 재는" 전역 스케일/오프셋만 틀렸다. 가설: generator는 cond-only인데 real_score는 CFG 2.5 — 가이던스가 포인트맵 채널의 절대 스케일을 밀어 student가 그 분포를 따라감(확인 필요)
- **6-4 설계 결정**: 포인트맵 consistency loss(구조) 전에 **스케일 앵커**가 먼저. 후보 ① 추론 시 보정: 조건 프레임의 포인트맵(입력으로 주어짐, GT 불필요)에 예측 프레임 0을 median 스케일 정렬 — 학습 없이 0비용, 격차의 ~46% 회수 기대 ② 학습 시 앵커 loss: 예측 포인트맵 스케일을 조건 프레임/ODE 쌍 z에 묶는 항 ③ real_score 가이던스 스케일 조정
**⭐ 추론 시 스케일 보정 실험 (08-22, bench_cond_**[**calib.py**](http://calib.py)** v2, 학습 없음·GT 불필요)**:
원리: Geo4D는 조건 프레임의 포인트맵을 입력으로 받음 → 예측 프레임 0의 깊이를 조건 포인트맵에 median 스케일로 맞추고 그 s를 전 프레임·양 뷰에 적용. 확인된 전제: 왼쪽 조조 프레임 ≈ GT 프레임 0 (AbsRel 0.013), 오른쪽 조건 포인트맵은 다른 좌표계(0.62)라 **왼쪽(참조 좌표계)에서 구한 s 하나를 양 뷰에 적용**해야 함 (1차 시도는 뷰별 자기 조건으로 해서 오른쪽 실패).
<table header-row="true">
<tr>
<td>뷰</td>
<td>설정</td>
<td>raw</td>
<td>조건 스케일 보정</td>
<td>oracle(GT 스케일)</td>
<td>oracle affine</td>
<td>S3−T25 raw → 보정 후</td>
</tr>
<tr>
<td>왼쪽</td>
<td>T25</td>
<td>0.080</td>
<td>0.072</td>
<td>0.072</td>
<td>0.096</td>
<td></td>
</tr>
<tr>
<td>왼쪽</td>
<td>**S3**</td>
<td>0.146</td>
<td>**0.086 (−41%, 10/10 개선)**</td>
<td>0.086</td>
<td>0.097</td>
<td>**+0.066 → +0.014 (잔존 21%)**</td>
</tr>
<tr>
<td>오른쪽</td>
<td>T25</td>
<td>0.418</td>
<td>0.417</td>
<td>0.406</td>
<td>0.397</td>
<td></td>
</tr>
<tr>
<td>오른쪽</td>
<td>S3</td>
<td>0.768</td>
<td>0.665 (−18%, 10/10 개선)</td>
<td>0.579</td>
<td>0.482</td>
<td>+0.350 → +0.247 (잔존 71%)</td>
</tr>
</table>
- **왼쪽 뷰: 무비용 보정이 oracle과 동일** (s_cond 0.908 = s_oracle 0.909, 소수점 셀째 자리까지) — teacher와의 잔차 +0.014는 오프셋 항(oracle affine에서 +0.0005로 소멸)
- **오른쪽 뷰: 보정 후도 격차 잔존** — S3의 오른쪽 오라클 스케일이 샘플마다 0.47–1.31로 틀어짐(왼쪽은 0.86–0.96으로 일관). 즉 **student는 뷰별로 스케일이 따로 놀고 있다** = 진짜 cross-view 불일치. 이건 스케일 앵커로 안 잡히고 6-4b(뷰 간 consistency loss)의 몴. CV-Chamfer가 이걸 못 잡은 것(S3가 최고였음)도 기록 — 지표 보완 필요
- **6-4 설계 확정**: (a) 전역 스케일 편향 → 조건 프레임 앵커(추론 시 보정은 즉시 포함, 학습 시 앵커 loss로 오프셋까지) + (b) 뷰 간 스케일 일치 → 포인트맵 consistency loss
- 결과: bench_out/cond_calib_6a_v2.txt (+_raw.json). 이 실행부터 데이터셋 RNG 고정(data_seed 1234, num_workers 0) — 이 세트는 오른쪽 뷰가 유독 어려움(teacher도 0.42)
- **주의(재현)**: 배치 10의 샘플이 실행마다 달라짐(데이터셋이 에피소드 내 윈도우를 랜덤 샘플링, RNG 소비 순서에 의존) → 실행 간 절대값 비교 금지, 같은 실행 내 paired만 유효. 데이터셋 RNG 고정 필요. 결과: bench_out/scale_diag_6a.txt
**자산**: ~/Geo4D/dmd_6a/dmd_gen_step{1000,1600}.pt, dmd_[gen.pt](http://gen.pt)(2000), dmd_[fake.pt](http://fake.pt) / 로그 ~/dmd_6a.log, ~/eval_6a.log / 결과 bench_out/student_sweep_results_dmd6a_s\*.txt / 정성 bench_out/dmd6a_qual/
### 🔶 Step 6-4 — geometry-aware DMD (진행 중)
**6-4 (a) 조건 앵커 추론 보정 — ✅ 완료 (08-22)**
**무엇을**: geo4d_[fewstep.py](http://fewstep.py)에 `enable_cond_anchor(model)` 추가 — 샘플러 출력의 왼쪽 프레임 0 깊이를 왼쪽 조건 포인트맵에 median 스케일로 맞춘 s를 양 뷰·전 프레임의 xyz에 적용. 학습·GT 불필요, 비용 0. 통합 평가기 bench_eval_[6x.py](http://6x.py)(데이터 시드 고정, paired Wilcoxon) 신설.
<table header-row="true">
<tr>
<td>설정</td>
<td>s/생성</td>
<td>PSNR</td>
<td>AbsRel 전체</td>
<td>AbsRel 왼쪽</td>
<td>AbsRel 오른쪽</td>
<td>LPIPS</td>
<td>CV</td>
<td>앵커 s</td>
</tr>
<tr>
<td>T25</td>
<td>24.8</td>
<td>20.09</td>
<td>0.249</td>
<td>0.080</td>
<td>0.418</td>
<td>0.125</td>
<td>0.183</td>
<td>—</td>
</tr>
<tr>
<td>S3</td>
<td>5.8</td>
<td>19.91</td>
<td>0.457</td>
<td>0.146</td>
<td>0.768</td>
<td>0.146</td>
<td>0.196</td>
<td>—</td>
</tr>
<tr>
<td>**S3a (+앵커)**</td>
<td>5.8</td>
<td>19.91</td>
<td>**0.375**</td>
<td>**0.086**</td>
<td>0.665</td>
<td>0.146</td>
<td>0.178</td>
<td>0.908±0.033</td>
</tr>
<tr>
<td>S1 / S1a</td>
<td>5.0</td>
<td>20.35</td>
<td>0.497 / 0.437</td>
<td>0.182 / 0.137</td>
<td>0.812 / 0.736</td>
<td>0.186</td>
<td>0.211 / 0.193</td>
<td>0.919</td>
</tr>
</table>
**paired vs T25 (n=20)**: S3a 왼쪽 AbsRel **+0.006, p=0.43, 40% 우수 → teacher와 통계적으로 구분 불가** (보정 전 +0.066, p=0.002). 오른쪽은 +0.35 → +0.25 (p=0.002, 여전히 악화). PSNR·LPIPS는 포인트맵 연산이라 불변.
**결론 (쉽게)**: 전역 스케일 편향은 입력만으로 고칠 수 있고, 왼쪽(참조) 뷰는 이것으로 끝. 남은 문제는 오른쪽 뷰가 왼쪽과 다른 스케일로 그려지는 **뷰 간 불일치** 하나 → 6-4 (c)의 표적. (b) 학습 앵커 loss는 잔차(오프셋 +0.006)가 작아 후순위.
**부수 발견**: 양 뷰를 같은 s로 줄였는데 CV-Chamfer가 0.196→0.178로 "개선" — 절대 거리 지표라 스케일에 비례. CV-Chamfer는 스케일 정규화(예: GT 깊이 평균으로 나눔) 없이는 비교 지표로 부적절 — 뷰 간 스케일 비율(median z_R/z_L) 지표 추가 필요.
**결과**: bench_out/eval_6x_6a_anchor.txt (+_raw.json)
**6-4 (c) 뷰 간 스케일 consistency loss — 🔶 학습 중 (08-22 저녁, ~/Geo4D/dmd_6b)**
**설계**: 전역 스케일에 불변인 뷰 간 깊이 비 r = mean z_R / mean z_L (참조 좌표계). 목표는 1이 아니라 **teacher의 r** (ODE 쌍 z 디코드로 샘플별·프레임별 사전계산, 평균 1.318, 샘플 std 0.33 — 카메라 배치상 r≠1이 정상). L_cv = (log r_student − log r_teacher)², generator 스텝에만. v2 초기 student는 DIAG 쌍에서 r 1.02 vs teacher 1.52 — 오른쪽을 너무 가깝게 그림.
**구현 교훈 (실패 2회)**: ① VAE grad 디코드 4프레임 fp32는 OOM → 1프레임(이미지 2장) + bf16 autocast + gradient checkpoint + expandable_segments, 피크 27.3GB ② 고정 λ=1은 cv 그래디언트가 DMD의 ~500배라 클리핑 후 DMD 신호 소멸 → **x0 수준 그래디언트 균형** λ = β·\|∂L_dmd/∂x0\|/\|∂L_cv/∂x0\| (β=1) 자동 결정, 실측 λ 2.5e-3~4e-2, \|g\| 6a와 동일
**실험**: 6a와 동일 조건 + β=1 → step 1600을 bench_eval_6x(뷰비오차 지표 추가)로 6a step1600과 paired 비교. 성공 기준: 오른쪽 AbsRel(앵커 후)·뷰비오차 하락, PSNR·선명도 유지. 중간(step 400): std 비율 0.813 (6a 동시점 0.867 — cv 항이 DMD 진행을 늦춤), DIAG r 아직 ≈1.0.
**탐색 기록 (08-22 밤, 3회 시도)**:
- 6b (β=1, teacher r 표적): step 1000까지 DIAG r 1.02→1.04 — 전혀 안 움직임, 선명도는 6a보다 느림(std 0.933 vs 0.993). 중단(step1000 ckpt 보존)
- **발견 6 — teacher의 오른쪽 뷰도 틀려 있다**: 142샘플 GT 뷰 비 평균 0.788 vs teacher 1.318 (DIAG 쌍: GT 0.92, teacher 1.52, v2 student 1.02). 데이터셋 코드 확인(spartan_video_[dataset.py:898/1206](http://dataset.py:898/1206)): GT pointmap_right는 inv(cam_extr_ref)@cam_extr로 **참조 카메라 프레임으로 변환됨**, cond_pointmaps_right는 변환 없이 자기 카메라 프레임(1235행) → 앞서 "조건 vs GT0 오른쪽 0.62"의 원인. 즉 평가 좌표계는 맞고, teacher가 일부 샘플에서 오른쪽 뷰 스케일을 크게 틀리는 것(정밀 분석의 2/r 0.92, 6/r 0.56, 8/r 2.8 샘플). → teacher r은 표적으로 부적절, **GT r을 표적으로** (학습 샘플 GT 포인트맵은 컨디션 사전계산 때 무료). 이는 novelty 축 "기하 개선(teacher 초과)"의 구체적 기회
- 6b2 (β=10, GT r 표적): 40회 업데이트만에 r 1.02→0.83→step 500 0.63–0.69로 **GT(0.92)를 지나 과조정**, 선명도 정체/하락(std 0.770, 6a 0.904) — cv가 DMD를 압도. 중단
- 6b3 (β=3, GT r 표적): step 500에 r 1.02→0.70 — DIAG 쌍 GT(0.92)를 지나 **데이터셋 평균 r(0.79) 쪽으로 수렴** = 샘플별 r을 조건에서 추론하지 못하고 평균으로 회귀. 선명도 정체(std 0.78 vs 6a 0.90). 최종(08-23 아침, step1600 평가): DIAG r은 0.86까지 올라 GT(0.92) 근방, 뷰비오차 0.51→0.30으로 개선 — **그러나 나머지 전부 붕괴**: PSNR 19.91→16.57(−3.5dB, 20/20 악화), LPIPS 0.146→0.357, AbsRel 왼쪽 0.146→0.231, 오른쪽 0.77→1.94(앵커 후 1.82). 뷰 비는 맞추었지만 양 뷰를 다 망가뜨리면서 맞춘 것. CV-Chamfer는 또 "최고"(0.128) — 함정 3번째 사례
- **핵심 통찰 (발견 7)**: 세 시도가 모두 선명도를 잃은 이유 — **DMD는 teacher의 분포를(오른쪽 뷰 오류까지) 그대로 증류하므로, teacher와 모순되는 기하 loss는 증류 목적함수와 정면 충돌**한다 (real_score는 r≈1.3을, cv loss는 0.8을 요구). GT 감독 loss로 teacher를 넘어서려면 DMD와 싸워야 하고 그 비용이 선명도. 반면 (a)의 조건 앵커는 입력 정보만 쓰므로 충돌이 없었고 왼쪽 뷰를 완전히 고쳤다.
- **(c′) 오른쪽 뷰 입력 앵커 — ✅ 구현·평가 (08-23)**: 배치의 cam_extr/cam_extr_right(프레임별 11×4×4, 고정 카메라)로 T=inv(E_ref)@E_right 변환 → 오른쪽 조건 포인트맵을 참조 프레임으로 옮겨 오른쪽 프레임 0 앵커 (`enable_cond_anchor(model, per_view=True)`, 평가 설정 접미사 b). 변환 검증: 변환된 조건 vs GT 오른쪽 프레임 0 AbsRel 0.62→0.098 (왼쪽은 0.013 — 잔차 원인 미확인). teacher는 오른쪽 프레임 0을 변환된 조건과 정확히 같게 냄(s_R=1.000) → teacher의 오른쪽 오류는 프레임 0 스케일이 아님.
<table header-row="true">
<tr>
<td>오른쪽 뷰 AbsRel</td>
<td>평균(10)</td>
<td>중앙값</td>
<td>teacher가 정상인 7샘플 평균</td>
</tr>
<tr>
<td>T25</td>
<td>0.418</td>
<td>0.092</td>
<td>0.093</td>
</tr>
<tr>
<td>S3 (앵커 없음)</td>
<td>0.768</td>
<td>0.303</td>
<td>0.240</td>
</tr>
<tr>
<td>S3a (왼쪽 앵커를 양 뷰에)</td>
<td>0.665</td>
<td>0.178</td>
<td>0.182</td>
</tr>
<tr>
<td>**S3b (뷰별 앵커)**</td>
<td>0.610</td>
<td>0.177</td>
<td>**0.168**</td>
</tr>
</table>
- 왼쪽은 불변(0.086, teacher 0.080). 오른쪽은 뷰별 앵커로 추가 −8%(샘플 6: 0.162→0.087) — 하지만 정상 샘플에서 여전히 teacher의 1.8배(0.168 vs 0.093). 이전 오라클 affine 정렬에서도 오른쪽 잔차(+0.085)가 남았던 것과 합치 → **오른쪽 뷰의 남은 오차는 스케일이 아니라 구조 오류**. 이상치 샘플(8, 9)은 teacher도 붕괴(2.8, 0.39)
- 지표 주의: 뷰비오차(SR)는 뷰별 자기 마스크로 평균을 내서 pred가 GT 무효 영역에 값을 쓰면 왜곡됨(teacher 0.51이 픽셀 AbsRel 0.09와 불일치) → 공통 마스크로 재정의 필요. 6b 계열의 teacher r(1.32) vs GT r(0.79)도 일부는 이 마스크 아티팩트일 수 있음(결론 "GT 감독 loss는 DMD와 충돌"은 PSNR/LPIPS 붕괴로 여전히 유효)
- **다음 진단 후보 (학습 없음, 5분)**: student의 오른쪽 뷰 분기(diffusion_model_2)를 teacher 가중치로 바꿔 끼워 평가 — 오른쪽 오차가 사라지면 DMD가 뷰2 분기(hd 경로)를 망가뜨린 것 → 뷰2 분기 동결/정규화 증류라는 구체적 해법으로 연결. 결과: bench_out/eval_6x_cprime.txt
- **분기 교체 진단 → 발견 8 (08-23): Geo4D의 두 뷰는 가중치를 100% 공유한다.** [wrappers.py:19](http://wrappers.py:19)-20 `diffusion_model_2 = copy(diffusion_model); diffusion_model_2.output_blocks = deepcopy(...)` — 얕은 복사가 `_modules` 딕셔너리를 공유하므로 새 output_blocks가 원본에도 그대로 들어감(토이 테스트로 확인, student ckpt에서도 795개 output_blocks 키 전부 동일). “뷰별 디코더” 의도는 무효, 전체 1.54B = SVD UNet 하나. 오른쪽 뷰를 구분하는 건 spatial_context(hd)와 조건 latent뿐 → 뷰별 분기 교체 진단은 불가. 처음 교체 실험의 “놀라운 개선”은 사실 **teacher 가중치 + 재노이징 3스텝 샘플러**(T3r)였음: PSNR 20.84 / AbsRel 0.230(L 0.067, R 0.392) / LPIPS 0.137 / 앵커 s 1.017 — 학습 없는 few-step teacher가 DMD student보다 수치가 높음. **DMD의 존재 가치 검증 ✅ (08-23, bench_precise_6a --configs T25 T3r T1r S3 S1, data_seed 1234)**:
<table header-row="true">
<tr>
<td>설정</td>
<td>학습</td>
<td>PSNR</td>
<td>AbsRel</td>
<td>LPIPS</td>
<td>선명도 (GT 0.0214)</td>
<td>시드 다양성</td>
</tr>
<tr>
<td>T25 teacher</td>
<td>—</td>
<td>20.09</td>
<td>0.249</td>
<td>0.125</td>
<td>0.0122</td>
<td>0.0227</td>
</tr>
<tr>
<td>T3r teacher 재노이징 3스텝</td>
<td>없음</td>
<td>20.84</td>
<td>0.240</td>
<td>0.137</td>
<td>0.0107 (−12%)</td>
<td>**0.0120 (−47%)**</td>
</tr>
<tr>
<td>T1r teacher 재노이징 1스텝</td>
<td>없음</td>
<td>21.20</td>
<td>0.238</td>
<td>0.151</td>
<td>0.0105</td>
<td>0.0049 (−78%)</td>
</tr>
<tr>
<td>**S3 DMD student 3스텝**</td>
<td>DMD</td>
<td>19.91</td>
<td>0.457</td>
<td>0.146</td>
<td>**0.0125 (teacher 수준)**</td>
<td>**0.0223 (teacher 수준)**</td>
</tr>
<tr>
<td>S1 DMD student 1스텝</td>
<td>DMD</td>
<td>20.35</td>
<td>0.497</td>
<td>0.186</td>
<td>0.0088</td>
<td>0.0115</td>
</tr>
</table>
결론: 학습 없는 few-step teacher는 샘플러와 무관하게(Euler든 재노이징이든) **조건부 평균으로 붕괴**(다양성 −47%, 선명도 −12%)하고 그 덕에 PSNR/AbsRel이 "좋아 보임"(발견 1 해리의 재확인). DMD student만 선명도·다양성을 teacher 수준으로 복원 → **DMD의 기여는 perception 축에서만 입증되며, 논문 표에는 T3r이 ‘학습 없는 few-step 기준선’으로 반드시 들어가야 함**. 이 비교에서 픽셀/깊이 distortion 지표만 보면 DMD가 허사로 보이는 것 자체가 평가 기여의 핵심 사례. 남은 숨제: student의 LPIPS +0.02, 오른쪽 뷰 깊이 구조(정상 샘플 0.168 vs teacher 0.093).
- **robust affine 앵커 (접미사 c, 08-23)**: teacher에는 효과 확실 — T25c 오른쪽 0.418→**0.368**(10/10, p=0.002), 왼쪽 0.080→0.068 = 학습 없이 입력 정보만으로 공개 teacher의 기하를 개선(작은 기여). student 오른쪽은 S3b 0.610 → S3c 0.626으로 불변.
- **오른쪽 뷰 정성 그리드 (bench_out/dmd6a_qual_lr/)**: student3 RGB는 정상, 깊이는 화면 전체가 균일하게 멀게 밀림 — 오차맵이 배경·탁자·로봇 모두 15–20% 분홍(teacher는 팔 가장자리만), 프레임 간 변화 없음 → 구조가 아니라 **전역 편향** (어제 "구조 오류" 판정 정정)
- **프레임별 편향 진단 (bench_frame_**[**diag.py**](http://diag.py)**, 08-23 심야)** — S3 오른쪽: raw 0.768 / 조건 affine 0.616 / 배경 시간 앵커 0.635 / **오라클 전체 affine 0.463 / 오라클 프레임별 affine 0.462** (teacher: 0.418 / 0.402 / 0.429 / 0.379 / 0.377). 왼쪽 S3: 0.146 / 0.087 / 0.085 / 0.085 / 0.085.
	- 프레임별 ≈ 전체 → **시간 드리프트 아님** (가설 기각). 배경 시간 앵커도 무효 (배경은 이미 프레임 간 일관)
	- 왼쪽은 조건 affine = 오라클 (0.087 vs 0.085, 완벽). 오른쪽은 조건 affine(0.616)이 오라클(0.463)에 크게 못 미침 → **오른쪽 앵커의 표적(변환된 조건 프레임)이 부정확** — 변환 조건 vs GT 프레임 0 잔차 0.098(왼쪽 0.013)과 일치. 원인 후보: 프레임별 외부 파라미터(11×4×4)에서 [0]만 사용 — 카메라가 프레임마다 움직이면 틀림(데이터셋은 배치 matmul로 프레임별 변환 적용). **다음 확인: cam_extr가 프레임 간 변하는지**
	- 오라클로도 남는 잔차: student 0.463 vs teacher 0.377 (+0.086) = 진짜 구조 오류는 이만큼(전체 격차 0.35의 1/4)
- **발견 10 (08-23 아침): 오른쪽 뷰 GT에 가짜 유효 픽셀이 35–55%** — check_cond_[transform.py](http://transform.py): 카메라 고정 확인, 변환 inv(E1)@E2는 정확(샘플 1·2에서 조건 vs GT0 AbsRel 0.001/0.006). 그런데 데이터셋이 pointmap_right를 만들 때 무효 픽셀(xyz=0)까지 변환해 **xyz = 이동벡터 t (z 0.18–0.65m)로 채움** → z\>0 마스크를 통과하는 가짜 깊이가 화면의 35–55%. 왼쪽은 변환이 없어 0으로 남아 정상 마스크. 함의: ① 지금까지의 오른쪽 AbsRel·CV·뷰비·스케일 진단은 전부 오염(GT 0.18m 픽셀에서 상대오차 폭발 → teacher 2.8 샘플도 이것) ② teacher는 이 GT로 학습돼 가짜 평면을 재현하도록 배웠을 가능성, student는 그걸 덜 재현 → "student 오른쪽 깊이 문제"의 상당 부분이 평가 아티팩트일 수 있음 ③ 6b 계열의 GT r 표적도 오염된 GT 기반. 수정: bench_eval_6x에 --fix_mask(기본 1) — GT xyz≈t 픽셀 제외, CV·뷰비도 공통 마스크. Geo4D 논문의 mIoU/AbsRel 평가가 이 가짜 픽셀을 어떻게 처리했는지도 확인 필요.
**⭐ 마스크 수정 후 재평가 (08-23, eval_6x_maskfix, 가짜 픽셀 53% 제외) — 오른쪽 뷰 문제는 대부분 평가 오염이었다**:
<table header-row="true">
<tr>
<td>설정</td>
<td>PSNR</td>
<td>AbsRel 전체</td>
<td>왼쪽</td>
<td>오른쪽</td>
<td>LPIPS</td>
<td>뷰비오차</td>
<td>시간</td>
</tr>
<tr>
<td>T25 teacher</td>
<td>20.09</td>
<td>0.072</td>
<td>0.080</td>
<td>**0.064** (이전 0.418)</td>
<td>0.125</td>
<td>0.036</td>
<td>24.7초</td>
</tr>
<tr>
<td>T25c (+affine 앵커)</td>
<td>20.09</td>
<td>0.060</td>
<td>0.068</td>
<td>0.052</td>
<td>0.125</td>
<td>0.028</td>
<td>24.9초</td>
</tr>
<tr>
<td>S3 student</td>
<td>19.91</td>
<td>0.175</td>
<td>0.146</td>
<td>0.205</td>
<td>0.146</td>
<td>0.062</td>
<td>5.8초</td>
</tr>
<tr>
<td>**S3b (+뷰별 스케일 앵커)**</td>
<td>19.91</td>
<td>**0.086**</td>
<td>0.086</td>
<td>**0.086**</td>
<td>0.146</td>
<td>0.029</td>
<td>5.8초</td>
</tr>
<tr>
<td>S3c (+affine 앵커)</td>
<td>19.91</td>
<td>0.085</td>
<td>0.089</td>
<td>0.081</td>
<td>0.146</td>
<td>0.032</td>
<td>5.8초</td>
</tr>
<tr>
<td>S1c</td>
<td>20.35</td>
<td>0.122</td>
<td>0.138</td>
<td>0.105</td>
<td>0.186</td>
<td>0.040</td>
<td>5.0초</td>
</tr>
</table>
- paired vs T25: S3b 왼쪽 +0.006 (p=0.43, 동급) / 오른쪽 +0.022 (p=0.01) / S3c 오른쪽 +0.016 (p=0.049). **student 3스텝 + 입력 앵커 = teacher 대비 AbsRel +0.014, 4.3배 빠름, 선명도·다양성 teacher 수준** — 사실상 본 연구의 주 결과
- **철회**: 발견 6("teacher 오른쪽 뷰 붕괴") 철회 — teacher 오른쪽은 0.064로 왼쪽보다 좋음. 6b 계열의 GT r 표적(가짜 픽셀 포함)도 오염 — 다만 "GT 감독 loss가 DMD와 충돌해 PSNR/LPIPS 붕괴"는 여전히 유효(오염된 표적이라 더 심했을 수는 있음). "오른쪽 구조 오류"·"뷰 간 스케일 불일치" 판정도 철회 — 남은 건 전역 스케일(앵커로 해결)과 잔차 +0.016
- **남은 격차**: LPIPS +0.020 (p<0.001), 오른쪽 AbsRel +0.016 — 이제는 학습 측(DMD 세부 조정, 더 긴 학습, lr) 영역. 지표 표준: bench_eval_6x --fix_mask 1, data_seed 1234
- (원래 메모) 다음 방향 (c′) — 오른쪽 뷰도 입력 앵커로: 카메라 리그가 고정이라 외부 파라미터(extrinsics.npz, 데이터셋 254행)를 안다 → 오른쪽 조건 포인트맵을 inv(cam_extr_ref)@cam_extr_right로 참조 프레임으로 변환하면 오른쪽 뷰도 프레임 0 앵커 가능 (데이터셋이 GT에 쓰는 바로 그 변환). 학습 없음·DMD와 무충돌·teacher도 같이 고쳐질 가능성 — "기하 개선(teacher 초과)"을 입력 정보로 달성. 그 다음에야 학습 loss(스케일이 아닌 구조)를 다시 검토 설계 쌍점: 포인트맵 스케일은 latent에서 선형이 아니라 VAE 디코드가 필요 → 프레임 일부만 grad로 디코드해 (i) 뷰 간 median 깊이 비 loss 또는 (ii) 미분 가능 chamfer. 메모리 예산: 현재 피크 24.4GB, 여유 ~7GB.
**6-3 설계 기록 (6-4가 그대로 상속)**:
- 역할 분담: generator = v2 student(학습) / real_score = teacher 동결(진짜 분포의 기준) / fake_score = teacher 사본(학습, student 분포 추적)
- 메모리 설계: 1.54B×3벌 bf16 + 8bit optim ≈ 28GB 경계선 — Step 5 교훈대로 컨디셔너(CLIP·VAE) 셔틀 + 메모리 프로브를 처음부터 넣고 시작
- 데이터: ODE 쌍(1000개 확장 중, 08-22 생성 가동) + 조건. 평가는 bench_student_sweep 동일 프로토콜
- **구현 완료 (08-22, 스모크 미실행)**: notebooks/geo4d_dmd_[train.py](http://train.py) + geo4d_[fewstep.py](http://fewstep.py). 설계 결정 — ① σ-공간 DMD(grad=(fake−real)/mean\|x0−real\|, Self-Forcing [dmd.py](http://dmd.py) 이식) ② real_score는 teacher+프레임별 CFG 가이더 그대로, generator/fake는 cond-only ③ DMD2 backward simulation(k∈{0,1,2}, 마지막 예측만 grad) ④ student σ=[700,70.5,2.3] 3스텝, **추론은 재노이징(RenoiseSampler)** — Euler로 평가하면 안 됨 ⑤ 컨디션 전부 사전계산 후 conditioner·VAE GPU 제거, 피크 예상 ~28GB(부족 시 --paged_optim) ⑥ 붕괴 감지용 DIAG: 동일 시드 full-step 생성의 std(x0)/std(z_teacher) 비율(1.0=teacher, ↓=안개)
- **아키텍처 확인**: Geo4D denoise 1회 = UNet 4회(뷰1 포인트맵→hd→뷰2 포인트맵이 spatial_context로 받음, 색은 뷰별 독립). 뷰 간 정보는 뷰1→뷰2 포인트맵 경로 하나뿐 → 6-4 기하 loss 설계의 기준점
**6-4 (=6b geometry-aware DMD)**: 포인트맵 consistency loss(두 뷰 예측 점군의 정합)를 DMD 목적함수에 통합 — **본 연구의 핵심 방법론 기여**
**순서의 이유**: 6a 없이는 6b의 개선을 증명할 대조군이 없다 — **"6a에서 기하 붕괴 → 6b에서 회복"이 논문의 핵심 결과표**가 된다
### Step 7 — 도메인 목표 검증 + 고정비 공격 (예정)
- **Step 7-② 결과 (08-23, bench_speed_**[**variants.py**](http://variants.py)**, student 3스텝, uc 생략, 5샘플 동일 시드)**:
<table header-row="true">
<tr>
<td>변형</td>
<td>총</td>
<td>UNet</td>
<td>cond</td>
<td>VAE 디코딩</td>
<td>PSNR L/R</td>
<td>AbsRel L/R</td>
</tr>
<tr>
<td>A fp32 기준</td>
<td>2.38초</td>
<td>1.18</td>
<td>0.30</td>
<td>0.90</td>
<td>19.34/19.98</td>
<td>0.1068/0.0993</td>
</tr>
<tr>
<td>B 디코더 bf16</td>
<td>2.16</td>
<td>1.18</td>
<td>0.30</td>
<td>0.68</td>
<td>19.37/19.97</td>
<td>0.1078/0.0991</td>
</tr>
<tr>
<td>C +UNet bf16</td>
<td>1.64</td>
<td>0.66</td>
<td>0.30</td>
<td>0.68</td>
<td>19.37/19.96</td>
<td>0.1115/0.0963</td>
</tr>
<tr>
<td>D +conditioner bf16</td>
<td>**1.64**</td>
<td>0.66</td>
<td>0.30</td>
<td>0.68</td>
<td>19.46/19.95</td>
<td>0.1063/0.0987</td>
</tr>
<tr>
<td>E D+compile(디코더)</td>
<td>1.29</td>
<td>0.66</td>
<td>0.30</td>
<td>0.33</td>
<td>19.39/19.91</td>
<td>0.1053/0.0981</td>
</tr>
<tr>
<td>**F E+compile(UNet)**</td>
<td>**1.18초**</td>
<td>0.55</td>
<td>0.30</td>
<td>0.33</td>
<td>19.41/19.93</td>
<td>0.1120/0.0944</td>
</tr>
</table>
→ 학습 없이 2.38→1.18초(2배), teacher 21.8초 대비 **18.5배**. 품질 변화는 PSNR +0.03~+0.12dB, AbsRel ±0.005 이내로 시드 노이즈 수준(5샘플 — 20샘플 재확인 필요). compile 워밍업 19+41초는 1회성. UNet이 fp32로 돌고 있었던 것(학습만 bf16)이 가장 큰 낭비였음.
- 이전 메모: 3스텝 순수 추론 2.81초 = UNet 1.19 + conditioner 0.73 + VAE 디코딩 0.88 → uc 제거로 2.44초
- 휴머노이드 0.3–0.5초는 UNet 1회 0.40초 + 고정비라 1스텝에서도 불가 → 양자화·해상도·프레임 수 조정 필요, 동적 0.1초는 한계 절에 정량 명시
### Step 8 — 기능 평가 + 논문화 (예정)
- FoundationPose 궤적 추출 오차, 가능하면 정책 성공률 — 최종 심판 지표지만 셋업 비용이 커서 핵심 결과(6-4) 확보 후 투자
