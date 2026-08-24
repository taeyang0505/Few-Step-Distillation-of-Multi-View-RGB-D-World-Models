# 정량 결과 색인

| 파일 | 내용 |
|---|---|
| `precise_6a_main20.txt` / `precise_6a_main20_fast.txt` / `precise_6a_main20_compile.txt` | 메인 표 (20샘플; fp32 / bf16 / bf16+UNet compile) + paired Wilcoxon, 샘플별 AbsRel |
| `precise_6a_renoise.txt`, `precise_6a.txt` | 10샘플 정밀 분석 (T3r 대조군, 발견 9) |
| `precise_6a_6d_s1600.txt`, `precise_6a_6d_s2000.txt` | 6-4(d) 자기 앵커 loss student (20샘플, bf16, S3/S3b + paired Wilcoxon) |
| `precise_6a_6d_s2400.txt`, `_s3200`, `_s4000` | 6-4(d) 4000스텝 연장의 체크포인트별 평가. step 3200이 전체 지표 최적(승격 후 철회, 로그 9.1~9.2절) |
| `precise_6a_6e_s3200.txt` | 6e(`--anchor_exclude_robot`) step 3200 평가. 과선명 +16%, 실패 판정 |
| `precise_6a_long_s1600.txt`, `_s2400`, `_s3200`, `_s4000` | 6-5 장기 학습 4000스텝의 체크포인트별 평가 (20샘플, bf16, S3b) |
| `precise_6a_steps45.txt` | 6-6 추론 스텝 수 4/5 (반복 σ 스케줄, 20샘플) |
| `precise_6a_avgfinal.txt` | 6-6 avg-final A4b/A6b (마지막 스텝 x0 평균, 20샘플) |
| `precise_6a_hybrid.txt` | 6-6 하이브리드 H3b/H4b (마지막 σ 스텝만 teacher, 20샘플) |
| `precise_6a_final60.txt` | 최종 60샘플(뷰 120개) 표: S3b/S4b/S5b/H3b/H4b vs teacher, 90% CI. 다양성은 20샘플×시드 4 |
| `precise_6a_6a_seed1.txt`, `precise_6a_6a_seed2.txt` | 6-7 학습 seed 편차: 6a 레시피 시드 1·2 (step 1600, S3b) |
| `policy_proxy.txt` | 정책 대리 지표: 그리퍼·사과·배경 영역 AbsRel/PSNR, 3D 중심 오차, 프레임별 값, 90% CI (6a step 1600 S3b / T3r / T25) |
| `policy_proxy_6d3200.txt`, `policy_proxy_6e3200.txt` | 대리 지표: 6d step 3200(그리퍼 +3.1 cm, 승격 철회 근거), 6e step 3200(+4.7 cm) |
| `policy_proxy_steps45.txt`, `policy_proxy_avgfinal.txt`, `policy_proxy_hybrid.txt` | 대리 지표: 추론 스텝 4/5, avg-final, 하이브리드 구성 |
| `policy_proxy_6aseed1.txt`, `policy_proxy_6aseed2.txt` | 대리 지표: seed 1·2 student (그리퍼 +2.4/+2.7 cm) |
| `policy_proxy_final60.txt` | 대리 지표: 최종 60샘플 (S3b 그리퍼 +0.66 cm — 20샘플의 +2.0 cm은 재현 안 됨) |
| `speed_variants.txt` | Step 7 가속 변형 A~F 시간·품질 |
| `eval_6x_maskfix.txt` | 오른쪽 GT 가짜 픽셀 제외 후 재평가 (발견 10) |
| `eval_6x_6a_anchor.txt`, `eval_6x_affine.txt`, `eval_6x_cprime.txt` | 앵커 보정 a/b/c 비교 |
| `eval_6x_6b3_s1600.txt`, `eval_6x_6a_s1600_v2.txt` | 6b3(GT 감독 loss) vs 6a |
| `scale_diag_6a.txt`, `cond_calib_6a*.txt`, `frame_diag.txt` | 깊이 편향 진단 (오라클 affine / 조건 앵커 / 프레임별) |
| `full_sweep_results.txt`, `step_sweep_results.txt`, `crossview_sweep_results.txt`, `blur_test_results.txt` | Step 1 학습 없는 스텝 축소 스윕·블러 3중 검증 |
| `student_sweep_results*.txt` | 6-2/6-3 student 스텝 스윕 (구 마스크, 참고용) |
| `*_raw.json` | 샘플별 원시 수치 |
| `logs/` | 학습(dmd_6a, train_6b*, train_6d, train_6d_long, train_6e, train_long)·생성(ode_gen_1000)·평가(policy_proxy, eval_6e 등) 로그 |

주의: `eval_6x_6a_anchor/affine/cprime`, `student_sweep_results*`, `precise_6a.txt`, `precise_6a_renoise.txt`의 오른쪽 뷰 AbsRel은 발견 10(가짜 픽셀) 이전 마스크 기준이라 오염되어 있음. 확정 수치는 `_maskfix` / `_main20*` 사용.
