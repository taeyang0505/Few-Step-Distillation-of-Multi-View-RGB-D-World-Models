# 정량 결과 색인

| 파일 | 내용 |
|---|---|
| `precise_6a_main20.txt` / `precise_6a_main20_fast.txt` / `precise_6a_main20_compile.txt` | 메인 표 (20샘플; fp32 / bf16 / bf16+UNet compile) + paired Wilcoxon, 샘플별 AbsRel |
| `precise_6a_renoise.txt`, `precise_6a.txt` | 10샘플 정밀 분석 (T3r 대조군, 발견 9) |
| `speed_variants.txt` | Step 7 가속 변형 A~F 시간·품질 |
| `eval_6x_maskfix.txt` | 오른쪽 GT 가짜 픽셀 제외 후 재평가 (발견 10) |
| `eval_6x_6a_anchor.txt`, `eval_6x_affine.txt`, `eval_6x_cprime.txt` | 앵커 보정 a/b/c 비교 |
| `eval_6x_6b3_s1600.txt`, `eval_6x_6a_s1600_v2.txt` | 6b3(GT 감독 loss) vs 6a |
| `scale_diag_6a.txt`, `cond_calib_6a*.txt`, `frame_diag.txt` | 깊이 편향 진단 (오라클 affine / 조건 앵커 / 프레임별) |
| `full_sweep_results.txt`, `step_sweep_results.txt`, `crossview_sweep_results.txt`, `blur_test_results.txt` | Step 1 학습 없는 스텝 축소 스윕·블러 3중 검증 |
| `student_sweep_results*.txt` | 6-2/6-3 student 스텝 스윕 (구 마스크, 참고용) |
| `*_raw.json` | 샘플별 원시 수치 |
| `logs/` | 학습(dmd_6a, train_6b*)·생성(ode_gen_1000)·평가 로그 |

주의: `eval_6x_6a_anchor/affine/cprime`, `student_sweep_results*`, `precise_6a.txt`, `precise_6a_renoise.txt`의 오른쪽 뷰 AbsRel은 발견 10(가짜 픽셀) 이전 마스크 기준이라 오염되어 있음. 확정 수치는 `_maskfix` / `_main20*` 사용.
