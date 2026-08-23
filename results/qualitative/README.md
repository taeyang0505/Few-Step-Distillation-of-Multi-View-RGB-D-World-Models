# 정성 결과 색인

| 폴더 | 내용 | 행 순서 |
|---|---|---|
| `qualitative/` | Step 1 학습 없는 스텝 축소 (Euler) | GT / 25 / 8 / 1스텝; `diversity_left.png`는 1스텝 시드 4개; `compare_*.gif` 애니메이션 |
| `student_qual/` | 6-2 ODE 회귀 초기화 v1 | GT / teacher25 / student4 / student1 (안개 붕괴) |
| `dmd6a_qual/` | 6-3 DMD student step1600 (왼쪽 뷰) | GT / teacher25 / student3 / student1; `diversity_student1.png` |
| `dmd6a_qual_lr/` | 같은 student, 왼쪽+오른쪽 뷰 + 상대 깊이 오차맵(`err_*.png`, 0~50%) | teacher25 / student3 / student1 |
| `t3r_qual/` | 학습 없는 teacher 재노이징 3스텝 | GT / teacher25 / T3r3 / T3r1; `diversity_T3r1.png` |
| `videos/` | Step 3·4 재현: Self-Forcing 4스텝, Causal-Forcing stage2a_ode / 2b_cd / 3_final (같은 프롬프트 3개) | |

깊이 그리드는 viridis(밝을수록 멂), 오차맵은 magma(밝을수록 오차 큼). 자세한 해석은 루트 README §3과 `docs/research_log.md`.
