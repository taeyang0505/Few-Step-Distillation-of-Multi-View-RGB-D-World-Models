# FACTS ADDENDUM (2026-08-23 night – 08-24). Same rules as FACTS.md: use only these numbers.

## N1. Self-anchor training loss (run "6d") — PARTIAL RESULT (geometry improved, sharpness not converged)
- Implementation: `geo4d_dmd_train.py --anchor_weight 1`. For predicted frame 0 of both views, log-L1 between decoded depth and the
  conditioning-frame depth (left: conditioning pointmap z; right: conditioning pointmap transformed by inv(E_L)E_R). No ground truth.
  Weight balanced at the x0-gradient level like the cross-view loss (lambda = beta * |g_DMD| / |g_anchor|, beta = 1). Same recipe as 6a
  otherwise (v2 init, 2000 steps, 1 h, peak 27.3 GB). Caveat: predicted frame 0 is 5 dataset steps after the conditioning frame, so the
  ~2-3% of robot pixels that move are also pulled toward the conditioning depth; a `--anchor_exclude_robot` flag exists but was NOT used.
- Training diagnostic (std ratio): 0.79 (0) -> 0.889 (800) -> 0.951 (1300) -> 0.980 (1600) -> 0.998 (1900). Slower rise than 6a
  (6a reached 0.99 at step 1000). No overshoot by 2000. Anchor loss fell 0.41 -> 0.08; anchor scales s_L 0.987, s_R 0.966 at the end.
- Eval (20 samples, paired vs teacher, same protocol as main table):
| ckpt | raw S3 AbsRel (L/R) | S3b AbsRel (L/R) | LPIPS | sharpness | PSNR | diversity |
|---|---|---|---|---|---|---|
| teacher | 0.066 (0.067/0.066) | — | 0.118 | 0.0134 | 20.62 | 0.0227 |
| 6a step 1600 (main) | 0.175 | 0.082 (0.076/0.088) | 0.136 | 0.0136 | 20.43 | 0.0224 |
| 6d step 1600 | 0.118 (0.099/0.136) | 0.090 (0.086/0.093) | 0.162 | 0.0096 (-28%) | 20.25 | 0.0250 |
| 6d step 2000 | 0.102 (0.091/0.113) | 0.081 (0.081/0.082) | 0.148 | 0.0110 (-18%) | 20.41 | 0.0226 |
- Paired 6d-2000 S3b vs teacher (n=40): PSNR -0.21 (p=0.034), AbsRel +0.015 (p<0.001), LPIPS +0.030 (p<0.001), sharpness -0.0024
  (p<0.001), CV +0.003 (p=0.73), diversity -0.0001 (p=1.00).
- Reading: the loss fixes the raw scale (0.175 -> 0.102 without any inference anchor) and closes the left/right gap after anchoring
  (0.081/0.082 vs 0.076/0.088), but sharpness and LPIPS lag; both improve from 1600 to 2000, so the run is not converged.
  Extension to 4000 steps (resume from 2000) is launched on 08-24 ~01:00, results pending — NOT DONE.

## N2. Longer DMD training without the anchor loss (run "train_long", 6a recipe, 4000 steps) — DONE
- Same recipe as 6a, 4000 steps (1.8 s/step, peak 24.1 GB). First attempt crashed at step 850 with CUDA OOM because another user's
  process held 8.87 GB on the shared GPU (log: "Process 416547 has 8.87 GiB memory in use"); a `--resume` flag and an OOM-retry launcher
  were added and the run was restarted; the second run completed without interruption.
- std ratio: 1.126 at step 3000 (overshoot), 1.072 at 3900, 1.068 at 4000 (came back down).
- Eval (20 samples, S3b = per-view anchor, bf16; paired vs teacher n=40):
| step | AbsRel (L/R) | dAbsRel (p) | LPIPS | dLPIPS (p) | sharpness | dsharp (p) | PSNR | dPSNR (p) | diversity | CV |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher | 0.066 (0.067/0.066) | — | 0.118 | — | 0.0134 | — | 20.62 | — | 0.0227 | 0.169 |
| 1600 | 0.082 (0.076/0.088) | +0.016 (<0.001) | 0.137 | +0.019 (<0.001) | 0.0132 | -0.0002 (0.007) | 20.45 | -0.17 (0.053) | 0.0228 | 0.133 |
| 2400 | 0.084 (0.067/0.100) | +0.017 (<0.001) | 0.142 | +0.024 (<0.001) | 0.0167 (+25%) | +0.0033 (<0.001) | 19.90 | -0.72 (<0.001) | 0.0242 | 0.136 |
| 3200 | 0.073 (0.065/0.081) | +0.006 (0.015) | 0.138 | +0.020 (<0.001) | 0.0157 (+17%) | +0.0023 (<0.001) | 20.01 | -0.60 (<0.001) | 0.0230 | 0.131 |
| 4000 | 0.078 (0.077/0.079) | +0.012 (<0.001) | 0.132 | +0.014 (<0.001) | 0.0137 | +0.0003 (0.053) | 20.20 | -0.42 (<0.001) | 0.0214 | 0.174 |
- Reading: step 1600 of this run reproduces the 6a step-1600 numbers (same recipe, same result). Longer training is not monotone:
  2400 and 3200 are over-sharpened (+25%/+17% vs teacher) with lower PSNR; 4000 returns to teacher-level sharpness, improves LPIPS
  (+0.014) and the right view (0.079), but PSNR is 0.42 dB lower and diversity 6% lower (n.s.). No single checkpoint passes all criteria.
  The std-ratio diagnostic tracks this: the overshoot above 1.1 coincides with the over-sharpened checkpoints.

## N3. Policy proxy (simulator-free), run on 6a step 1600 S3b, T3r, T25; 20 samples (40 view-samples; apple present in 37)
- Regions from GT label maps: gripper = ids 29-35 (2.35% of pixels), apple = id 44 (0.23%), background = rest. AbsRel/PSNR on 3-px
  dilated masks; 3D centroid error = ||mean pointmap inside GT mask (pred) - mean (GT)|| in cm. Frames 1..10.
| metric | teacher | T3r | S3b | S3b - teacher [90% CI] | Wilcoxon p |
|---|---|---|---|---|---|
| gripper AbsRel | 0.206 | 0.203 | 0.225 | +0.019 [-0.004, +0.042] | 0.24 |
| gripper centroid error (cm) | 12.3 | 12.8 | 14.3 | +2.0 [+0.3, +3.7] | 0.11 |
| gripper centroid, last frame (cm) | 12.4 | 13.6 | 15.4 | +3.0 [+0.2, +5.8] | 0.064 |
| apple AbsRel | 0.127 | 0.124 | 0.115 | -0.012 [-0.025, +0.001] | 0.27 |
| apple centroid error (cm) | 11.4 | 11.3 | 12.9 | +1.6 [-0.1, +3.2] | 0.13 |
| background AbsRel | 0.063 | 0.063 | 0.078 | +0.015 [+0.010, +0.021] | <0.001 |
- T3r region PSNR is significantly higher than the teacher's (gripper +0.80 dB, p<1e-11; apple +0.63, p<1e-4): blur helps PSNR (Finding 1).
- Per-frame gripper centroid error (cm), frames 1..10: teacher 11.0 12.6 13.7 13.9 13.2 12.8 11.8 11.2 10.8 12.4;
  S3b 11.9 11.7 16.6 17.3 15.4 14.9 14.6 13.2 12.5 15.4.
- Caveats: (1) the GT mask is applied to the prediction, so if the predicted arm is displaced the mask covers background pixels and the
  centroid jumps; the teacher's own 12 cm is therefore inflated by mask mismatch and by the multimodal future, and only differences are
  interpretable; (2) minimal detectable effects at n=40: gripper centroid ~3 cm, apple ~2.9 cm, gripper AbsRel 0.039.
- Reading: the whole-image AbsRel gap (+0.015) comes from the static background; on the gripper and apple regions the student is not
  statistically distinguishable from the teacher by Wilcoxon, but the gripper centroid error is 2.0 cm higher with a 90% CI that excludes 0,
  so it should be reported as a likely small degradation, not as equality.

## N4. Equivalence-margin framework (proposed for the paper; margins are the authors' choices grounded in the cited literature)
- Report TOST-style: a metric is "preserved" if the 90% CI of the paired difference lies within [-delta, +delta].
- Margins and grounds: PSNR 0.5 dB (0.5 dB significance convention, US patent 11216692B2; ~1 dB per JND); AbsRel strict 0.005 (one
  SOTA generation on KITTI/NYU, Depth Anything V2 Table 4) / practical 0.010 (half of RealSense D435 +/-2% spec, ~1 cm at 1 m; closed-loop
  grasping tolerates ~10 mm depth noise, arXiv:2607.06186; open-loop depth-anchored policy drops 42%->30% at 1 cm, NoTVLA Table 15);
  LPIPS perceptual 0.05 (|dLPIPS|<0.05 gives ~52% human agreement, Hou et al. ECCV 2022) / strict 0.01; sharpness +/-5% (half of the
  -11% of the training-free 3-step teacher); diversity +/-10%; CV-Chamfer one-sided (not worse); gripper centroid +1 cm strict (pose
  tracker noise, FoundationPose 3-10 mm, RGBTrack Table I) / +2 cm practical (Tremblay et al. 2020: >2 cm pose error fails a 4 cm-stroke grasp).
- Minimal detectable effects with 40 view-samples: PSNR 0.26 dB, AbsRel 0.010, LPIPS 0.004, sharpness 0.0004, diversity 0.0042 (n=6),
  CV 0.034 (n=20). Detecting AbsRel 0.005 needs ~115 view-samples (~60 samples); diversity needs 20 samples x 4 seeds.
- Status of the 6a-1600 student under these margins: PSNR pass (-0.19, CI [-0.34,-0.04]); AbsRel fail (+0.0155, CI [+0.010,+0.021]);
  LPIPS perceptual pass / strict fail (+0.018, CI [+0.015,+0.020]); sharpness pass (+1.6%); diversity provisional pass (CI wider than
  margin); CV pass; gripper centroid practical borderline (+2.0, CI [+0.3,+3.7]).

## N5. Self-anchor loss extended to 4000 steps (run 6d, resumed from 2000) — SUCCESS at step 3200 (added 08-24 04:50)
- Training: resumed at 2000 with --resume, one OOM crash (another user's job) auto-recovered by the retry launcher. std ratio 1.075 at 4000;
  anchor loss 0.07-0.09 at the end, anchor scales s_L/s_R 0.95-1.03.
- Eval (20 samples, paired vs teacher n=40):
| ckpt | raw S3 AbsRel (L/R) | S3b AbsRel (L/R) | dAbsRel [90% CI] | LPIPS (d) | sharpness (d, p) | PSNR (d, p) | diversity |
|---|---|---|---|---|---|---|---|
| teacher | 0.066 (.067/.066) | — | — | 0.118 | 0.0134 | 20.62 | 0.0227 |
| 6d 2400 | 0.095 (.086/.104) | 0.080 (.079/.081) | +0.013 | 0.143 (+0.025) | 0.0117 (-0.0017, <0.001) | 20.45 (-0.17, 0.048) | 0.0219 |
| **6d 3200** | 0.098 (.093/.103) | **0.078 (.076/.080)** | **+0.011 [+0.007, +0.016]** | 0.135 (+0.017 [+0.015,+0.019]) | **0.0135 (+0.0001, p=0.94)** | 20.45 (-0.16 [-0.31,-0.01], p=0.07) | 0.0213 (-0.0014, n.s., n=6) |
| 6d 4000 | 0.093 (.089/.098) | 0.079 (.076/.082) | +0.013 | 0.135 (+0.017) | 0.0151 (+0.0017, <0.001, over) | 20.28 (-0.34, 0.001) | 0.0218 |
- CV-Chamfer at 3200: +0.0115 vs teacher (p=0.29, not worse; note 6a-1600 was better than teacher on CV).
- Against the pre-set success criteria: sharpness >= 0.0127 PASS (0.0135, p=0.94); raw AbsRel <= 0.10 PASS (0.098); LPIPS <= 0.136 PASS
  (0.135); anchored AbsRel <= 0.076 MISSED by 0.002 (0.078). Three of four met.
- Reading: with 3200 steps the self-anchor loss delivers what neither 6a (raw 0.175) nor plain longer training (over-sharpening) did:
  raw scale mostly fixed without any inference anchor (0.098), left/right gap closed (0.076/0.080), sharpness exactly teacher level,
  PSNR within margin, AbsRel gap reduced from +0.0155 to +0.0113 (90% CI upper 0.0157, still above the 0.010 practical margin, so
  equivalence is not yet claimable at n=40). At 4000 the over-sharpening pattern (Finding 14) begins (+0.0017, p<0.001).
- 6d step 3200 is the best checkpoint overall and a candidate to replace 6a step 1600 as the main student. Timing is unchanged
  (same architecture, 1.64 s bf16). NOT yet promoted in the paper — pending author decision.

## N6. PROMOTION DECISION (author-approved, 08-24): 6d step 3200 is the MAIN student
- The paper's main student is now "DMD + self-anchored training loss (beta=1, 3200 steps) + per-view inference anchor + bf16".
  Inference pipeline and timing unchanged: 1.64 s (13.1x), 1.18 s with compile — same architecture and sampler as before.
- Main numbers (20 samples, paired vs teacher n=40):
  PSNR 20.45 (d -0.16 [90% CI -0.31, -0.01], p=0.070); AbsRel 0.078, L 0.076 / R 0.080 (d +0.0113 [+0.0069, +0.0157], p<0.001);
  LPIPS 0.135 (d +0.0172 [+0.0150, +0.0194], p<0.001); sharpness 0.0135 (d +0.0001, p=0.94); diversity 0.0213 (d -0.0014, p=0.44, n=6);
  CV-Chamfer 0.181 (d +0.0115, p=0.29, within the 0.034 MDE; note the 6a student was better than the teacher on CV, -0.032).
  Raw AbsRel without the inference anchor: 0.098 (vs 0.175 for the 6a student); the inference anchor then takes 0.098 -> 0.078.
- Margin status of the new main student: PSNR pass; sharpness pass; diversity provisional pass; CV pass (one-sided, n.s.);
  LPIPS perceptual pass / strict fail (+0.017); AbsRel practical margin now NEARLY met (CI [+0.007, +0.016] vs margin 0.010 — center above
  by 0.001, upper bound 0.016; still not claimable as equivalent at n=40).
- The 6a student (no training anchor, step 1600) remains in the paper as the ablation "without the self-anchored training loss":
  AbsRel 0.082 (0.076/0.088), raw 0.175, LPIPS 0.137, sharpness 0.0136, PSNR 20.43, diversity 0.0224, CV 0.137.
- The policy proxy (N3) numbers were measured on the 6a student and have NOT been rerun on 6d-3200 — say so where cited.
- Checkpoint file: ~/Geo4D/dmd_6d/dmd_gen_step3200.pt. Training cost of the main student: 2000 + 2000 resumed steps, ~2 h total.

## N7. Policy proxy rerun on the 6d-3200 student (08-24 05:30) — REVERSAL: moving regions got WORSE
| metric (paired vs teacher, n=40/37) | 6a-1600 student | 6d-3200 student |
|---|---|---|
| background AbsRel | +0.015 (p<0.001) | +0.010 (p<0.001) — improved |
| gripper AbsRel | +0.019 (p=0.24, n.s.) | +0.032 (p=0.006) — worse |
| gripper centroid error | +2.0 cm (p=0.11) | +3.1 cm (p=0.009) — worse |
| apple AbsRel | -0.012 (n.s.) | +0.006 (p=0.27) |
| apple centroid error | +1.6 cm (p=0.13) | +2.7 cm (p<0.001) — worse |
- Per-frame gripper centroid (6d-3200): f1 10.9 (teacher 11.0), f3-f7 16.6-17.5 vs teacher 13.7-11.8.
- Interpretation: the anchor loss pulls predicted frame-0 depth toward the conditioning frame INCLUDING the ~2-3% moving robot pixels;
  the static background (97% of pixels) improves, which whole-image AbsRel rewards; the whole-image gain partly comes at the cost of the
  policy-relevant moving regions. Source files: policy_proxy_6d3200.txt / _raw.json.

## N8. Run 6e (anchor loss with --anchor_exclude_robot, 4000 steps attempted) — FAILED (08-24 ~09:00)
- Training crashed at step 3200 when the disk filled (checkpoint copy truncated; dmd_gen.pt at step 3200 was intact and used).
  Six obsolete checkpoints (18 GB) were deleted with user approval to recover space; evaluation rerun afterwards.
- 6e step 3200 (20 samples): whole-image AbsRel 0.082 (+0.0158), LPIPS 0.136 (+0.0180), sharpness 0.0155 (+16% over-sharpened, p<0.001),
  PSNR 20.18 (-0.44, p<0.001), diversity n.s. std ratio 1.081 at 3200 and rising -> the checkpoint sits in the over-sharpening region,
  unlike 6d-3200, so the two runs are not at the same convergence point.
- Policy proxy (paired vs teacher): gripper AbsRel +0.039 (p<0.001), gripper centroid +4.7 cm (p<0.001), apple centroid +4.4 cm
  (p<0.001), background +0.0148 (p<0.001; the background gain of 6d is gone too).
- Reading: excluding the robot pixels did not repair the moving-region degradation and lost the background gain; across 6d and 6e the
  anchor-loss variants consistently hurt the policy-relevant moving regions even when whole-image metrics improve. Run-to-run variance
  caveat: 6d and 6e are single runs each.
- Standing insight for the paper: a well-motivated geometry loss can improve whole-image depth metrics while degrading the regions a
  policy reads; whole-image metrics alone are insufficient for checkpoint or method selection (the policy-level analogue of Finding 1).
- Decision pending (author): revert the main student to 6a-1600 (recommended; N6 promotion edits exist only locally), keep 6d-3200, or
  evaluate 6e-2400 first.

## N9. Inference-time step count (no training): 4 and 5 re-noising steps with repeated trained sigmas (08-24 14:20)
- Schedules repeat trained sigmas only: 4 steps [700, 70.5, 2.3, 2.3]; 5 steps [700, 70.5, 70.5, 2.3, 2.3]. Student = 6a-1600 + per-view anchor, bf16.
- 20 samples, paired vs teacher, 90% CI:
| metric | 3 steps | 4 steps | 5 steps |
|---|---|---|---|
| pure inference (measured 08-25) | 1.64 s | 1.86 s | 2.08 s |
| AbsRel | 0.082, +0.0155 [+0.0098,+0.0211] | 0.075, +0.0085 [+0.0023,+0.0146] | 0.072, +0.0058 [-0.0001,+0.0117] |
| LPIPS | +0.0177 | +0.0147 [+0.0129,+0.0165] | +0.0147 [+0.0131,+0.0163] |
| sharpness | +0.0002 | +0.0002 | +0.0003 |
| diversity | -1% [-12,+9] | +6% [-6,+18] (n=6) | +17% [+2,+32] (n=6) — OVER-DISPERSED, exceeds the +/-10% margin |
| PSNR | -0.19 [-0.34,-0.04] | -0.38 [-0.52,-0.24] (fails 0.5 dB TOST by 0.02) | -0.35 [-0.48,-0.23] (passes narrowly) |
| CV | -0.032 | -0.027 | -0.021 |
| gripper centroid (proxy) | +2.0 [+0.3,+3.7] cm | +1.7 [-0.2,+3.6] | +1.3 [-0.7,+3.3] |
| gripper AbsRel (proxy) | +0.019 | +0.00 [-0.02,+0.03] | +0.00 |
| background AbsRel (proxy) | +0.015 | +0.007 | +0.008 |
- Reading: extra low-sigma refinement (no training) closes most of the depth gap (+0.0155 -> +0.0058) and the moving-region gaps
  (gripper AbsRel to +0.00, centroid CI includes 0), supporting the "insufficient low-sigma refinement" diagnosis. The cost is a new
  trade-off axis: each re-noise injects fresh noise, so repeated sigmas over-disperse the sampler (diversity +17% at 5 steps, above the
  margin) and PSNR drops accordingly (consistent with E||y_hat - y*||^2 = Var(q) + Var(p) + bias^2). 4 steps sits in the margin for
  diversity (+6%) but fails the PSNR TOST by 0.02 dB at n=40. Decision on the default step count deferred to the seed-variance runs and
  the 60-sample rerun. Candidate follow-up: repeat the high sigma instead ([700, 700, 70.5, 2.3]) to seek geometry gains without
  over-dispersion; test alongside the hybrid last-step experiment. Source: precise_6a_steps45.txt, policy_proxy_steps45.txt.

## N10. Training-seed variance of the 6a recipe (three seeds, step 1600, S3b; 08-24 17:45)
| seed | AbsRel (L/R) | LPIPS | sharpness | PSNR | diversity | gripper centroid vs teacher |
|---|---|---|---|---|---|---|
| 0 (main) | 0.0820 (.076/.088) | 0.136 | 0.0136 | 20.43 | 0.0224 | +2.0 cm (p=0.11) |
| 1 | 0.0849 (.080/.090) | 0.138 | 0.0132 | 20.40 | 0.0232 | +2.4 cm (p=0.03) |
| 2 | 0.0852 (.079/.091) | 0.138 | 0.0131 | 20.41 | 0.0230 | +2.7 cm (p=0.03) |
- Range across seeds: AbsRel 0.003 (sd ~0.002), LPIPS 0.002, sharpness 0.0005, PSNR 0.03 dB, gripper 0.7 cm.
  A same-seed rerun (train_long step 1600) reproduced seed 0 almost exactly (AbsRel 0.0820) -> the spread comes from the seed.
- Consequences: (1) single-run training-side differences of <~0.004 AbsRel or <~1 cm gripper are within ~2x seed noise —
  the 6d-3200 anchored-AbsRel gain (0.004) is not claimable from single runs; the 6d raw-scale fix (0.175 -> 0.098) and the 6e gripper
  degradation (+4.7 cm) are far outside the noise and stand. (2) Inference-side comparisons (S4/S5/A4/H3) reuse one checkpoint, so
  training noise cancels; they are the statistically clean improvement path. (3) The original main student (seed 0) happens to be the
  best of the three seeds — note as a selection-bias caveat; the recipe's gripper deficit is real (+2.0 to +2.7 cm across seeds).

## N11. Avg-final (A4/A6) and hybrid last-step (H3/H4) results (08-24 19:30) — the dial is inside the last step
- 20 samples, anchor applied, bf16. teacher: AbsRel 0.066, LPIPS 0.118, PSNR 20.62, sharp 0.0134, div 0.0227.
| config | calls | AbsRel (L/R) | LPIPS | PSNR | sharpness | diversity | gripper centroid |
|---|---|---|---|---|---|---|---|
| S3b | 3 | 0.082 (.076/.088) | 0.136 | 20.43 | 0.0136 | 0.0224 | +2.0 cm |
| S4b | 4 | 0.075 (.069/.081) | 0.133 | 20.24 | 0.0136 | 0.0240 | +1.7 |
| A4b (3 sigma + avg2 at final) | 4 | 0.082 (.079/.085) | 0.142 | 20.76 | 0.0131 | 0.0172 (-24%) | +2.7 (p=0.009) |
| A6b (4 sigma + avg2) | 5 | 0.073 (.068/.079) | 0.137 | 20.48 | 0.0134 | 0.0196 (-14%) | +1.4 |
| H3b (student 700/70.5 + teacher 2.3) | 3 | 0.062 (.063/.061) — BETTER than teacher | 0.136 | 21.33 | 0.0120 (-10%) | 0.0131 (-42%) | +1.5 (p=0.03); gripper AbsRel +0.005 n.s.; background BETTER than teacher (-0.005) |
| H4b | 4 | 0.060 (.061/.058) — best ever | 0.130 | 21.09 | 0.0123 (-8%) | 0.0158 (-30%) | +1.5 (p=0.20), gripper AbsRel +0.003 n.s. |
- Mechanism (unified): sigma = 2.3 exceeds the latent scale (~1), so a large share of seed-to-seed diversity is CREATED by sampling at
  the final step. The teacher denoiser is a mean predictor at every sigma; the DMD student was trained to emit samples at sigma 2.3.
  Any mean-ward operation at the final step (teacher swap, x0 averaging) therefore trades diversity/sharpness for AbsRel/PSNR — the
  perception-distortion dial lives inside the sampler's last step and can be set at inference time without retraining.
  (Corrects the earlier assumption that diversity is decided at sigma 700.)
- A4's premise (variance reduction without distribution shift) was disproved for the diversity axis: averaging collapses diversity -24%
  and does not reproduce S4b's AbsRel gain (the S4b gain comes from the extra stochastic refinement cycle, not variance reduction).
- Practical outcome: mode-switchable inference from ONE checkpoint — "diverse mode" S3b (1.64 s, all soft metrics at teacher level) for
  sampling futures; "precise mode" H4b (~1.9 s, AbsRel 0.060 below the teacher, LPIPS 0.130) for metric-depth readout of a chosen future.
  Proposed pipeline: plan with the diverse mode, then one extra teacher step (+~0.2 s) on the selected future for precise geometry.
- Main-table default stays S3b (advantage preservation is the claim); H4b presented as the precise mode. Deployment note: hybrid keeps
  the teacher UNet in memory (+ a few GB).

## N12. FINAL 60-sample table (120 view-samples; diversity 20 samples x 4 seeds; 08-24 22:56) — definitive margins
- Teacher baseline on 60 samples: PSNR 20.34, AbsRel 0.0725, LPIPS 0.1227, sharpness 0.0127, diversity 0.0185, CV 0.1685,
  gripper centroid 17.0 cm. (Harder than the first 20 samples; all comparisons remain paired.)
- Differences vs teacher, 90% CI:
| config | calls/time | AbsRel | LPIPS | PSNR | sharpness | diversity | CV | gripper centroid |
|---|---|---|---|---|---|---|---|---|
| S3b | 3 / 1.64 s | +0.0133 [+0.0097,+0.0168] FAIL 0.010 | +0.0180 | -0.149 [-0.246,-0.053] PASS | +2% PASS | +5% [+1,+8] PASS (provable now) | -19% PASS | +0.66 [-0.62,+1.94] cm PASS 2cm |
| S4b | 4 / 1.86 s | +0.0066 [+0.0029,+0.0102] marginal FAIL (upper 0.0102) | +0.0146 | -0.364 PASS | PASS | +8% [+4,+11] marginal FAIL | PASS | (not measured) |
| S5b | 5 / 2.08 s | +0.0048 [+0.0011,+0.0085] **PASS practical 0.010 with full CI** | +0.0151 | -0.311 [-0.397,-0.225] PASS | PASS | +10% [+6,+14] marginal FAIL | PASS | -0.75 [-2.25,+0.75] PASS |
| H3b | 3 / ~1.64 s | -0.0034 (better) | +0.0175 | +0.688 FAIL (above +0.5) | -10% FAIL | -39% FAIL | +2% n.s. | — |
| H4b | 4 / ~1.9 s | -0.0062 [-0.0084,-0.0040] better than teacher | +0.0109 (best) | +0.469 [+0.377,+0.562] marginal | -8% FAIL | -33% FAIL | +5% [+2,+8] FAIL | +1.00 [+0.12,+1.89] PASS 2cm |
- LPIPS strict (0.01) fails for every config (best H4b +0.0109); perceptual margin (0.05) passes for all.
- CORRECTION of N3/N7-era claims: at n=120 the S3b gripper centroid difference is +0.66 cm [-0.62,+1.94] and S5b is -0.75 cm —
  the 20-sample "+2.0 [+0.3,+3.7], likely degradation" conclusion does not replicate; the first 20 samples were unfavorable.
  Lesson recorded: small-n CIs can exclude zero and still not replicate; region metrics need the larger n.
- Final picture: no single config passes every strict margin; the configurations form an inference-time dial:
  S3b = fastest, all soft metrics pass, depth fails; S5b = depth passes (sensor-level), mild over-dispersion (+10% [+6,+14]), 2.08 s;
  H4b = precise mode, depth below teacher, diversity/sharpness sacrificed. Paper main table should present the dial.

## N13. Region-wise sharpness — the whole-image sharpness metric is background-dominated (08-26, PRELIMINARY)
Measured offline on the saved native-resolution qualitative grid (results/qualitative/dmd6a_qual_lr/rgb_left.png):
ONE sample, left view, Laplacian variance averaged over the 10 predicted frames, two hand-drawn ROIs.
| row | whole image (Table 1, 20 samples) | moving ROI (arm + gripper over the bowl) | static ROI (right gripper) |
|---|---|---|---|
| GT | 0.0197 | 1461 | 1326 |
| teacher 25 steps | 0.0134 (-32% vs GT) | 692 (-53% vs GT) | 779 (-41% vs GT) |
| student 3 steps | 0.0136 (+2% vs teacher) | 413 (**-40% vs teacher**) | 807 (+4% vs teacher) |
| student 1 step | 0.0107 (-20%) | 117 (-83%) | 631 (-19%) |
- Reading: the headline "sharpness preserved" (+2%) is an average dominated by static pixels. Split by region the two
  directions are opposite: the student MATCHES the teacher on static content and loses 40% on the moving arm. This is the
  same structure as the AbsRel finding (whole-image gap came from the background) with the sign reversed, and it means the
  reviewer complaint "the arm looks blurry" is supported by measurement, not just impression.
- Every frame shows the same ordering (student < teacher in the moving ROI at all 10 frames), so it is not a frame artifact,
  but this is ONE sample with hand-drawn boxes. CONCLUSIVE MEASUREMENT STILL REQUIRED: rerun with the GT label masks
  (gripper ids 29-35, plus a dilated moving-object mask) over 20-60 samples, same protocol as the policy proxy, and add
  region-wise LPIPS at the same time. Until then this must not be quoted as a result.
- Consequence for the paper: the sharpness margin of Section 4.1 (+/-5%) was applied to a whole-image metric; a region-wise
  version of the same margin is the honest test, and the main student may fail it. Figure 9 (fig9_zoom_qualitative.png)
  shows the crops with these numbers.

## N14. D1 Task 1 (cereal_box) — the recipe transfers, but the diverse mode degrades far more; the hybrid rescues it (08-26)
Pipeline: 160 ODE pairs (apple had 284), 1200-step regression init, 2000-step DMD, checkpoint step 1600 (std ratio 1.021, the same
criterion used for apple). Evaluation: 20 samples = 40 view-samples, same protocol as apple. GT sharpness 0.01216 (apple 0.02137).
| config | s/gen | PSNR | AbsRel (L/R) | LPIPS | sharpness | CV | diversity |
|---|---|---|---|---|---|---|---|
| T25 teacher | 24.7 | 18.91 | 0.1325 (.1388/.1262) | 0.1610 | 0.00889 | 0.0538 | 0.02683 |
| S3b | 4.4 | 18.51 | 0.1845 | 0.1861 | 0.01184 | 0.0557 | 0.02676 |
| S5b | 4.8 | 18.04 | 0.1600 | 0.1811 | 0.01236 | 0.0488 | 0.02959 |
| H4b | 4.6 | 19.25 | 0.1390 | 0.1781 | 0.00906 | 0.0408 | 0.01804 |
Paired vs T25: S3b AbsRel +0.0519 (p<0.001), LPIPS +0.0251, PSNR -0.39, sharpness +0.0029 (**+33%, 100% of samples**), diversity n.s.
S5b AbsRel +0.0275, PSNR -0.87. **H4b AbsRel +0.0065 (p=0.383, NOT significant), PSNR +0.34, sharpness n.s. (p=0.259), CV better;
only diversity -33%.**
- Comparison with apple (same S3b recipe): AbsRel gap +0.052 here vs +0.013 there; LPIPS +0.025 vs +0.018; sharpness +33% vs +2%.
  The teacher is also much weaker on this task (AbsRel 0.133 vs 0.073), so the task is harder for both.
- Policy proxy (object id 39, 20 samples): gripper centroid 11.04 -> 18.23 cm (+7.19, p=1.3e-11, 95% of samples worse), object centroid
  2.82 -> 8.63 cm (+5.80), gripper AbsRel +0.102, background AbsRel +0.044. Far worse than apple's +0.66 cm at n=120.
- Diagnosis: the student is OVER-SHARPENED on this task (sharpness 0.0118 against a teacher of 0.0089 and a ground truth of 0.0122):
  it hallucinates high-frequency detail while the geometry degrades. The std-ratio checkpoint criterion did NOT catch this - the ratio
  moves smoothly from 0.99 (step 1200) to 1.05 (step 2000) with no signal at 1600 (1.021). This is a limitation of our own selection
  diagnostic and must be reported as such.
- Reading for the paper: the recipe transfers in the sense that DMD still restores diversity (n.s. vs teacher) and gives a 5.6x speed-up,
  but on this harder task the 3-step diverse mode is NOT teacher-equivalent on geometry. The hybrid precise mode IS
  (AbsRel indistinguishable, PSNR better). The honest claim becomes: the dial generalizes, and on harder tasks the precise mode is the
  one that preserves geometry, at the cost of a third of the diversity.
- Only two checkpoints survive (step 1600 and step 2000); intermediates were overwritten. Testing whether an earlier checkpoint avoids
  the over-sharpening needs a 1 h retrain with per-step saving. NOT yet done.
- Source: results/quantitative/precise_6a_cereal_box.txt, policy_proxy_cereal_box.txt.

## N15. Region-wise sharpness and LPIPS, CONFIRMED at 20 samples (08-26) — supersedes the preliminary N13
Protocol: apple, LEFT VIEW ONLY (20 samples), masks = motion mask (GT 3D moved > 2 cm from the conditioning frame; 3.6% of pixels),
GT label gripper ids 29-35 (2.14%) and object id 44 (0.21%). Sharpness = Laplacian variance inside the mask after 1 px erosion;
LPIPS = spatial LPIPS map averaged inside the mask. Paired 90% CI by bootstrap.
RIGHT VIEW EXCLUDED: the first run's motion mask compared the right-view GT (reference frame) with the right conditioning pointmap
(right-camera frame) and flagged 53% of pixels as moving. Fixed in the script (extrinsic transform, same as cond_anchor_scale_right),
but the numbers below are left-view only and the right view has NOT been re-measured.
| region | GT sharpness | teacher (% of GT) | S3b (% of GT) | S5b | H4b |
|---|---|---|---|---|---|
| whole | 0.02171 | 0.01624 (75%) | 0.01622 (75%) | 0.01647 (76%) | 0.01527 (70%) |
| static | 0.02175 | 0.01629 (75%) | 0.01635 (75%) | 0.01656 (76%) | 0.01542 (71%) |
| moving | 0.03759 | 0.01705 (45%) | 0.01193 (**32%**) | 0.01457 (39%) | 0.00906 (**24%**) |
| gripper | 0.03281 | 0.01516 (46%) | 0.01337 (41%) | 0.01399 (43%) | 0.01194 (36%) |
| object | 0.03134 | 0.01715 (55%) | 0.01397 (45%) | 0.01539 (49%) | 0.01251 (40%) |
S3b - teacher, paired: whole sharpness **-0.1% [CI -0.00038, +0.00040] (parity)**; static +0.3% (parity);
**moving -30.0% [-0.00606, -0.00421]**; gripper -11.8% [-0.00302, -0.00044]; object -18.5% [-0.00566, -0.00052].
LPIPS by region (teacher -> S3b): whole 0.1154 -> 0.1356 (+17.6%); static 0.1115 -> 0.1312 (+17.6%); moving 0.3336 -> 0.3868 (+15.9%);
gripper 0.2434 -> 0.2990 (+22.9%); object 0.2132 -> 0.2501 (+17.3%).
- FINDING 1: the paper's "sharpness preserved" is an artifact of a background-dominated metric. The moving region, 3.6% of pixels,
  loses 30% of its sharpness with a CI that excludes zero. The +/-5% sharpness margin of Section 4.1 PASSES on the whole image and
  FAILS by a wide margin on the moving region. This must be corrected in the abstract, Section 4.1, 4.3 and Table 1.
- FINDING 2: the LPIPS gap is uniform across regions in relative terms (16-23%), so it cannot be dismissed as a background-texture
  effect, and it is not specific to the arm either. It is a global texture difference introduced by DMD.
- FINDING 3: the hybrid H4b, our "precise mode", is the WORST configuration on moving-region sharpness (24% of GT against the
  teacher's 45%, i.e. -46%) and on moving/gripper LPIPS, even though its depth is the best. The dial therefore trades appearance in the
  moving region for depth accuracy; S5b is the best compromise there (39% vs teacher 45%, -13%).
- Source: results/quantitative/region_perceptual_apple.txt / _raw.json (right-view rows invalid, see above).

## N16. Step-count curves, T3r region comparison, and best-of-N (08-27, apple, 20 samples unless noted)
### (a) Metric vs step count, ONE run each (teacher curve and student curve measured separately, T25 in both)
teacher weights + re-noising sampler, no training:
| steps | AbsRel | LPIPS | sharpness | diversity |
|---|---|---|---|---|
| 1 | 0.0676 | 0.1455 | 0.01169 | 0.00492 (22%) |
| 2 | 0.0646 | 0.1333 | 0.01194 | 0.01123 (49%) |
| 3 | 0.0655 | 0.1321 | 0.01193 | 0.01195 (53%) |
| 4 | 0.0642 | 0.1288 | 0.01215 | 0.01344 (59%) |
| 5 | 0.0632 | 0.1294 | 0.01212 | 0.01414 (62%) |
| 6 | 0.0632 | 0.1276 | 0.01228 | 0.01474 (65%) |
| 8 | 0.0664 | 0.1264 | 0.01232 | 0.01289 (57%) |
| 25 | 0.0664 | 0.1179 | 0.01340 | 0.02271 (100%) |
**The teacher's depth accuracy is flat from 1 to 25 steps (0.063-0.068). Steps buy diversity and LPIPS, not depth.**
This answers "why did the authors use 25 steps": not for geometry.
DMD student + per-view anchor:
| steps | AbsRel | LPIPS | sharpness | diversity |
|---|---|---|---|---|
| 1 | 0.1161 | 0.1766 | 0.01065 | 0.01167 |
| 2 | 0.0822 | 0.1367 | 0.01344 | 0.02235 |
| 3 | 0.0819 | 0.1357 | 0.01362 | 0.02242 |
| 4 | 0.0749 | 0.1326 | 0.01363 | 0.02398 |
| **5** | **0.0722 (minimum)** | 0.1327 | 0.01372 | 0.02660 |
| 6 | 0.0737 | 0.1312 | 0.01377 | 0.02719 |
| 8 | 0.0763 | 0.1309 | 0.01406 | 0.02598 |
**AbsRel has a clear optimum at 5 steps and worsens at 6 and 8; sharpness passes the teacher at ~8 steps
(over-sharpening) and diversity over-disperses to 120% at 6. Five steps is the balance point.**
### (b) Region-wise, ALL FOUR configs in ONE run (20 samples, left view) — sharpness as % of GT
| region | T25 | T3r (no training) | S3b (DMD) | S5b |
|---|---|---|---|---|
| whole | 75% | 68% | 75% | 75% |
| static | 75% | 69% | 76% | 76% |
| **moving** | **49%** | **25%** | **37%** | **43%** |
| gripper | 46% | 33% | 41% | 43% |
| object | 58% | 40% | 50% | 52% |
LPIPS vs teacher: moving T3r +23.7% / S3b +15.1% / S5b +11.2%; gripper T3r +23.3% / S3b +18.3% / S5b +13.3%;
static T3r **+11.1%** / S3b +15.4% / S5b +13.0%.
**DECISIVE FOR THE DMD JUSTIFICATION: the training-free 3-step teacher loses HALF the teacher's moving-region
sharpness (25% vs 49%) while its whole-image number looks fine (68% vs 75%). DMD restores it to 37% (S3b) and
43% (S5b). DMD trades static-region fidelity (where T3r wins) for moving-region fidelity (where DMD wins) —
i.e. it buys back exactly the region a policy reads.** This is the strongest evidence we have that DMD is not decorative.
### (c) best-of-N (10 samples = 20 view-samples, 8 seeds, AbsRel, oracle selection) — NEGATIVE RESULT
| config | N=1 | N=8 | gain | seed-to-seed std of AbsRel |
|---|---|---|---|---|
| T25 | 0.0724 | 0.0598 | **17.3%** | 0.0087 |
| T3r | 0.0720 | 0.0613 | **14.9%** | 0.0066 |
| S3b | 0.0868 | 0.0784 | **9.7%** | 0.0074 |
| S5b | 0.0788 | 0.0726 | **7.8%** | 0.0072 |
- The DMD student has ~2x the pixel diversity of T3r (0.0224 vs 0.0120) yet gains LESS from drawing 8 samples
  (9.7% vs 14.9%). Its diversity is in appearance, not in depth accuracy: seed-to-seed AbsRel spread is
  comparable (0.0074 vs 0.0066).
- Even with 8 draws and oracle selection, S3b (0.0784) never reaches T3r's single draw (0.0720).
- **CONFOUND, must be resolved before quoting this**: S3b/S5b have the per-view anchor, which rescales each
  sample's depth to the conditioning frame and therefore compresses exactly the seed-to-seed AbsRel spread this
  experiment measures. T3r has no anchor. A fair rerun needs T3r+anchor (or S3b without anchor); ~15 min GPU.
- Caveat 2: this measures diversity of DEPTH ERROR, not diversity of plausible futures. A metric over
  trajectories (e.g. gripper path spread) would test the planning argument more directly.
- Honest reading as of now: (b) supports DMD on appearance in the moving region; (c) does NOT support the claim
  that DMD's diversity converts into better depth, even under oracle selection.
