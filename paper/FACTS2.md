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
