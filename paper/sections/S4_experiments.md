# 4 Experiments

## 4.1 Setup

**Task and data.** We evaluate on Task 3 of Geo4D (Liu et al., 2025), PlaceAppleFromBowlIntoBin, a bimanual task in which the left arm lifts an apple out of a bowl and hands it to the right arm, which places it in a bin. We use the simulation data released with Geo4D, recorded in the Drake-based LBM environment (Toyota Research Institute, 2025): 4 inference episodes (218 frames) from 4 cameras (scene_13 to scene_16), stride 5, one conditioning frame and 10 predicted frames at 256 × 320, two cameras drawn at random per sample. The teacher is the public Geo4D checkpoint throughout.

**Protocol.** Final numbers use 20 samples, i.e. 40 view-samples, with fixed data seed 1234 and generation seed 0. Differences are tested with a paired Wilcoxon signed-rank test (Wilcoxon, 1945) over the 40 view-samples. Seed diversity is measured on the first 3 samples, each generated with seeds 0 to 3. Early sweeps used 10 samples and are marked as such.

**Metrics.** PSNR is computed on RGB in [0, 1]. AbsRel is the absolute relative error of predicted depth $z$ in meters under a valid-pixel mask. LPIPS uses the AlexNet backbone (Zhang et al., 2018). Sharpness is the variance of the Laplacian of the gray image, read relative to the teacher. Diversity is the per-pixel standard deviation across the 4 seeds, averaged over pixels. CV-Chamfer is the symmetric Chamfer distance between the point clouds predicted for the two views, both in the reference camera frame. Lower is better, but the metric is scale dependent and rewards two views that are wrong in the same way, so we never read it alone (Section 4.5).

**Fake-pixel mask.** The Geo4D data pipeline transforms the right-view pointmap into the reference frame including invalid pixels with $xyz = 0$. After the transform these pixels equal the camera-to-camera translation $t$ ($z$ between 0.18 and 0.65 m), so they pass a $z > 0$ mask although they carry no geometry; 35% to 55% of right-view ground-truth pixels (53% on average) are of this kind. Excluding pixels with $|xyz_{gt} - t| < 2 \cdot 10^{-3}$ changes the teacher's right-view AbsRel from 0.418 to 0.064. All final numbers exclude these pixels; the left view is not transformed and needs no correction.

**Timing and hardware.** Pure inference time covers the conditioner, the sampler, and the decoding of the generated latents only. Our earlier measurements (25.4 s teacher, 5.8 s student) included 2.9 s of evaluation-only work (an extra UNet call for the evaluation loss, ground-truth decoding and encoding, and a duplicated conditioner call). All times below are pure inference times unless stated otherwise, measured on a single RTX 5090 (32 GB), the only GPU used in this work.

## 4.2 Few-Step Sampling Without Training

Reducing the number of Euler steps is the simplest way to make the teacher faster. Table 2 reports the sweep from 25 to 1 step on 10 samples, and Figure 2 plots each metric against the step count. It was run before the fake-pixel mask existed, so the right-view part of CV-Chamfer is contaminated and we use the table for trends only; the times include evaluation overhead. PSNR rises from 19.75 dB at 25 steps to 20.64 dB at 1 step (peaking at 20.66 dB at 2 steps), while CV-Chamfer worsens from 0.176 to 0.194. Between the endpoints, LPIPS increases by 20.7% (0.1222 to 0.1474), sharpness decreases by 12.5%, and seed diversity decreases by 77% (0.0188 to 0.0043).

Table 2: Training-free step reduction of the Geo4D teacher (Euler, 10 samples, pre-maskfix). Fewer steps raise PSNR while LPIPS and diversity degrade; CV-Chamfer degrades below 4 steps. LPIPS and diversity were measured at the two endpoints only.

| steps | s/sample | PSNR ↑ | CV-Chamfer ↓ | LPIPS ↓ | diversity |
|---|---|---|---|---|---|
| 25 | 25.8 | 19.75 | 0.1758 | 0.1222 | 0.0188 |
| 8 | 11.4 | 20.06 | 0.1692 | — | — |
| 4 | 7.9 | 20.35 | 0.1796 | — | — |
| 2 | 6.4 | 20.66 | 0.1919 | — | — |
| 1 | 5.4 | 20.64 | 0.1943 | 0.1474 | 0.0043 |

![Figure 2: Metrics against step count for the training-free teacher. PSNR improves as steps are removed while diversity, LPIPS, sharpness, and CV-Chamfer degrade, each starting at a different step count; the dashed line marks the 25-step value.](figures/fig2_step_sweep.png)

The metrics do not collapse together. Diversity degrades first, from 8 steps; LPIPS from 4 steps; geometry from 2 steps; PSNR never. The rise in PSNR is the perception-distortion trade-off of Blau & Michaeli (2018): with few steps the sampler returns something close to the conditional mean, a blurred average that a pixel-wise metric prefers, and the static background dominates the image. At 1 step the background stays sharp, only the moving arm becomes a translucent smear, and the 4 seeds produce the same image. The ratio of CV-Chamfer to the Chamfer distance between the two ground-truth views is about 2.0 at every step count, consistent with the mIoU of 0.56 to 0.70 that Liu et al. (2025) report; it is a property of the released model, not of the step reduction.

## 4.3 Distillation Results

Table 1 compares the teacher, its training-free few-step variants, the ODE-regression initialization, and the DMD student on the 20-sample protocol. Figure 4 shows qualitative results and Figure 5 the relative depth error maps.

Table 1: Multi-view 4D video generation results. 20 samples (40 view-samples), data seed 1234, fake pixels excluded, diversity measured on 3 samples × 4 seeds. AbsRel is averaged over both views, with left/right in parentheses where measured separately.

| method | training | steps | pure inference | PSNR ↑ | AbsRel ↓ (L/R) | LPIPS ↓ | sharpness | diversity | CV-Chamfer ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Geo4D teacher (Euler) | none | 25 | 21.80 s | 20.62 | 0.066 (0.067/0.066) | 0.118 | 0.0134 | 0.0227 | 0.169 |
| teacher, fewer steps (Euler) | none | 4 | 4.84 s | 21.22 | 0.064 | 0.132 | 0.0121 | 0.0131 | 0.165 |
| teacher, fewer steps (re-noising) | none | 3 | 2.81 s | 21.30 | 0.066 | 0.132 | 0.0119 | 0.0120 | 0.167 |
| ODE regression init (v2) | regression | 4 | 7.9 s* | 14.42 | 0.547 | — | fog | seeds identical | 0.137 |
| DMD student + per-view anchor (bf16) | DMD | 3 | 1.64 s | 20.43 | 0.082 (0.076/0.088) | 0.136 | 0.0136 | 0.0224 | 0.137 |
| DMD student + per-view anchor | DMD | 1 | 1.65 s* | 20.56 | 0.116 | 0.177 | 0.0107 | 0.0117 | 0.165 |

\* v2 time includes evaluation overhead; the 1-step time is fp32. Sharpness is reported relative to the teacher.

**DMD training.** The student is initialized from the ODE-regression model v2 (Section 4.5) and trained with DMD (Yin et al., 2024a) for 2000 steps in 1 h at a peak of 24.4 GB. Figure 3 plots the training diagnostic: every 100 steps we generate 2 fixed pairs and compute the ratio of the standard deviation of the student's $x_0$ prediction to that of the teacher latent (1.0 is teacher level, lower is fog). The ratio rises from 0.79 at step 0 to 0.87 (400), 0.99 (1000), 1.06 (1600), and 1.11 (2000). We select the checkpoint closest to 1.0, step 1600; step 2000 overshoots and evaluates worse. At step 1600 the fog of v2 is gone (10-sample pre-maskfix protocol: PSNR 19.63 dB and AbsRel 0.201 against 14.42 dB and 0.547 for v2).

![Figure 3: DMD training diagnostic. The ratio of student to teacher latent standard deviation rises from 0.79 to 1.11 over 2000 steps; the checkpoint closest to 1.0 (step 1600) is selected.](figures/fig3_dmd_training.png)

**Raw student.** Before calibration, the paired 20-sample comparison shows the student is 3% sharper than the teacher (p=0.024) with equal diversity (p=0.06), but its AbsRel is about twice the teacher's on every sample. An affine scale-and-offset fit to ground-truth depth removes 97% of the gap: the error is a global scale shift, not broken structure. The student predicts depth about 9% too far (median scale 0.909 versus 0.979 for the teacher) and its error maps in Figure 5 are spatially uniform. The input anchor of Section 3.4 reduces the raw 3-step AbsRel of 0.175 to 0.082 without ground truth (Section 4.5).

![Figure 4: Qualitative results, left view. Rows: ground truth, 25-step teacher, training-free 3-step re-noising teacher, 3-step DMD student, 1-step DMD student, at predicted frames 1, 4, 7, and 10; RGB on the left, depth on the right. The 25-step teacher and the 3-step DMD student keep a sharp moving arm, whereas the training-free 3-step teacher and the 1-step student render it as a translucent smear while the static background stays sharp.](figures/fig4_qualitative.png)

![Figure 5: Relative depth error maps for both views. Rows: 25-step teacher, 3-step student, 1-step student, all without the anchor, at predicted frames 1, 5, and 10; brighter is larger error, gray marks pixels without valid depth. The student's error is a spatially uniform tint over the whole scene, stronger in the right view and at 1 step, which is what a global depth scale shift looks like and what the anchor removes; the arm and edge errors are shared with the teacher.](figures/fig5_error_maps.png)

**Anchored student versus teacher.** Paired over the 40 view-samples, the 3-step student differs from the 25-step teacher by −0.19 dB PSNR (p=0.026), +0.015 AbsRel (p<0.001; left +0.008, right +0.022), and +0.018 LPIPS (p<0.001). Sharpness (+0.0001, p=0.95) and diversity (−0.0004, p=0.84) are indistinguishable from the teacher, and CV-Chamfer is lower by 0.034 (p=0.006).

**Training-free versus distilled.** The essential control is the 3-step re-noising teacher in Table 1. It keeps AbsRel at 0.066 and has a higher PSNR (21.30 dB), but its diversity drops by 47% (0.0227 to 0.0120) and its sharpness falls below the teacher's; its 4 seeds produce the same image and the arm is smeared (Figure 4), as in Section 4.2. The 4-step Euler teacher behaves the same way. The DMD student restores diversity (0.0224) and sharpness (0.0136) to the teacher's level at the cost of +0.015 AbsRel and +0.018 LPIPS. This is a trade-off, not a free improvement: if only a point estimate of depth is needed, the training-free few-step teacher is adequate. A world model used for planning must sample several futures and rank them, which a sampler whose seeds all return the same blurred average cannot do; for that use the distilled student is the only few-step option that keeps what the teacher offers.

## 4.4 Inference Time

Table 3 breaks pure inference time into UNet, conditioner, and VAE decoding; Figure 6 plots each configuration against its inference time, and Figure 7 shows the breakdown. In fp32 the 25-step teacher takes 21.75 s, of which 20.13 s is the UNet; the 3-step student takes 2.81 s. The student's UNet costs 0.40 s per call against 0.81 s for the teacher because it runs without classifier-free guidance (Ho & Salimans, 2022), a batch of 20 rather than 40. The remaining 1.6 s (conditioner 0.73 s, decoding 0.88 s) are fixed costs that the step count does not touch.

Table 3: Pure inference time (s) and acceleration variants. Upper block: fp32 breakdown on the eval-free path. Lower block: variants applied cumulatively to the 3-step student, 5 samples with the same seed; AbsRel (L/R) checks for quality drift.

| configuration | UNet | conditioner | VAE decode | total | AbsRel L/R |
|---|---|---|---|---|---|
| teacher, Euler 25 steps (CFG, batch 40), fp32 | 20.13 | 0.73 | 0.89 | 21.75 | — |
| teacher, Euler 4 steps, fp32 | 3.22 | 0.73 | 0.89 | 4.83 | — |
| student, re-noising 3 steps, fp32 | 1.19 | 0.73 | 0.88 | 2.81 | — |
| student, re-noising 1 step, fp32 | 0.40 | 0.73 | 0.88 | 2.01 | — |
| student 3 steps, fp32, unconditional branch omitted | 1.18 | 0.30 | 0.90 | 2.38 | 0.107 / 0.099 |
| + VAE decoder bf16 | 1.18 | 0.30 | 0.68 | 2.16 | 0.108 / 0.099 |
| + UNet and conditioner bf16 | 0.66 | 0.30 | 0.68 | 1.64 | 0.106 / 0.099 |
| + torch.compile (decoder) | 0.66 | 0.30 | 0.33 | 1.29 | 0.105 / 0.098 |
| + torch.compile (UNet) | 0.55 | 0.30 | 0.33 | 1.18 | 0.112 / 0.094 |
| + skip color decoder (depth-only use) | 0.55 | 0.30 | 0.16 | 1.02 | — |

![Figure 6: Speed against quality. AbsRel, LPIPS, and diversity of each configuration against pure inference time; the DMD student is the only few-step point that keeps the teacher's diversity.](figures/fig6_speed_quality.png)

![Figure 7: Pure inference time breakdown. UNet, conditioner, and decoder time for the teacher and each student variant; once the UNet is fast, the fixed cost of conditioner and decoder becomes the bottleneck.](figures/fig7_time_breakdown.png)

The largest single saving comes from precision. The released inference path runs the UNet in fp32 although the model was trained in bf16; casting the UNet and conditioner to bf16 takes the student from 2.16 s to 1.64 s. On the 20-sample protocol bf16 is lossless (AbsRel 0.0813 to 0.0819, LPIPS 0.1360 to 0.1357, diversity identical), so 1.64 s is the student time in Table 1, a 13.3x reduction from the teacher's 21.80 s. torch.compile on the decoder and then the UNet reaches 1.18 s on 5 samples. The compiled UNet was verified on all 20 samples: relative to the teacher, AbsRel +0.016 (p<0.001), LPIPS +0.018 (p<0.001), sharpness +0.0002 (p=0.69), diversity −0.0003 (p=1.00), at about 1.5 s in the evaluation harness. The compiled decoder could not be run in the evaluation path (the sgm VideoDecoder sets a `timesteps` attribute on the module at call time, which conflicts with the compile wrapper), so 1.18 s is verified on 5 samples only. Compiling the conditioner gives no gain; skipping the color decoder when only depth is needed gives 1.02 s.

Against the budgets of Section 1, the quasi-static target of 2 s is met (1.64 s, or 1.18 s with compilation). The humanoid target of 0.3 to 0.5 s and the dynamic target of 0.1 s are not met. The fixed cost of 0.63 s (conditioner 0.30 s, decoder 0.33 s) alone exceeds the humanoid budget, and the UNet (0.55 s for 3 calls) would have to shrink through quantization or lower resolution, which we leave to future work. DreamDojo (Gao et al., 2026) and RoboWorld (Jeon et al., 2026) report 10.8 and 15.3 FPS for robot world models without multi-view geometry evaluation, so their numbers are not comparable to ours.

## 4.5 Ablations and Negative Results

**Input-anchored depth calibration.** Table 4 and Figure 8 isolate the anchor on the 10-sample protocol with the fake-pixel mask. For the left view the anchor scale computed from the conditioning pointmap equals the ground-truth oracle scale to three decimals, and the left AbsRel moves from 0.146 to 0.086 against 0.080 for the teacher (p=0.43). The right view needs its own anchor, because its conditioning pointmap is in the right camera frame while predictions are in the reference frame; transformed with the dataset extrinsics, it matches ground-truth frame 0 to an AbsRel of 0.001 and 0.006 on the first two samples. With a per-view anchor the student reaches 0.086 on both views from a raw 0.146 / 0.205; an affine variant with scale and offset gave no further gain. The anchor also improves the teacher, from 0.072 to 0.060, so part of the teacher's own error is a global scale shift. On 20 samples the student's AbsRel goes from 0.175 to 0.082 (Table 1).

Table 4: Anchor ablation (AbsRel, 10 samples, fake pixels excluded; mean over both views). The per-view anchor closes most of the gap between the raw student and the teacher, and also improves the teacher; adding an offset to the scale gives no further gain.

| configuration | left | right | mean |
|---|---|---|---|
| teacher, 25 steps | 0.080 | 0.064 | 0.072 |
| teacher, 25 steps + per-view anchor | 0.068 | 0.052 | 0.060 |
| student, 3 steps, raw | 0.146 | 0.205 | 0.175 |
| student, 3 steps + per-view anchor (scale) | 0.086 | 0.086 | 0.086 |
| student, 3 steps + per-view anchor (scale and offset) | 0.089 | 0.081 | 0.085 |

![Figure 8: Input-anchored calibration. Per-view AbsRel (10 samples, fake pixels excluded) of the 25-step teacher, the raw 3-step student, the student with the per-view anchor, the student with the scale-plus-offset anchor, and the teacher with the per-view anchor; dashed lines mark the unanchored teacher. The anchor needs only the conditioning pointmap.](figures/fig8_anchor.png)

**ODE regression initialization.** Regressing the student from noise to the teacher's 25-step output, using 284 (noise, final latent) pairs (142 samples × 2 seeds), failed in two forms. Version v1 regressed on pseudo-trajectories $x_\sigma = z + \sigma \varepsilon$ for 300 steps; the loss fell from 0.45 to 0.22 and CV-Chamfer improved by 31%, but AbsRel became three times worse and PSNR fell by 4.2 dB: the two views agreed with each other and were both wrong, the cross-view metric trap of Section 4.1. Version v2 regressed on real intermediate states captured at $\sigma$ = 700, 70.5, and 2.3 for 1200 steps; the $\sigma$ = 2.3 term fit, the two high-noise terms plateaued at 0.40, and the output was global fog with 4 identical seeds (PSNR 14.42 dB, AbsRel 0.547, CV-Chamfer 0.137 on 10 samples). Mean-squared regression from noise to sample converges to the mean even with true trajectories, as also observed in Appendix C.2 of Causal Forcing (Zhu et al., 2026); v2 served only as the DMD initialization.

**GT-supervised cross-view depth-ratio loss.** We added to DMD a log error between the ratio of mean view depths $r = \bar{z}_R / \bar{z}_L$ and a target ratio, decoding one frame through the pointmap VAE in bf16 with gradient checkpointing (four frames in fp32 ran out of memory). The weight was balanced at the $x_0$-gradient level, $\lambda = \beta \, |g_{\mathrm{DMD}}| / |g_{\mathrm{cv}}|$; a fixed $\lambda = 1$ made the cross-view gradient 500 times the DMD gradient and erased distillation. Table 5 summarizes the three runs; all lost sharpness. A supervision signal that contradicts the teacher distribution fights the distillation objective, whereas the input-only anchor, which never enters training, does not. The target in 6b2 and 6b3 was the contaminated right-view ground truth of Section 4.1, so whether a clean target also conflicts is unverified. A self-anchor loss that matches predicted frame-0 depth to the conditioning depth without ground truth is implemented and training is in progress at the time of writing; we report no result for it.

Table 5: GT-supervised cross-view depth-ratio loss added to DMD. None of the three runs improved the depth ratio without losing sharpness.

| run | target | β | outcome |
|---|---|---|---|
| 6b | teacher $r$ | 1 | $r$ unchanged through 1000 steps, sharpness recovery slowed |
| 6b2 | GT $r$ | 10 | $r$ overshoots target, sharpness stalls |
| 6b3 | GT $r$ | 3 | $r$ regresses to dataset mean; finished 2000 steps; eval PSNR −3.5 dB, LPIPS 0.36 (collapse) |

**One-step student.** With a single re-noising step (Table 1) the same student reaches AbsRel 0.116 and LPIPS 0.177, and its diversity halves to 0.0117, the level of the training-free few-step teacher. It saves 0.79 s of UNet time in fp32 (Table 3) but does not meet the quality bar, so we do not use it.
