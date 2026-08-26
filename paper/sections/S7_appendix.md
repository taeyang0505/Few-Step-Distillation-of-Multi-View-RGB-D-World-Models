# A Appendix

## A.1 Model and Sampler Details

Geo4D (Liu et al., 2025) is built on SVD (Blattmann et al., 2023) with the EDM parameterization (Karras et al., 2022). One sample is a latent of shape (2 views × 10 frames, 8 channels, 32 × 40) for 256 × 320 images, 4 latent channels for the pointmap and 4 for color. The teacher sampler is EulerEDM with 25 steps, σ_max = 700, and a linear classifier-free guidance guider from 1.0 to 2.5 (Ho & Salimans, 2022), so every step runs the UNet on a batch of 40 (20 conditional plus 20 unconditional).

**Sigma schedule.** The four-step EDM schedule with σ_max = 700 produced by the Geo4D sampler has sigmas 700, 70.5, 2.3, and a final slot near zero. The student uses the first three; the final slot corresponds to its x₀ output. The same sigmas are the intermediate states captured for the ODE regression v2 initialization.

**Re-noising sampler.** The student predicts x₀ at each sigma and is re-noised to the next sigma with fresh Gaussian noise; there is no Euler integration and no guidance.

```
sigmas = [700, 70.5, 2.3]
x = sigmas[0] * randn_like(z)                 # pure noise
for i, s in enumerate(sigmas):
    x0 = D_student(x, s, c)                   # x0 prediction, no CFG
    if i + 1 < len(sigmas):
        x = x0 + sigmas[i + 1] * randn_like(x0)   # re-noise to next sigma
return x0
```

**Guidance.** The CFG guider (1.0 to 2.5) is used only inside the teacher, at teacher inference and as the real-score network in DMD. The student runs without guidance, which is why a student UNet call costs 0.40 s in fp32 against 0.81 s for the teacher.

## A.2 Dataset and Evaluation Details

**Data.** We use the four inference episodes (218 frames) of PlaceAppleFromBowlIntoBin from the LBM environment released by Geo4D, with four cameras (scene_13 to scene_16), stride 5, one conditioning frame followed by 10 predicted frames, and 256 × 320 images. Two cameras are sampled at random per sample with data seed 1234; the left camera is the reference frame.

**Fake-pixel mask.** The dataset transforms the right-view pointmap into the reference frame including invalid pixels with xyz = 0. After the transform these pixels equal the camera-to-camera translation t, with z between 0.18 m and 0.65 m, so they pass a z > 0 mask. Between 35% and 55% of right-view ground-truth pixels (53% on average) are of this kind. We exclude pixels with |gt_xyz - t| < 2e-3, which changes the teacher's right-view AbsRel from 0.418 to 0.064. The left view is not transformed, so its invalid pixels remain zero. All final numbers use this mask.

**CV-Chamfer.** The symmetric chamfer distance between the two views' predicted point clouds, both in the reference frame. It is scale dependent and rewards two views that are wrong in the same way (Finding 3). Its ratio to the chamfer distance between the two ground-truth clouds is about 2.0 regardless of the number of steps, consistent with the cross-view mIoU of 0.56-0.70 in Geo4D Table 1; this is a property of the released model, not of our samplers.

**Seed diversity.** For the first 3 samples we generate with seeds 0-3, compute the per-pixel standard deviation across seeds, and average it.

**Statistics.** All comparisons against the teacher are paired Wilcoxon signed-rank tests (Wilcoxon, 1945) over n = 40 view-samples with the same data and generation seeds.

## A.3 Training Details

**ODE pairs.** 284 (noise, final latent) pairs were generated from the 25-step teacher (142 dataset samples × 2 seeds) in about 2 h. The v2 regression initialization was trained for 1200 steps on intermediate states at σ = 700, 70.5, and 2.3; the σ = 2.3 term fits while the other two plateau at 0.40.

**DMD.** The generator is the v2-initialized student, the real score is the frozen teacher with its CFG guider, and the fake score is a trainable copy of the teacher. The DMD loss (Yin et al., 2024a) is expressed in EDM sigma space with DMD2-style backward simulation (Yin et al., 2024b): a random k in {0, 1, 2} student steps are run and only the last x₀ prediction receives gradient. Training runs 2000 steps at batch 1 in 1 h. To fit 32 GB, all three UNets are in bf16 (9.2 GB), the optimizer is AdamW8bit, the conditions (c, uc) are precomputed on CPU, and the conditioner and VAEs are moved off the GPU; peak memory is 24.4 GB. Every 100 steps we generate two fixed pairs and record std(student x₀) / std(teacher latent), which selects step 1600.

**Cross-view loss.** The GT-supervised depth-ratio loss is the log error of r = mean(z_R) / mean(z_L) against the teacher's or the ground-truth ratio. It decodes one frame through the pointmap VAE in bf16 with gradient checkpointing (four frames in fp32 ran out of memory). Its weight is set at the x₀-gradient level as λ = β |g_DMD| / |g_cv|; a fixed λ = 1 made the cross-view gradient 500x the DMD gradient and erased the DMD signal.

**Runs on a shared GPU.** The GPU was shared with other users. The first attempt at the 4000-step run of Section 4.6 stopped at step 850 with a CUDA out-of-memory error because another process held 8.87 GB at the time. We added a `--resume` flag that restarts from the periodic checkpoints written every 200 steps (the optimizer moments restart from zero) and a launcher that retries after an out-of-memory error; the second attempt completed without interruption.

## A.4 Additional Results and Findings

The numbered findings from the experiment log are: (1) fewer steps raise PSNR but worsen LPIPS, diversity, and geometry; (2) collapse order is diversity, LPIPS, geometry, then PSNR; (3) CV-Chamfer rewards two views wrong in the same way and must not be used alone; (4) MSE regression initialization converges to the mean even with true trajectories, consistent with Causal Forcing (Zhu et al., 2026, App. C.2); (5) DMD restores sharpness and diversity within about 400 updates and the std ratio selects the checkpoint; (6) "the teacher right view collapses", retracted once the fake-pixel contamination was found; (7) GT-supervised geometry loss conflicts with DMD in three runs, the input-only anchor does not; (8) the two Geo4D views share all UNet weights; (9) a training-free few-step teacher loses 47% of diversity regardless of sampler; (10) 53% of right-view GT pixels are transformed invalid pixels; (11) earlier times included evaluation scaffolding; (12) the input anchor also improves the teacher (0.072 to 0.060).

**Weight sharing.** In Geo4D's wrappers.py the second-view UNet is a shallow copy of the first, followed by a deep copy of output_blocks only. The shallow copy shares the `_modules` dictionary, so the new output_blocks are written into the original module as well. We verified this with a toy test and by checkpoint comparison (all 795 output_blocks keys identical). The right view differs only through the spatial context (left-view hidden states) and its own conditioning latent.

**Timing correction.** The earlier figures of 25.4 s (teacher) and 5.8 s (student) came from the log_images path and included 2.9 s of evaluation-only work: an extra UNet call for the evaluation loss (0.40 s), ground-truth reconstruction decoding (0.88 s), ground-truth encoding (0.49 s), and a duplicated conditioner call (1.09 s). Pure inference, defined as conditioner, sampler, and decoding of the generated latents, is 21.75 s and 2.81 s in fp32 (Table 3). Because the rows of Table 3 were collected in separate runs, we later re-measured the teacher and the three student schedules together in one process; the teacher came out at 21.49 s and the students at 1.64, 1.86, and 2.08 s, within 1.2% of the separate runs. The speed-ups reported in the paper use this joint run, so the headline is 13.1x rather than the 13.3x implied by the two separate measurements. The difference is run-to-run variance, not a change of configuration.

## A.5 The Use of Large Language Models

An LLM assistant (Claude) was used to write and debug experiment scripts, to draft text, and to organize the experiment notes. All numbers were measured by the author, and the author verified every figure and claim in the paper.

## A.6 Toward Policy-Level Evaluation

For context, Geo4D Table 2 reports success rates over 30 rollouts of 0.53 on PlaceAppleFromBowlIntoBin (Dreamitate 0.10, DP 0.00, DP3 0.00; Liang et al., 2024; Chi et al., 2023; Ze et al., 2024), 0.73 on StoreCerealBoxUnderShelf, and 0.67 on PutSpatulaOnTableFromUtensilCrock, at about 30 s per 10 frames on an RTX 4090. Reproducing this requires the LBM simulator, now open-sourced as lbm_eval (Toyota Research Institute, 2025), the 6-DoF pose tracker of Geo4D Section 3.4 (not released), gripper-openness inference, and the policy interface; none of these are in the released Geo4D code, and we did not run it.

As a simulator-free proxy we ran bench_policy_proxy.py on the main student (6a step 1600, per-view anchor), the training-free 3-step re-noising teacher, and the 25-step teacher on the same 20 samples (40 view-samples; the apple is visible in 37). Regions come from the ground-truth label maps: gripper = label ids 29 to 35 (2.35% of pixels), apple = id 44 (0.23%), background = the rest. AbsRel and PSNR are computed inside the masks dilated by 3 pixels, and the 3D centroid error is the distance in cm between the mean predicted pointmap and the mean ground-truth pointmap inside the ground-truth mask, over predicted frames 1 to 10. Table 9 gives the results.

Table 9: Region metrics on the gripper and the apple (20 samples, 40 view-samples, frames 1 to 10). The student's whole-image AbsRel gap comes from the background; on the gripper and the apple it is not distinguishable from the teacher by the Wilcoxon test, but its gripper centroid error is 2.0 cm higher with a 90% confidence interval that excludes zero.

| metric | teacher | training-free 3-step | student 3-step | student − teacher [90% CI] | p |
|---|---|---|---|---|---|
| gripper AbsRel | 0.206 | 0.203 | 0.225 | +0.019 [−0.004, +0.042] | 0.24 |
| gripper centroid error (cm) | 12.3 | 12.8 | 14.3 | +2.0 [+0.3, +3.7] | 0.11 |
| gripper centroid error, last frame (cm) | 12.4 | 13.6 | 15.4 | +3.0 [+0.2, +5.8] | 0.064 |
| apple AbsRel | 0.127 | 0.124 | 0.115 | −0.012 [−0.025, +0.001] | 0.27 |
| apple centroid error (cm) | 11.4 | 11.3 | 12.9 | +1.6 [−0.1, +3.2] | 0.13 |
| background AbsRel | 0.063 | 0.063 | 0.078 | +0.015 [+0.010, +0.021] | <0.001 |

Per frame, the gripper centroid error in cm over frames 1 to 10 is 11.0, 12.6, 13.7, 13.9, 13.2, 12.8, 11.8, 11.2, 10.8, 12.4 for the teacher and 11.9, 11.7, 16.6, 17.3, 15.4, 14.9, 14.6, 13.2, 12.5, 15.4 for the student. The training-free 3-step teacher has a higher region PSNR than the 25-step teacher (gripper +0.80 dB, p<0.001; apple +0.63 dB, p<0.001), so blur helps PSNR on the moving regions as well (Finding 1).

Two caveats apply. First, the ground-truth mask is applied to the prediction, so when the predicted arm is displaced the mask covers background pixels and the centroid jumps; the teacher's own error of about 12 cm is inflated by this mask mismatch and by the multimodal future, and only differences between methods are interpretable. Second, with 40 view-samples the minimal detectable effects are about 3 cm for the gripper centroid, 2.9 cm for the apple centroid, and 0.039 for gripper AbsRel. With these caveats, the whole-image AbsRel gap of +0.015 (Section 4.3) comes from the static background, where the student's difference is the same +0.015 (p<0.001); on the gripper and apple regions the student is not statistically distinguishable from the teacher by the Wilcoxon test, but its gripper centroid error is 2.0 cm higher with a 90% confidence interval [+0.3, +3.7] that excludes zero, which sits at the practical 2 cm margin of Section 4.1 and should be read as a likely small degradation, not as equality. On the 60-sample rerun of Section 4.7, however, the main student's gripper centroid difference is +0.66 cm [−0.62, +1.94] and the 5-step configuration's is −0.75 cm [−2.25, +0.75]; the 20-sample degradation above did not replicate, underscoring the small-sample caveat.

The same region metrics, rerun on the self-anchored students of Section 4.6, give a gripper centroid error of +3.1 cm against the teacher (p=0.009) for run 6d at step 3200 and +4.7 cm (p<0.001) for run 6e at step 3200, versus +2.0 cm (p=0.11) for the main student; this is the basis for not adopting those checkpoints.
