# FACTS SHEET — single source of truth for the paper draft

Every number below was measured on one RTX 5090 (32 GB) between 2026-08-20 and 2026-08-23, on the public Geo4D checkpoint for the
PlaceAppleFromBowlIntoBin task. Writers MUST use only numbers from this sheet. Do NOT invent numbers, baselines, tasks, or results.
Anything marked "NOT DONE" must be described as future work, never as a result.

## 0. Working title and author
- Title: "Few-Step Distillation of Multi-View RGB-D World Models: Does Geometry Survive?"
  (alternative: "Distilling Geometry-Aware 4D Video Generation to Three Steps")
- Author: Tae Yang Hong (undergraduate researcher, NAIS lab). Affiliation line: "NAIS Lab, [University — to be confirmed]".
- Code: https://github.com/taeyang0505/Few-Step-Distillation-of-Multi-View-RGB-D-World-Models

## 1. Problem and setting
- Teacher: Geo4D (Liu et al., 2025; arXiv:2507.01099) — "Geometry-aware 4D Video Generation for Robot Manipulation". SVD-based latent video
  diffusion that, given one RGB-D (pointmap + color) frame per view for two camera views, generates 10 future RGB-D frames for both views.
  Cross-view pointmap alignment supervision at training time; both views' pointmaps are expressed in the left (reference) camera frame;
  no camera poses at inference. Sampler: EulerEDM, 25 steps, sigma_max = 700, classifier-free guidance (linear guider 1.0 -> 2.5).
  Latent shape per sample: (2 views x 10 frames, 8 channels, 32 x 40) for 256 x 320 images; 4 latent channels pointmap + 4 color.
  Geo4D's own stated limitation (Sec. 5): inference ~30 s for 10 frames on an RTX 4090, too slow for closed-loop planning.
- Target inference budgets we set: quasi-static manipulation (pick-and-place) 2 s; humanoid 0.3-0.5 s; dynamic manipulation 0.1 s.
- Goal: cut inference time AND measure whether Geo4D's advantages (cross-view geometric consistency, video quality) survive; repair if not.
- What is new (honest framing): DMD, re-noising few-step sampling, bf16, torch.compile are existing techniques. Contributions are
  (1) first geometry-aware evaluation of few-step distillation for a multi-view RGB-D world model, (2) evaluation findings/pitfalls
  (pixel metrics hide collapse; collapse order; fake-pixel contamination; cross-view metric trap), (3) input-anchored per-view depth
  calibration that needs no ground truth, (4) negative results (ODE regression init; GT-supervised geometry loss conflicts with DMD).
- Geo4D's two views share 100% of UNet weights: in wrappers.py, `diffusion_model_2 = copy(diffusion_model)` then deepcopy of output_blocks
  only; the shallow copy shares the `_modules` dict so the new output_blocks are written into the original too. Verified by a toy test and by
  checkpoint key comparison (all 795 output_blocks keys identical). The right view differs only via spatial_context (left view hidden states)
  and its own conditioning latent.

## 2. Data and evaluation protocol
- Task 3 of Geo4D: PlaceAppleFromBowlIntoBin (bimanual; left arm lifts the apple from the bowl, hands it to the right arm, right arm
  places it into the bin). Simulation data from the LBM (Drake) environment released by Geo4D; 4 inference episodes (218 frames),
  4 cameras (scene_13..16), stride 5, 1 conditioning frame + 10 predicted frames, 256 x 320, two cameras sampled at random per sample.
- Evaluation: 20 samples (= 40 view-samples), fixed data seed 1234, fixed generation seed 0, paired Wilcoxon signed-rank tests
  (n = 40 view-samples). Seed diversity: first 3 samples x 4 seeds (0-3), per-pixel std across seeds averaged. Early sweeps used 10 samples.
- Metrics: PSNR (RGB, [0,1]); AbsRel on depth z (meters) with valid mask; LPIPS (AlexNet); sharpness = variance of Laplacian of the
  gray image; seed diversity (std across 4 seeds); CV-Chamfer = symmetric chamfer distance between the two views' predicted point clouds
  (both in the reference frame) — lower is better, but it is scale dependent and rewards "two views wrong in the same way".
- Fake-pixel finding (Finding 10): the dataset transforms the right view pointmap into the reference frame INCLUDING invalid pixels
  (xyz = 0), which become xyz = the camera-to-camera translation (z 0.18-0.65 m); 35-55% of right-view GT pixels (53% on average) pass a z>0
  mask but are fake. Excluding them (pixels where |gt_xyz - t| < 2e-3) changes teacher right-view AbsRel from 0.418 to 0.064. All final
  numbers exclude these pixels. Left view has no transform so its invalid pixels stay 0.
- Pure inference time = conditioner -> sampler -> decoding of the generated latents only. Earlier reported numbers (25.4 s teacher,
  5.8 s student) included 2.9 s of evaluation-only work (extra UNet call for eval loss 0.40 s, GT reconstruction decoding 0.88 s,
  GT encoding 0.49 s, duplicated conditioner call 1.09 s). Finding 11.

## 3. Teacher profile (Step 0)
- log_images path: 25.4 s total, UNet 21.14 s (26 calls, 0.81 s/call, 83.8%), conditioner 1.81 s (7.2%), VAE decode 1.75 s (7.0%),
  VAE encode 0.48 s (1.9%); peak VRAM 13.2 GB. Bottleneck = running the same UNet 25 times, not memory.
- Pure inference (uc included, CFG batch of 40): UNet 20.13, conditioner 0.73, VAE decode 0.89, total 21.75 s (rounded 21.8 s).

## 4. Step reduction without training (Step 1) — teacher, Euler, 10 samples, old mask (contaminated right view; use for trends only)
| steps | s/sample | PSNR | CV-Chamfer |
|---|---|---|---|
| 25 | 25.8 | 19.75 | 0.1758 |
| 8 | 11.4 | 20.06 | 0.1692 |
| 4 | 7.9 | 20.35 | 0.1796 |
| 2 | 6.4 | 20.66 | 0.1919 |
| 1 | 5.4 | 20.64 | 0.1943 |
- From 25 to 1 step (blur test): LPIPS +20.7% (0.1222 -> 0.1474), Laplacian sharpness -12.5%, seed std -77% (0.0188 -> 0.0043).
- Collapse order: diversity (from 8 steps) -> LPIPS (4 steps) -> geometry (2 steps) -> PSNR (never). PSNR rises because few-step output
  is the conditional mean (blurry average); background pixels dominate. Qualitatively at 1 step only the moving arm becomes a translucent
  smear; background stays sharp; 4 seeds give the same image. (Perception-distortion tradeoff, Blau & Michaeli 2018.)
- CV-Chamfer ratio to GT-vs-GT chamfer is ~2.0 regardless of steps, consistent with Geo4D Table 1 mIoU 0.56-0.70 (a property of the
  released model, not a defect).

## 5. Distillation
### 5.1 ODE pairs and ODE regression initialization (Step 6-1, 6-2) — FAILED (negative result)
- 284 (noise, final latent) pairs from the 25-step teacher (142 dataset samples x 2 seeds), ~2 h.
- v1: pseudo trajectories x_sigma = z + sigma*eps, 300 steps, loss 0.45 -> 0.22; eval: CV-Chamfer -31% ("better"), AbsRel 3x worse,
  PSNR -4.2 dB. Two views agree with each other but both are wrong (cross-view metric trap, Finding 3).
- v2: real intermediate states captured at sigma = 700, 70.5, 2.3; 1200 steps; sigma 2.3 term fits, sigma 700/71 terms plateau at 0.40;
  output = global fog, 4 seeds identical. v2 eval: PSNR 14.42, AbsRel 0.547, CV 0.137 (10 samples). Used only as DMD initialization.
  Conclusion: MSE regression from noise to sample converges to the mean even with true trajectories (Finding 4); consistent with
  Causal Forcing App. C.2.
### 5.2 DMD (Step 6-3) — SUCCESS
- geo4d_dmd_train.py: generator = v2-initialized student; real score = frozen teacher with CFG guider; fake score = trainable teacher copy.
  DMD loss (Yin et al., 2024, eq. 7-8) moved to EDM sigma space; DMD2-style backward simulation (random k in {0,1,2}; only the last x0
  prediction receives gradient). Student schedule = three sigmas [700, 70.5, 2.3] (4-step EDM schedule minus the sigma~0 slot).
  Inference uses x0 prediction followed by re-noising to the next sigma (RenoiseSampler), not Euler integration. No CFG at inference.
- Memory recipe for 32 GB: all three UNets in bf16 (9.2 GB), AdamW8bit, conditions (c, uc) precomputed and kept on CPU, conditioner and
  VAEs moved off GPU during training; peak 24.4 GB; 2000 steps in 1 h; batch 1; lr as in script (AdamW8bit).
- Training diagnostic every 100 steps: generate 2 fixed pairs, std ratio = std(student x0) / std(teacher latent) (1.0 = teacher level,
  lower = fog). Trajectory: 0.79 (step 0) -> 0.87 (400) -> 0.99 (1000) -> 1.06 (1600) -> 1.11 (2000). Best checkpoint = step 1600
  (ratio closest to 1.0); step 2000 overshoots and is worse. Cheap checkpoint selection criterion (Finding 5).
- First eval of step 1600 (10 samples, old mask): 3-step PSNR 19.63, AbsRel 0.201, CV 0.167 — fog gone vs v2 (14.42 / 0.547).
- Precise analysis (n=20 paired): student sharpness +3% vs teacher (p=0.024), diversity equal (p=0.06), AbsRel ~2x worse on every sample;
  aligning predicted depth to GT with an affine (scale+offset) fit removes 97% of the AbsRel gap -> global scale/offset shift, not broken
  structure. Student predicts depth consistently ~9% too far (median scale 0.909 vs teacher 0.979).
### 5.3 Input-anchored depth calibration (Step 6-4a, training-free) — SUCCESS
- Geo4D receives the conditioning frame's pointmap as input, so a scale that aligns predicted frame 0 to the conditioning pointmap can be
  computed at inference WITHOUT ground truth (median ratio). Left view: scale equals the GT oracle scale to 3 decimals; left AbsRel
  0.146 -> 0.086 vs teacher 0.080 (p=0.43, indistinguishable) [10-sample, pre-maskfix numbers].
- Right view needs its own anchor: the right conditioning pointmap is in the right camera frame while predictions are in the reference
  frame, so we transform it with inv(E_left) @ E_right using the dataset extrinsics (verified: transformed condition vs GT frame 0 AbsRel
  0.001/0.006 on samples 1-2). Per-view anchor = config "b". A robust affine variant (scale+offset, "c") gave no further gain.
- With fake-pixel mask (eval_6x_maskfix, 10 samples): teacher L/R 0.080 / 0.064; student 3-step raw 0.146 / 0.205; student + per-view
  anchor 0.086 / 0.086.
- The anchor also improves the teacher itself: teacher AbsRel 0.072 -> 0.060 (Finding 12).
- Main-table (20 samples) raw student S3 AbsRel without anchor: 0.175 (for reference); with anchor 0.082.
### 5.4 GT-supervised cross-view depth-ratio loss (Step 6-4c) — FAILED 3 times (negative result)
- Added to DMD: log error of view depth ratio r = mean(z_R)/mean(z_L) vs teacher's r (6b) or GT's r (6b2, 6b3); decode 1 frame through the
  pointmap VAE in bf16 with gradient checkpointing (4 frames fp32 OOM); weight balanced automatically at the x0-gradient level
  (lambda = beta * |g_DMD| / |g_cv|; fixed lambda=1 made the cv gradient 500x the DMD gradient and erased DMD).
| run | target | beta | outcome |
|---|---|---|---|
| 6b | teacher r | 1 | r unchanged through 1000 steps, sharpness recovery slowed |
| 6b2 | GT r | 10 | r overshoots target, sharpness stalls |
| 6b3 | GT r | 3 | r regresses to dataset mean; finished 2000 steps; eval PSNR -3.5 dB, LPIPS 0.36 (collapse) |
- All three lost sharpness: a supervision signal that contradicts the teacher distribution fights the distillation objective (Finding 7).
  Caveat: the GT target was the contaminated right-view GT (Sec. 2), so whether clean GT also conflicts is unverified.
- Self-anchor loss (match predicted frame-0 depth to the conditioning-frame depth, no GT) is implemented, training in progress — NOT DONE.

## 6. Inference acceleration (Step 7)
### 6.1 Pure inference breakdown (fp32, eval-free path)
| path | UNet | conditioner | VAE decode | total |
|---|---|---|---|---|
| teacher Euler 25 steps (CFG, batch 40) | 20.13 | 0.73 | 0.89 | 21.75 s |
| teacher Euler 4 steps | 3.22 | 0.73 | 0.89 | 4.83 s |
| student re-noising 3 steps | 1.19 | 0.73 | 0.88 | 2.81 s |
| student re-noising 1 step | 0.40 | 0.73 | 0.88 | 2.01 s |
- Student UNet per call 0.40 s = half the teacher's 0.81 s because no CFG (batch 20 instead of 40). Fixed cost = 0.73 + 0.88 = 1.6 s.
### 6.1b Joint re-measurement, teacher and students in ONE run (08-25, bench_timing_final.py, 5 batches, RTX 5090)
Authoritative source of the headline speed-up: teacher and student measured in the same process, same harness, same data.
| configuration | total | UNet (calls) | conditioner (calls) | VAE decode | anchor |
|---|---|---|---|---|---|
| T25 teacher, Euler 25 steps, CFG, fp32 | 21.49 s | 19.95 (25) | 0.60 (2) | 0.92 | - |
| S3b student, re-noising 3 steps, anchor, bf16 | 1.64 s (13.1x) | 0.66 (3) | 0.30 (1) | 0.68 | 0.001 |
| S4b student, 4 steps | 1.86 s (11.6x) | 0.88 (4) | 0.30 (1) | 0.68 | 0.001 |
| S5b student, 5 steps | 2.08 s (10.3x) | 1.09 (5) | 0.30 (1) | 0.68 | 0.001 |
- Replaces the earlier separate measurements (teacher 21.75 s) and the ESTIMATED 4/5-step times of N12 (1.86 / 2.08 s), which the
  re-measurement reproduced exactly. Paper headline updated to 1.64 s vs 21.49 s = 13.1x (was 13.3x vs 21.8 s).
- Teacher UNet per call 0.798 s (CFG, batch 40) vs student 0.219 s (no CFG, batch 20) = 3.6x, of which about 2x is the dropped uc branch.
- Per-view anchor costs 0.001 s — free relative to everything else.
- Fixed cost in this bf16, non-compiled configuration = conditioner 0.30 + decode 0.68 = 0.98 s (0.63 s after decoder compile, 6.2).
- H4b (hybrid 4 steps) routes only the last call to the teacher UNet without CFG at the same batch, so its cost equals S4b.
- Raw output: results/quantitative/timing_final.txt and timing_final_raw.json.
### 6.2 Acceleration variants (5 samples, same seed; AbsRel L/R shown to verify no quality drift)
| variant | total | UNet | cond | decode | AbsRel L/R |
|---|---|---|---|---|---|
| fp32, uc omitted | 2.38 | 1.18 | 0.30 | 0.90 | 0.107 / 0.099 |
| + VAE decoder bf16 | 2.16 | 1.18 | 0.30 | 0.68 | 0.108 / 0.099 |
| + UNet and conditioner bf16 | 1.64 | 0.66 | 0.30 | 0.68 | 0.106 / 0.099 |
| + torch.compile (decoder) | 1.29 | 0.66 | 0.30 | 0.33 | 0.105 / 0.098 |
| + torch.compile (UNet) | 1.18 | 0.55 | 0.30 | 0.33 | 0.112 / 0.094 |
| + compile conditioner | 1.18 (no gain; conditioner stays 0.30) | | | | |
| + skip color decoder (depth-only use) | 1.02 | 0.55 | 0.30 | 0.16 | — |
- Biggest waste: the released inference path ran the UNet in fp32 (only training was bf16).
- bf16 verified lossless on 20 samples (AbsRel 0.0813 -> 0.0819, LPIPS 0.1360 -> 0.1357, diversity identical).
- UNet torch.compile verified on 20 samples paired: vs teacher AbsRel +0.016 (p<0.001), LPIPS +0.018 (p<0.001), sharpness +0.0002
  (p=0.69), diversity -0.0003 (p=1.00) — same conclusions as bf16; time about 1.5 s in the eval harness. Decoder compile could not be run in
  the eval path (sgm VideoDecoder sets a `timesteps` attribute on the module at call time, which conflicts with the torch.compile wrapper),
  so 1.18 s is verified on 5 samples only. Headline number in the paper: 1.64 s (bf16), 13.1x faster than 21.49 s (joint re-measurement, 6.1b).
- Remaining fixed cost 0.63 s (conditioner 0.30 + decode 0.33) blocks the humanoid target; the UNet itself (0.55 s for 3 calls) must shrink
  (quantization, resolution) for 0.3-0.5 s.

## 7. MAIN TABLE (20 samples = 40 view-samples, data seed 1234, fake pixels excluded, diversity 3 samples x 4 seeds)
| method | training | steps | pure inference | PSNR ^ | AbsRel v (L/R) | LPIPS v | sharpness | diversity | CV-Chamfer v |
|---|---|---|---|---|---|---|---|---|---|
| Geo4D teacher (Euler) | none | 25 | 21.49 s | 20.62 | 0.066 (0.067/0.066) | 0.118 | 0.0134 | 0.0227 | 0.169 |
| teacher, fewer steps (Euler) | none | 4 | 4.84 s | 21.22 | 0.064 | 0.132 | 0.0121 | 0.0131 | 0.165 |
| teacher, fewer steps (re-noising) | none | 3 | 2.81 s | 21.30 | 0.066 | 0.132 | 0.0119 | 0.0120 | 0.167 |
| ODE regression init (v2) | regression | 4 | 7.9 s* | 14.42 | 0.547 | — | fog | seeds identical | 0.137 |
| DMD student + per-view anchor (bf16) | DMD | 3 | 1.64 s | 20.43 | 0.082 (0.076/0.088) | 0.136 | 0.0136 | 0.0224 | 0.137 |
| DMD student + per-view anchor | DMD | 1 | 1.65 s* | 20.56 | 0.116 | 0.177 | 0.0107 | 0.0117 | 0.165 |
* v2 time includes eval overhead; 1-step time is fp32. GT sharpness reference ~ same order as teacher (report only relative).
- Paired Wilcoxon, student 3-step vs teacher (n=40): PSNR -0.19 (p=0.026); AbsRel +0.015 (p<0.001; left +0.008, right +0.022);
  LPIPS +0.018 (p<0.001); sharpness +0.0001 (p=0.95); diversity -0.0004 (p=0.84); CV-Chamfer -0.034 (p=0.006, i.e. better).
- Training-free 3-step teacher (re-noising, "T3r"): AbsRel unchanged, PSNR even higher, but diversity -47% (0.0227 -> 0.0120) and lower
  sharpness; 4 seeds produce the same image; arm smeared. Essential control row (Finding 9).
- 1-step student: diversity halves (0.0117) and LPIPS 0.177 -> does not meet the quality bar.
- Speed target status: quasi-static 2 s met (1.64 s; 1.18 s with compile); humanoid 0.3-0.5 s not met; dynamic 0.1 s not met.

## 8. Qualitative assets (PNG grids in repo results/qualitative/)
- dmd6a_qual/rgb_left.png, depth_left.png: rows = GT, teacher 25, DMD student 3-step, student 1-step (left view).
- dmd6a_qual_lr/err_left.png, err_right.png: relative depth error maps, rows = teacher 25, student 3, student 1 (brighter = larger error);
  student error is spatially uniform (global scale shift), removed by the anchor.
- dmd6a_qual_lr/rgb_right.png, depth_right.png: right view.
- qualitative/: training-free step reduction (1 step: arm smear, 4 seeds identical); compare_left.gif animated.
- student_qual/: ODE regression init fog.  t3r_qual/: training-free 3-step teacher; 4 seeds identical.
- diversity_*.png: 4 seeds side by side.

## 9. Findings list (numbered as in the log; 6 was retracted)
1 Fewer steps raise PSNR but worsen LPIPS/diversity/geometry — pixel metrics hide few-step collapse.
2 Collapse order: diversity -> LPIPS -> geometry -> PSNR.
3 CV-Chamfer rewards "two views wrong in the same way" and scales with depth; never use alone.
4 MSE regression init converges to the mean even with true trajectories.
5 DMD restores sharpness and diversity to teacher level within ~400 updates; std ratio ~1.0 selects the best checkpoint.
6 (retracted) "teacher right view collapses" — evaluation contamination.
7 GT-supervised geometry loss conflicts with DMD (3 runs); input-only anchor does not.
8 Geo4D's two views share 100% of weights (shallow-copy bug).
9 Training-free few-step teacher loses 47% diversity regardless of sampler.
10 53% of right-view GT pixels are transformed invalid pixels (0.418 -> 0.064).
11 Reported times included 2.9 s of evaluation scaffolding; pure inference is 21.5 s / 2.81 s (fp32).
12 Input anchor also improves the teacher (0.072 -> 0.060).

## 10. Status of the policy-level evaluation — NOT DONE
- Geo4D Table 2: success rates on 30 rollouts; Task 3 PlaceAppleFromBowlIntoBin: Geo4D 0.53 (Dreamitate 0.10, DP 0.00, DP3 0.00);
  Task 1 StoreCerealBoxUnderShelf 0.73; Task 2 PutSpatulaOnTableFromUtensilCrock 0.67; inference ~30 s per 10 frames on RTX 4090.
- Reproducing it needs the LBM simulator (now open-sourced as lbm_eval, Drake-based), a 6-DoF pose tracker (Geo4D Sec. 3.4, not released),
  gripper-openness inference, and the policy interface. None of this is in the released Geo4D code. We did NOT run it.
- A simulator-free proxy is implemented (bench_policy_proxy.py): region metrics on the gripper (label ids 29-35) and the apple (id 44) —
  AbsRel, PSNR, and 3D centroid error (cm) of the predicted vs GT pointmap inside the same mask, per frame 1..10. NOT RUN yet; describe as
  planned/appendix, no numbers.

## 11. Related work facts (cite only these; verify arXiv ids)
- Geo4D: Liu, Li, Cousineau, Feng, Burchfiel, Song. arXiv:2507.01099 (2025). Stanford / TRI.
- DMD: Yin et al., "One-step Diffusion with Distribution Matching Distillation", CVPR 2024, arXiv:2311.18828.
- DMD2: Yin et al., "Improved Distribution Matching Distillation for Fast Image Synthesis", NeurIPS 2024, arXiv:2405.14867.
- CausVid: Yin et al., "From Slow Bidirectional to Fast Autoregressive Video Diffusion Models", arXiv:2412.07772.
- Self-Forcing: Huang et al., "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion", arXiv:2506.08009.
- Causal Forcing: arXiv:2602.02214 (2026).
- Seaweed-APT: arXiv:2501.08316. MAGI-1, StreamDiT (streaming/AR video), TeaCache / MagCache (caching), SVDQuant (quantization).
- DreamDojo (10.8 FPS) and RoboWorld (15.3 FPS): robot world-model acceleration, no multi-view geometry evaluation.
- Perception-distortion tradeoff: Blau & Michaeli, CVPR 2018, arXiv:1711.06077.
- LPIPS: Zhang et al., CVPR 2018, arXiv:1801.03924.
- SVD: Blattmann et al., "Stable Video Diffusion", arXiv:2311.15127. EDM: Karras et al., NeurIPS 2022, arXiv:2206.00364.
- Consistency models: Song et al., ICML 2023, arXiv:2303.01469. Progressive distillation: Salimans & Ho, ICLR 2022, arXiv:2202.00512.
- Dreamitate: Liang et al., 2024 (video-generation visuomotor policy, arXiv:2406.16862). Diffusion Policy: Chi et al., RSS 2023, arXiv:2303.04137.
  DP3: Ze et al., RSS 2024, arXiv:2403.03954.
- Classifier-free guidance: Ho & Salimans, arXiv:2207.12598. Wilcoxon signed-rank test (Wilcoxon 1945).
- LBM eval simulator: Toyota Research Institute, "A Careful Examination of Large Behavior Models..." arXiv:2507.05331; lbm_eval GitHub.
- Reproduced on 5090 (context only): Self-Forcing Wan 1.3B 4-step 81 frames 832x480 in 9.8 s = 8.2 FPS; Causal-Forcing 3 stages all 0.45 s
  first chunk / 12 FPS. sm_120 needed xformers -> SDPA patch; flash-attn does not build.
