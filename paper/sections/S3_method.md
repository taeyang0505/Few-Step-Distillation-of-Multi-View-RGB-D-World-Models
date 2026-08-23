# 3 Method

We take the released Geo4D model as a fixed teacher and ask how few denoising steps it can run in while keeping its multi-view geometry usable. Figure 1 gives an overview: a few-step sampler (Section 3.2), distillation (Section 3.3), a training-free depth calibration (Section 3.4), and the inference-path changes (Section 3.5).

## 3.1 Preliminaries: Geometry-Aware 4D Video Diffusion

Geo4D (Liu et al., 2025) is a latent video diffusion model built on Stable Video Diffusion (Blattmann et al., 2023). Given one RGB-D frame per view for two camera views, it generates 10 future RGB-D frames for both views. Each frame is a color image and a pointmap, and the pointmaps of both views are expressed in the frame of the left (reference) camera; no camera poses are given at inference. The latent of one sample has shape (2 views × 10 frames, 8 channels, 32 × 40) for 256 × 320 images, with 4 channels for the pointmap and 4 for color. We write $x$ for this latent and $c$ for the conditioning.

The denoiser follows the EDM parameterization (Karras et al., 2022). For a noisy latent $x_\sigma = x_0 + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$, the network $F_\theta$ is wrapped as

$$ D_\theta(x_\sigma; \sigma, c) = c_{\mathrm{skip}}(\sigma)\, x_\sigma + c_{\mathrm{out}}(\sigma)\, F_\theta\!\left(c_{\mathrm{in}}(\sigma)\, x_\sigma;\ c_{\mathrm{noise}}(\sigma),\ c\right), $$ (1)

an estimate of the clean latent $x_0$. The released sampler is a 25-step Euler integration of the probability-flow ODE from $\sigma_{\max} = 700$ with classifier-free guidance (Ho & Salimans, 2022) whose scale rises linearly from 1.0 to 2.5, so each step evaluates the UNet on a conditional and an unconditional batch (40 view-frames per sample).

One factual note on the released architecture: Geo4D builds the right-view UNet by shallow-copying the left one and deep-copying only its output blocks, but the shallow copy shares the module dictionary, so the new output blocks are written back into the original and the two views share all weights (all 795 output-block keys in the checkpoint are identical). The right view differs only through the spatial context it receives from the left view's hidden states and its own conditioning latent. Distilling the model therefore means distilling one UNet.

## 3.2 Few-Step Sampling by Re-noising

We use three noise levels, $\sigma \in \{700, 70.5, 2.3\}$, the teacher's four-step EDM schedule without the slot near $\sigma = 0$. At each level the network predicts $x_0$, and the sample is moved to the next level by adding fresh noise rather than by integrating the ODE:

$$ \hat{x}_0^{(i)} = D_\theta(x_{\sigma_i}; \sigma_i, c), \qquad x_{\sigma_{i+1}} = \hat{x}_0^{(i)} + \sigma_{i+1}\, \epsilon_{i+1}, \quad \epsilon_{i+1} \sim \mathcal{N}(0, I), $$ (2)

with output $\hat{x}_0^{(3)}$. This is the multi-step rule of consistency models (Song et al., 2023) and DMD2 (Yin et al., 2024b). A few-step Euler integration accumulates discretization error along the trajectory, whereas Equation (2) only asks the network for the quantity it was trained to predict. Equation (2) alone, however, does not make a few-step sampler: a denoiser trained with the standard objective returns the conditional mean of $x_0$ given $x_\sigma$, and with few steps the output is this blurred, seed-independent mean (Blau & Michaeli, 2018). Section 4.2 confirms this on the teacher with either sampler. The student therefore has to be trained so that its $x_0$ prediction at high $\sigma$ is a sample rather than a mean.

## 3.3 Distribution Matching Distillation for Multi-View RGB-D

We distill with distribution matching distillation (DMD; Yin et al., 2024a) in EDM sigma space. Three copies of the Geo4D UNet are involved: the student $G_\theta$, the real score, which is the frozen teacher evaluated with its classifier-free guidance, and the fake score $D_\phi$, a trainable copy of the teacher that is continually fit to the student's outputs with the denoising loss. For a student sample $\hat{x}_0$ re-noised to $x_\sigma = \hat{x}_0 + \sigma\epsilon$ at a random $\sigma$, the generator gradient is

$$ \nabla_\theta \mathcal{L}_{\mathrm{DMD}} = \mathbb{E}_{\sigma, \epsilon}\left[ w(\sigma) \left( D_\phi(x_\sigma; \sigma, c) - D_{\mathrm{teacher}}^{\mathrm{cfg}}(x_\sigma; \sigma, c) \right) \frac{\partial \hat{x}_0}{\partial \theta} \right], $$ (3)

where $w(\sigma)$ is the per-sample normalization of Yin et al. (2024a) and the difference of the two $x_0$ estimates equals the fake-minus-real score difference up to $\sigma^2$. No ground-truth video enters the objective. Following DMD2 we train on the student's own trajectory: each update draws $k \in \{0, 1, 2\}$, runs Equation (2) for $k$ steps without gradient, and applies Equation (3) to the $x_0$ prediction at step $k$ only, as in CausVid (Yin et al., 2025) and Self-Forcing (Huang et al., 2025).

Before DMD we tried to initialize the student by regressing it onto 284 (noise, final latent) pairs from the 25-step teacher, first with pseudo-trajectories $x_\sigma = x_0 + \sigma\epsilon$ (the loss fell from 0.45 to 0.22, but AbsRel became 3x worse and PSNR dropped 4.2 dB while CV-Chamfer improved by 31%, i.e. the two views agreed with each other and both were wrong), and then with the teacher's true intermediate states at $\sigma = 700, 70.5, 2.3$, where the loss terms at $\sigma = 700$ and $70.5$ plateaued at 0.40 and the student produced a uniform fog, identical across seeds. Mean-squared regression from noise to sample converges to the conditional mean even with the true trajectory (Finding 4; cf. Causal Forcing, Zhu et al., 2026, App. C.2), so we keep the second variant only as the DMD initialization.

DMD has no validation loss, so every 100 updates we generate two fixed conditioning pairs and compute

$$ \rho = \operatorname{std}(\hat{x}_0^{\mathrm{student}})\, /\, \operatorname{std}(x^{\mathrm{teacher}}), $$ (4)

the ratio of latent standard deviations; $\rho < 1$ indicates residual fog and $\rho > 1$ over-sharpening. In our run $\rho$ rose from 0.79 at initialization to 0.99 (update 1000), 1.06 (1600) and 1.11 (2000); we take the checkpoint with $\rho$ closest to 1, update 1600; update 2000, where $\rho$ overshoots, evaluates worse (Finding 5).

To fit one 32 GB GPU, all three UNets are kept in bf16 (9.2 GB), the optimizer is 8-bit AdamW, the conditioning tensors are precomputed and held on the CPU, and the conditioner and VAEs are moved off the GPU. At batch size 1 the peak is 24.4 GB and 2000 updates take 1 h.

## 3.4 Input-Anchored Depth Calibration

After DMD the student matches the teacher in sharpness and seed diversity, but its depth is globally too far by about 9% (median scale to ground truth 0.909 versus 0.979 for the teacher), and fitting one scale and offset per sample removes 97% of its AbsRel gap. The error is a global shift, not broken structure (Figure 5), so we correct it with one scalar per view. Geo4D receives the conditioning frame's pointmap as input, and the first predicted frame should coincide with it, so we set

$$ s = \operatorname{median}_{p \in \Omega}\ \frac{z_{\mathrm{cond}}(p)}{z_{\mathrm{pred},0}(p)}, \qquad \tilde{z}_{\mathrm{pred},t} = s \cdot z_{\mathrm{pred},t}\ \ \text{for all } t, $$ (5)

where $\Omega$ is the set of conditioning-frame pixels with valid depth. No ground-truth future frame is used; on the left view $s$ agrees with the oracle scale fitted to ground truth to three decimals.

The right view needs its own anchor. Its conditioning pointmap is in the right camera frame while the prediction is in the reference frame, so we transform it with $T = E_L^{-1} E_R$ from the dataset extrinsics before computing $s$ for that view (the transformed frame matches ground-truth frame 0 to AbsRel 0.001 and 0.006 on two checked samples). A scale-plus-offset variant fitted per view gave no further gain (Section 4.5). The same anchor also reduces the teacher's AbsRel (Finding 12), so it is a property of the input rather than a repair specific to the student.

## 3.5 Inference Path

The student is sampled with Equation (2) without classifier-free guidance, so the unconditional branch is dropped from the conditioner and the UNet batch; each UNet call processes 20 view-frames instead of 40. The released inference code ran the UNet in fp32 although training was in bf16; we run the UNet, conditioner, and VAE decoders under bf16 autocast, which changed AbsRel and LPIPS by less than 0.001 on 20 samples (Section 4.4). Optionally the UNet and decoders are compiled with torch.compile, and when only depth is needed for planning the color decoder is skipped. Table 3 reports the time of each variant.
