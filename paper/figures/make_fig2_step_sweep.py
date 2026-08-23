"""Figure 2: training-free step reduction of the Geo4D teacher (Euler sampler).

Every number is copied verbatim from the two result files below; nothing is
derived except the percentage / dB deltas printed in the annotations, which are
the deltas stated in the same files (and in FACTS.md, Sec. 4).

repo/results/quantitative/full_sweep_results.txt (10 samples, old mask):
  steps  time(s)  PSNR   CV-Chamfer
    25    25.8    19.75   0.1758
    16    18.1    19.76   0.1754
     8    11.4    20.06   0.1692
     4     7.9    20.35   0.1796
     2     6.4    20.66   0.1919
     1     5.4    20.64   0.1943
  relative to 25 steps at 1 step: PSNR +4.5 %, CV-Chamfer +10.5 %

repo/results/quantitative/blur_test_results.txt (2 samples x 4 seeds):
  steps  LPIPS   sharpness  seed diversity
    25   0.1222  0.013864   0.01878
    16   0.1228  0.013644   0.01735
     8   0.1269  0.013096   0.01406
     4   0.1361  0.012621   0.01050
     2   0.1473  0.012154   0.00430
     1   0.1474  0.012134   0.00429
  relative to 25 steps at 1 step: LPIPS +20.7 %, sharpness -12.5 %, diversity -77.1 %

Time per sample is not plotted: those timings are the log_images path, which
includes 2.9 s of evaluation-only work (FACTS.md, Finding 11).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_step_sweep.png")

# ---------------------------------------------------------------- data
steps = np.array([25, 16, 8, 4, 2, 1])

# full_sweep_results.txt
psnr = np.array([19.75, 19.76, 20.06, 20.35, 20.66, 20.64])
cv_chamfer = np.array([0.1758, 0.1754, 0.1692, 0.1796, 0.1919, 0.1943])

# blur_test_results.txt
lpips = np.array([0.1222, 0.1228, 0.1269, 0.1361, 0.1473, 0.1474])
sharpness = np.array([0.013864, 0.013644, 0.013096, 0.012621, 0.012154, 0.012134])
diversity = np.array([0.01878, 0.01735, 0.01406, 0.01050, 0.00430, 0.00429])

# ---------------------------------------------------------------- style
# Okabe-Ito blue for the single measured series (same hue as Figure 8),
# validated on white: contrast >= 3:1. The reference line is secondary ink.
C_SERIES = "#0072B2"
C_REF = "#52514e"
INK = "#0b0b0b"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FS = 9  # minimum font size (pt) at 12 x 3.2 in

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.edgecolor": AXIS,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
})

fig = plt.figure(figsize=(12.0, 3.2), dpi=200, facecolor="white")
# column 0 holds two stacked axes (PSNR over CV-Chamfer); columns 1-3 span
# both rows. Separate axes instead of a twin y-axis.
gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.12,
                      wspace=0.42, left=0.055, right=0.985, top=0.84, bottom=0.17)
ax_psnr = fig.add_subplot(gs[0, 0])
ax_cv = fig.add_subplot(gs[1, 0], sharex=ax_psnr)
ax_lpips = fig.add_subplot(gs[:, 1])
ax_sharp = fig.add_subplot(gs[:, 2])
ax_div = fig.add_subplot(gs[:, 3])


def style(ax, xlabel=True):
    ax.set_facecolor("white")
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.78, 32)
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", length=3, color=AXIS)
    if xlabel:
        ax.set_xlabel("sampling steps")


def draw(ax, y, ylabel, ylim, yticks, fmt):
    # reference: 25-step teacher value
    ax.axhline(y[0], color=C_REF, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.plot(steps, y, color=C_SERIES, lw=2.0, marker="o", ms=5.5,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3,
            solid_joinstyle="round")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))


# (a) PSNR (higher is better) over CV-Chamfer (lower is better)
draw(ax_psnr, psnr, "PSNR (dB) ↑", (19.5, 21.0), [19.5, 20.0, 20.5, 21.0], "%.1f")
style(ax_psnr, xlabel=False)
plt.setp(ax_psnr.get_xticklabels(), visible=False)
ax_psnr.tick_params(axis="x", length=0)

draw(ax_cv, cv_chamfer, "CV-Chamfer ↓", (0.160, 0.200), [0.16, 0.18, 0.20], "%.2f")
style(ax_cv)

# (b) LPIPS, (c) Laplacian sharpness, (d) seed diversity
draw(ax_lpips, lpips, "LPIPS ↓", (0.118, 0.152), [0.12, 0.13, 0.14, 0.15], "%.2f")
style(ax_lpips)
draw(ax_sharp, sharpness, "Laplacian sharpness ↑", (0.0118, 0.0142),
     [0.012, 0.013, 0.014], "%.3f")
style(ax_sharp)
draw(ax_div, diversity, "seed diversity (std over 4 seeds) ↑", (0.0, 0.021),
     [0.000, 0.005, 0.010, 0.015, 0.020], "%.3f")
style(ax_div)

# ---------------------------------------------------------------- annotations
# one endpoint label per panel: change from 25 steps to 1 step
# (values 25 steps -> 1 step); each label is placed in empty space of its panel
ann = dict(fontsize=FS, color=INK, va="center")
ax_psnr.text(1.0, 20.1, "+0.9 dB", ha="left", **ann)
ax_cv.text(1.0, 0.184, "+10.5%", ha="left", **ann)
ax_lpips.text(28, 0.1445, "+20.7%\n0.1222 → 0.1474", ha="right", **ann)
ax_sharp.text(1.0, 0.01335, "−12.5%", ha="left", **ann)
ax_div.text(28, 0.0045, "−77%\n0.0188 → 0.0043", ha="right", **ann)

# panel letters (outside the axes, above the y-axis)
for ax, letter in ((ax_psnr, "(a)"), (ax_lpips, "(b)"), (ax_sharp, "(c)"),
                   (ax_div, "(d)")):
    ax.text(-0.19, 1.06, letter, transform=ax.transAxes, fontsize=FS + 1,
            fontweight="bold", ha="left", va="bottom")

# single figure-level legend (the same two marks appear in every panel)
handles = [
    Line2D([0], [0], color=C_SERIES, lw=2.0, marker="o", ms=5.5,
           markeredgecolor="white", markeredgewidth=1.2,
           label="Geo4D teacher, Euler sampler, fewer steps"),
    Line2D([0], [0], color=C_REF, lw=0.9, ls=(0, (4, 3)),
           label="25-step teacher (reference)"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.52, 1.0), handlelength=2.2, columnspacing=2.0)

fig.savefig(OUT, dpi=200, facecolor="white")
print("saved", OUT)
