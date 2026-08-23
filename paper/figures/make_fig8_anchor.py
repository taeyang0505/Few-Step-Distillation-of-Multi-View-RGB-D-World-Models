"""Figure 8: input-anchor ablation, grouped bars of AbsRel (left / right view).

Source of every number: repo/results/quantitative/eval_6x_maskfix.txt
(10 samples, data seed 1234, fake-pixel mask), columns AbsRel L / AbsRel R.
  T25  0.0803 / 0.0644   teacher, 25 Euler steps
  S3   0.1459 / 0.2047   DMD student, 3 steps, no anchor
  S3b  0.0861 / 0.0859   student + per-view input anchor (config b)
  S3c  0.0894 / 0.0805   student + robust affine anchor (config c)
  T25c 0.0679 / 0.0517   teacher + per-view input anchor
Values are rounded to three decimals in the figure, as in FACTS.md.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig8_anchor.png")

configs = [
    ("Teacher\n25 steps", 0.080, 0.064),
    ("Student\n3 steps, raw", 0.146, 0.205),
    ("Student\n+ per-view anchor", 0.086, 0.086),
    ("Student + robust\naffine anchor", 0.089, 0.081),
    ("Teacher\n+ per-view anchor", 0.068, 0.052),
]
labels = [c[0] for c in configs]
left = np.array([c[1] for c in configs])
right = np.array([c[2] for c in configs])

# Okabe-Ito blue / vermillion: validated colorblind-safe on white
# (CVD dE 21.9 protan, normal-vision dE 31.2, both >= 3:1 contrast).
C_LEFT = "#0072B2"
C_RIGHT = "#D55E00"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FS = 9  # minimum font size in points at 7 x 3.4 in

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

fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200, facecolor="white")
ax.set_facecolor("white")

x = np.arange(len(configs))
w = 0.36
gap = 0.02  # small surface gap between the two bars of a group

bars_l = ax.bar(x - w / 2 - gap / 2, left, width=w, color=C_LEFT,
                label="Left view", zorder=3, linewidth=0)
bars_r = ax.bar(x + w / 2 + gap / 2, right, width=w, color=C_RIGHT,
                label="Right view", zorder=3, linewidth=0)

# direct value labels in ink (text never wears the series color)
for b, v in list(zip(bars_l, left)) + list(zip(bars_r, right)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
            ha="center", va="bottom", fontsize=FS, color=INK, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

# reference line: teacher 25-step AbsRel, mean of the two views is not in the
# facts sheet, so draw the per-view teacher values as thin dashed guides.
ax.axhline(left[0], color=C_LEFT, lw=0.8, ls=(0, (4, 3)), alpha=0.55, zorder=2)
ax.axhline(right[0], color=C_RIGHT, lw=0.8, ls=(0, (4, 3)), alpha=0.55, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("AbsRel ↓  (fake-pixel mask, 10 samples)")
ax.set_ylim(0, 0.235)
ax.set_yticks(np.arange(0, 0.21, 0.05))
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color(AXIS)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(axis="both", length=3, color=AXIS)
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.6, len(configs) - 0.4)

leg = ax.legend(loc="upper right", frameon=False, ncol=2,
                handlelength=1.2, handleheight=0.9, columnspacing=1.2)

fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=200, facecolor="white")
print("saved", OUT)
