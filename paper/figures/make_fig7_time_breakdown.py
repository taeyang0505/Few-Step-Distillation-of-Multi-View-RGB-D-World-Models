"""Figure 7: pure inference time breakdown (UNet / conditioner / VAE decode).

All numbers are copied from FACTS.md Sections 6.1 and 6.2 (one RTX 5090, one
sample = 2 views x 10 frames at 256 x 320). Horizontal stacked bars with a
broken x axis so that the 21.49 s teacher bar and the ~1 s student bars are
readable on the same scale. Printed totals are the totals reported in FACTS.md.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig7_time_breakdown.png")

# ---------------------------------------------------------------- data (FACTS.md 6.1-6.2)
# label, UNet, conditioner, VAE decode, total as reported in FACTS.md
ROWS = [
    ("Teacher, Euler 25 steps (CFG)",          19.95, 0.60, 0.92, 21.49),
    ("Teacher, Euler 4 steps",                  3.22, 0.73, 0.89,  4.83),
    ("Student, 3 steps, fp32",                  1.19, 0.73, 0.88,  2.81),
    ("+ no unconditional branch",               1.18, 0.30, 0.90,  2.38),
    ("+ bf16 (UNet, conditioner, decoder)",     0.66, 0.30, 0.68,  1.64),
    ("+ torch.compile (UNet, decoder)",         0.55, 0.30, 0.33,  1.18),
    ("+ skip color decoder",                    0.55, 0.30, 0.16,  1.02),
]
COMPONENTS = ["UNet", "Conditioner", "VAE decode"]
# dataviz reference palette, categorical slots 1-3 (validated on white: adjacent CVD dE 9.2)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# ---------------------------------------------------------------- chrome
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#b8b6ae"
FS = 9  # pt, minimum font size everywhere

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK,
    "axes.edgecolor": AXIS,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

FIG_W, FIG_H, DPI = 8.0, 3.6, 200
XL = (0.0, 5.65)      # left panel, seconds
XR = (19.85, 22.85)   # right panel, seconds (same scale as left: width ratio = span ratio)
BAR_H = 0.54
GAP_LW = 1.0          # white gap between stacked segments (points)

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
gs = fig.add_gridspec(1, 2, width_ratios=[XL[1] - XL[0], XR[1] - XR[0]], wspace=0.035)
axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1], sharey=axL)

n = len(ROWS)
ys = [n - 1 - i for i in range(n)]  # first row at the top

for ax in (axL, axR):
    for (label, unet, cond, dec, total), y in zip(ROWS, ys):
        left = 0.0
        for val, col in zip((unet, cond, dec), COLORS):
            ax.barh(y, val, left=left, height=BAR_H, color=col,
                    edgecolor="white", linewidth=GAP_LW, zorder=3)
            left += val
    ax.set_ylim(-0.65, n - 1 + 0.65)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="x", length=3, color=AXIS)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    # thin separator between the teacher rows and the student rows
    ax.axhline(n - 1 - 1.5, color=GRID, linewidth=0.8, zorder=1)

axL.set_xlim(*XL)
axR.set_xlim(*XR)
axL.set_xticks([0, 1, 2, 3, 4, 5])
axR.set_xticks([20, 21, 22])
axR.tick_params(axis="y", length=0, labelleft=False)
axL.tick_params(axis="y", length=0)

# row labels: left-aligned text column; cumulative "+" rows indented
axL.set_yticks(ys)
axL.set_yticklabels(["" for _ in ys])
label_x = 0.012  # figure fraction
tr = blended_transform_factory(fig.transFigure, axL.transData)
label_artists = []
for (label, *_), y in zip(ROWS, ys):
    indent = 0.022 if label.startswith("+") else 0.0
    t = axL.text(label_x + indent, y, label, transform=tr, ha="left", va="center",
                 fontsize=FS, color=INK)
    label_artists.append(t)

# totals at the bar ends (values as reported in FACTS.md)
for (label, unet, cond, dec, total), y in zip(ROWS, ys):
    end = unet + cond + dec
    ax = axR if end > XL[1] else axL
    ax.text(end + 0.07, y, f"{total:.2f} s", ha="left", va="center",
            fontsize=FS, color=INK, zorder=4)

# broken-axis marks on the bottom spine
d = 0.5
kw = dict(marker=[(-1, -d), (1, d)], markersize=9, linestyle="none",
          color=AXIS, mec=AXIS, mew=0.9, clip_on=False, zorder=5)
axL.plot([1], [0], transform=axL.transAxes, **kw)
axR.plot([0], [0], transform=axR.transAxes, **kw)

# legend (three series -> legend required; order = stacking order)
handles = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(COLORS, COMPONENTS)]
axL.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.2,
           handleheight=0.9, borderaxespad=0.3, labelspacing=0.35)

# shared x label centred under both panels
fig.supxlabel("Pure inference time per sample (s)", fontsize=FS, color=INK, y=0.035)

# ---------------------------------------------------------------- layout
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
max_right_px = max(t.get_window_extent(renderer).x1 for t in label_artists)
left_frac = (max_right_px + 10) / (FIG_W * DPI)
fig.subplots_adjust(left=left_frac, right=0.985, top=0.975, bottom=0.165)

fig.savefig(OUT, dpi=DPI)
print("saved", OUT, "left margin fraction", round(left_frac, 3))
