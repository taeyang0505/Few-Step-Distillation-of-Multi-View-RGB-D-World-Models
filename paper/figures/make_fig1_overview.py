"""Figure 1 (overview): Geo4D teacher vs. our 3-step DMD student.

All numbers come from FACTS.md:
  teacher: 25 Euler steps, sigma_max = 700, CFG (batch 2x), fp32, 2 views x 10 RGB-D frames, 21.8 s
  student: 3 re-noising steps at sigma = 700 -> 70.5 -> 2.3, x0 prediction then re-noise, no CFG, bf16,
           per-view input anchor (scale from conditioning pointmap; right view via inv(E_L) E_R),
           1.64 s (13.3x), AbsRel +0.015, LPIPS +0.018, sharpness and seed diversity preserved.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = ("/private/tmp/claude-501/-Users-hongtaeyang-Desktop-NAIS/"
       "4d37aec7-cae7-40a3-965b-4ed2a4ecc968/scratchpad/paper/figures/fig1_overview.png")

# ---------------------------------------------------------------- palette
INK = "#222222"        # text and edges
GRAY_EDGE = "#555555"  # teacher box edges
GRAY_FILL = "#EFEFEF"  # teacher box fill
GRAY_MID = "#9A9A9A"   # frame thumbnails
ACCENT = "#0072B2"     # Okabe-Ito blue (colorblind-safe); student-only parts
ACCENT_FILL = "#DCEBF5"

plt.rcParams.update({
    "font.family": ["Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "text.color": INK,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
})

W, H = 12.0, 4.5
fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


# ---------------------------------------------------------------- helpers
def box(x, y, w, h, text, fc=GRAY_FILL, ec=GRAY_EDGE, fs=10, weight="normal", lw=1.1, tc=INK):
    """Rounded box centred at (x, y); size in inches."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.07",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight,
            color=tc, zorder=3, linespacing=1.25)


def arrow(x1, y1, x2, y2, color=INK, lw=1.1, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                shrinkA=0, shrinkB=0, mutation_scale=11,
                                linestyle=ls), zorder=2)


def polyline_arrow(pts, color=INK, lw=1.0):
    """Orthogonal route through pts, arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=2, solid_capstyle="round")
    arrow(xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw)


def loop(x, y, color=INK, label="", lw=1.1):
    """Small circular-arrow marker to the right of a box edge with a label."""
    ax.annotate("", xy=(x, y + 0.12), xytext=(x, y - 0.12),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                connectionstyle="arc3,rad=-1.6", mutation_scale=9), zorder=3)
    ax.text(x + 0.25, y, label, ha="left", va="center", fontsize=10, weight="bold", color=color)


def view_cards(x, yc, ec=GRAY_EDGE):
    """Two stacked RGB-D input views (left / right camera)."""
    w, h, gap = 0.80, 0.30, 0.06
    for i, lab in enumerate(["view L", "view R"]):
        y = yc + (h + gap) / 2 - i * (h + gap)
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec=ec, lw=1.0, zorder=2))
        ax.add_patch(Rectangle((x - w / 2 + 0.04, y - h / 2 + 0.05), 0.16, h - 0.10,
                               fc=GRAY_MID, ec="none", zorder=3))
        ax.text(x + 0.10, y, lab, ha="center", va="center", fontsize=9, zorder=3)
    ax.text(x, yc - (h + gap) - 0.02, "2 RGB-D input\nframes (t = 0)", ha="center", va="top",
            fontsize=9, linespacing=1.2)
    return w


def frame_stack(x, yc, ec=GRAY_EDGE, n=4):
    """Output: two stacks of frames (one per view)."""
    w, h = 0.44, 0.24
    for row, lab in enumerate(["L", "R"]):
        y0 = yc + 0.22 - row * 0.44
        for i in range(n - 1, -1, -1):
            off = i * 0.055
            ax.add_patch(Rectangle((x - w / 2 + off, y0 - h / 2 - off * 0.6), w, h,
                                   fc="white" if i else GRAY_FILL, ec=ec, lw=0.8,
                                   zorder=2 + (n - i) * 0.01))
        ax.text(x - w / 2 - 0.07, y0 - 0.05, lab, ha="right", va="center", fontsize=9)


def bracket(x1, x2, y, color=INK, tick=0.08):
    ax.plot([x1, x1, x2, x2], [y + tick, y, y, y + tick], color=color, lw=1.0, zorder=2)


# ---------------------------------------------------------------- geometry
Y_MAIN = 2.72     # main pipeline row
BH = 0.70         # box height
Y_NOISE = 3.62    # noise node
Y_SCHED = 1.80    # sampling schedule row
Y_BRK = 1.22      # time bracket
Y_TIME = 0.98
Y_RES = 0.52
w_cond, w_unet, w_vae = 0.82, 0.95, 0.82

# ================================================================ (a) teacher
ax.text(0.15, 4.25, "(a) Geo4D teacher (Liu et al., 2025)", fontsize=11, weight="bold", va="center")

x_in, x_cond, x_unet, x_vae, x_out = 0.62, 1.72, 2.86, 3.95, 4.90

view_cards(x_in, Y_MAIN)
box(x_cond, Y_MAIN, w_cond, BH, "conditioner")
box(x_unet, Y_MAIN, w_unet, BH, "UNet\n(fp32)", weight="bold")
box(x_vae, Y_MAIN, w_vae, BH, "VAE\ndecode")
frame_stack(x_out, Y_MAIN)
ax.text(x_out + 0.06, Y_MAIN + 0.40, "2 views x 10\nRGB-D frames", ha="center", va="bottom", fontsize=9)

arrow(x_in + 0.40, Y_MAIN, x_cond - w_cond / 2, Y_MAIN)
arrow(x_cond + w_cond / 2, Y_MAIN, x_unet - w_unet / 2, Y_MAIN)
arrow(x_unet + w_unet / 2, Y_MAIN, x_vae - w_vae / 2, Y_MAIN)
arrow(x_vae + w_vae / 2, Y_MAIN, x_out - 0.30, Y_MAIN)

box(x_unet, Y_NOISE, 0.95, 0.32, "noise (seed)", fs=9, fc="white")
arrow(x_unet, Y_NOISE - 0.16, x_unet, Y_MAIN + BH / 2)
loop(x_unet + w_unet / 2 - 0.06, Y_MAIN + BH / 2 + 0.18, label="x25")

# schedule: 25 Euler steps from sigma = 700 to 0, with CFG
sx1, sx2 = 1.50, 4.40
arrow(x_unet, Y_MAIN - BH / 2, x_unet, Y_SCHED + 0.10, style="-", ls=":", color=GRAY_EDGE, lw=0.9)
ax.plot([sx1, sx2], [Y_SCHED, Y_SCHED], color=INK, lw=1.0, zorder=2)
for i in range(26):
    xt = sx1 + (sx2 - sx1) * i / 25
    ax.plot([xt, xt], [Y_SCHED - 0.05, Y_SCHED + 0.05], color=INK, lw=0.8, zorder=2)
ax.text(sx1 - 0.06, Y_SCHED, "σ = 700", ha="right", va="center", fontsize=9)
ax.text(sx2 + 0.06, Y_SCHED, "0", ha="left", va="center", fontsize=9)
ax.text((sx1 + sx2) / 2, Y_SCHED - 0.13, "25 Euler steps, CFG guider (batch 2x)",
        ha="center", va="top", fontsize=9)

bracket(x_cond - w_cond / 2, x_vae + w_vae / 2, Y_BRK)
ax.text((x_cond - w_cond / 2 + x_vae + w_vae / 2) / 2, Y_TIME, "21.8 s per prediction",
        ha="center", va="center", fontsize=11, weight="bold")

# ================================================================ divider
ax.plot([5.50, 5.50], [0.35, 4.35], color="#BBBBBB", lw=0.9, ls=(0, (4, 3)), zorder=1)

# ================================================================ (b) student
ax.text(5.72, 4.25, "(b) Our DMD student: 3 re-noising steps + per-view input anchor",
        fontsize=11, weight="bold", va="center")

x_in2, x_cond2, x_unet2, x_vae2, x_anc2, x_out2 = 6.18, 7.26, 8.36, 9.42, 10.48, 11.42
w_anc = 1.04

view_cards(x_in2, Y_MAIN)
box(x_cond2, Y_MAIN, w_cond, BH, "conditioner")
box(x_unet2, Y_MAIN, w_unet, BH, "UNet\n(bf16)", fc=ACCENT_FILL, ec=ACCENT, weight="bold", lw=1.4)
box(x_vae2, Y_MAIN, w_vae, BH, "VAE\ndecode")
box(x_anc2, Y_MAIN, w_anc, BH, "per-view\ninput anchor", fc=ACCENT_FILL, ec=ACCENT, lw=1.4)
frame_stack(x_out2, Y_MAIN)
ax.text(x_out2 + 0.06, Y_MAIN + 0.40, "2 views x 10\nRGB-D frames", ha="center", va="bottom", fontsize=9)

arrow(x_in2 + 0.40, Y_MAIN, x_cond2 - w_cond / 2, Y_MAIN)
arrow(x_cond2 + w_cond / 2, Y_MAIN, x_unet2 - w_unet / 2, Y_MAIN)
arrow(x_unet2 + w_unet / 2, Y_MAIN, x_vae2 - w_vae / 2, Y_MAIN)
arrow(x_vae2 + w_vae / 2, Y_MAIN, x_anc2 - w_anc / 2, Y_MAIN)
arrow(x_anc2 + w_anc / 2, Y_MAIN, x_out2 - 0.30, Y_MAIN)

box(x_unet2, Y_NOISE, 0.95, 0.32, "noise (seed)", fs=9, fc="white")
arrow(x_unet2, Y_NOISE - 0.16, x_unet2, Y_MAIN + BH / 2)
loop(x_unet2 + w_unet / 2 - 0.06, Y_MAIN + BH / 2 + 0.18, color=ACCENT, label="x3", lw=1.2)

# conditioning pointmap feeds the anchor (routed above the pipeline; no ground truth needed)
y_route = 3.98
polyline_arrow([(x_in2, Y_MAIN + 0.36), (x_in2, y_route), (x_anc2, y_route), (x_anc2, Y_MAIN + BH / 2)],
               color=ACCENT, lw=1.0)
ax.text((x_unet2 + 0.55 + x_anc2) / 2, y_route - 0.05, "input pointmap (no GT)",
        ha="center", va="top", fontsize=9, color=ACCENT)

# schedule: three sigmas, x0 prediction then re-noise
sig_x = [7.28, 8.36, 9.44]
sig_lab = ["σ = 700", "σ = 70.5", "σ = 2.3"]
nw, nh = 0.66, 0.28
arrow(x_unet2, Y_MAIN - BH / 2, x_unet2, Y_SCHED + nh / 2, style="-", ls=":", color=GRAY_EDGE, lw=0.9)
for xv, lab in zip(sig_x, sig_lab):
    box(xv, Y_SCHED, nw, nh, lab, fc="white", ec=ACCENT, fs=9, lw=1.1)
for xa, xb in zip(sig_x[:-1], sig_x[1:]):
    arrow(xa + nw / 2, Y_SCHED, xb - nw / 2, Y_SCHED, color=ACCENT, lw=1.0)
    ax.text((xa + xb) / 2, Y_SCHED + 0.17, r"$\hat{x}_0$, re-noise", ha="center", va="bottom",
            fontsize=9, color=ACCENT)
ax.text(sig_x[1], Y_SCHED - nh / 2 - 0.05,
        "predict $\\hat{x}_0$, then re-noise to the next σ\nno CFG (batch 1x), bf16",
        ha="center", va="top", fontsize=9, linespacing=1.2)

# anchor detail under the anchor box
ax.text(x_anc2 + 0.22, Y_SCHED + 0.30,
        "depth scale = median ratio\nof predicted frame 0 to\ninput pointmap (no GT);\nright view via inv($E_L$)$\,E_R$",
        ha="center", va="top", fontsize=9, linespacing=1.2)

bracket(x_cond2 - w_cond / 2, x_anc2 + w_anc / 2, Y_BRK, color=ACCENT)
ax.text((x_cond2 - w_cond / 2 + x_anc2 + w_anc / 2) / 2, Y_TIME,
        "1.64 s per prediction (13.3x faster)", ha="center", va="center",
        fontsize=11, weight="bold", color=ACCENT)
ax.text((x_cond2 - w_cond / 2 + x_anc2 + w_anc / 2) / 2, Y_RES,
        "sharpness and seed diversity preserved;  AbsRel +0.015,  LPIPS +0.018  (vs. teacher)",
        ha="center", va="center", fontsize=9.5)

fig.savefig(OUT, dpi=200, facecolor="white")

# flatten to opaque RGB on white, keep the 200 dpi metadata
from PIL import Image
im = Image.open(OUT).convert("RGBA")
bg = Image.new("RGB", im.size, (255, 255, 255))
bg.paste(im, mask=im.split()[3])
bg.save(OUT, dpi=(200, 200))
print("saved", OUT, bg.size)
