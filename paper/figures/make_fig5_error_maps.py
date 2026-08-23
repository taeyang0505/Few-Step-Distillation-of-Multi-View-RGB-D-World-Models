"""Figure 5: relative depth error maps (teacher vs. DMD student, no anchor), left and right view.

Source grids (repo results/qualitative/dmd6a_qual_lr/err_{left,right}.png) are 3200 x 768 px:
10 frames (t = 0..9, 320 px each) x 3 rows (256 px each) = teacher 25-step / student 3-step / student 1-step.
They were produced by code/geo4d/bench_student_qual.py: e = clip(|z_pred - z_gt| / z_gt, 0, 0.5) / 0.5 -> magma;
pixels with z_pred <= 0 or z_gt <= 0 painted gray (40, 40, 40). No input anchor is applied in that script.
Each tile has a small burned-in label box in its top 17 px; we crop those rows off uniformly.
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

SRC = ("/private/tmp/claude-501/-Users-hongtaeyang-Desktop-NAIS/4d37aec7-cae7-40a3-965b-4ed2a4ecc968/"
       "scratchpad/repo/results/qualitative/dmd6a_qual_lr")
OUT = ("/private/tmp/claude-501/-Users-hongtaeyang-Desktop-NAIS/4d37aec7-cae7-40a3-965b-4ed2a4ecc968/"
       "scratchpad/paper/figures/fig5_error_maps.png")

TILE_W, TILE_H = 320, 256
CROP_TOP = 18                      # rows 0..17 hold the burned-in "teacher25 t=0" / "t=k" label box
FRAMES = [0, 4, 9]                 # 0-indexed -> predicted frames 1, 5, 10
FRAME_LABELS = ["frame 1", "frame 5", "frame 10"]
ROW_LABELS = ["Teacher\n25 steps", "Student\n3 steps\n(no anchor)", "Student\n1 step\n(no anchor)"]
VIEWS = [("left", "Left view (reference camera)"), ("right", "Right view")]
GRAY = (40 / 255,) * 3

# ---- layout in inches (figure 12 x 4.85 in at 200 dpi = 2400 x 970 px) ----
FIG_W, FIG_H = 12.0, 4.85
LEFT = 1.10          # room for horizontal row labels
CB_X = 11.02         # colour bar x position
TILE_AREA_R = 10.88  # right edge of the right block
BLOCK_GAP = 0.30     # gap between the two view blocks
TILE_GAP = 0.06
ROW_GAP = 0.09
BOTTOM = 0.40        # room for the gray-pixel legend line

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def load_tiles(view):
    arr = np.asarray(Image.open(f"{SRC}/err_{view}.png").convert("RGB"))
    assert arr.shape == (3 * TILE_H, 10 * TILE_W, 3), arr.shape
    return [[arr[r * TILE_H + CROP_TOP:(r + 1) * TILE_H, t * TILE_W:(t + 1) * TILE_W]
             for t in FRAMES] for r in range(3)]


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
    ax_in = lambda x, y, w, h: fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])

    tile_w = (TILE_AREA_R - LEFT - BLOCK_GAP - 4 * TILE_GAP) / 6
    tile_h = tile_w * (TILE_H - CROP_TOP) / TILE_W
    rows_top = BOTTOM + 3 * tile_h + 2 * ROW_GAP

    for b, (view, heading) in enumerate(VIEWS):
        tiles = load_tiles(view)
        bx0 = LEFT + b * (3 * tile_w + 2 * TILE_GAP + BLOCK_GAP)
        bx1 = bx0 + 3 * tile_w + 2 * TILE_GAP
        for r in range(3):
            y = rows_top - (r + 1) * tile_h - r * ROW_GAP
            for j in range(3):
                x = bx0 + j * (tile_w + TILE_GAP)
                ax = ax_in(x, y, tile_w, tile_h)
                ax.imshow(tiles[r][j], interpolation="antialiased")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4); s.set_edgecolor("0.6")
                if r == 0:
                    ax.set_title(FRAME_LABELS[j], pad=3, fontsize=10)
                if b == 0 and j == 0:
                    ax.text(-0.05, 0.5, ROW_LABELS[r], transform=ax.transAxes,
                            ha="right", va="center", fontsize=10, linespacing=1.25)
        # block heading with a thin rule, just above the frame labels
        y_rule = rows_top + 0.30
        fig.add_artist(plt.Line2D([bx0 / FIG_W, bx1 / FIG_W], [y_rule / FIG_H] * 2, color="0.3", lw=0.8))
        fig.text((bx0 + bx1) / 2 / FIG_W, (y_rule + 0.05) / FIG_H, heading,
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    # colour bar: magma, 0 .. 50 % relative depth error (the source maps clip at 50 %)
    cax = ax_in(CB_X, BOTTOM, 0.15, rows_top - BOTTOM)
    sm = cm.ScalarMappable(norm=Normalize(0, 50), cmap="magma")
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 10, 20, 30, 40, 50])
    cb.set_ticklabels(["0", "10", "20", "30", "40", "≥50"], fontsize=9.5)
    cb.ax.tick_params(length=2.5, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.set_label("relative depth error  $|\\hat{z}-z|/z$  (%)", fontsize=10)

    # legend entry for masked pixels
    fig.add_artist(Rectangle((LEFT / FIG_W, 0.10 / FIG_H), 0.13 / FIG_W, 0.13 / FIG_H,
                             transform=fig.transFigure, facecolor=GRAY, edgecolor="0.4", lw=0.5))
    fig.text((LEFT + 0.20) / FIG_W, 0.165 / FIG_H,
             "gray: pixels without valid depth (masked)",
             ha="left", va="center", fontsize=9.5)

    fig.savefig(OUT, dpi=200, facecolor="white")
    im = Image.open(OUT).convert("RGB")   # flatten RGBA -> RGB on the white background
    im.save(OUT, dpi=(200, 200))
    print("saved", OUT, im.size, "tile", round(tile_w, 3), "x", round(tile_h, 3), "in")


if __name__ == "__main__":
    main()
