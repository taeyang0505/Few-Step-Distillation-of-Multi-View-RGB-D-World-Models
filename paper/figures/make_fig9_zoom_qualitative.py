"""Figure 9: native-resolution zoom comparison — is the arm blurred by our distillation, or by Geo4D itself?

Source: repo/results/qualitative/dmd6a_qual_lr/rgb_left.png (3200 x 1024 = 10 frames x 320 px, 4 rows x 256 px;
rows = GT / teacher 25 steps / DMD student 3 steps / DMD student 1 step, left view, predicted frame index 7).
Crops are taken at native resolution and enlarged with nearest-neighbour so that no interpolation is added:
what looks smooth here is smooth in the model output, not in the figure.

Panels: full frame | moving region (left arm + gripper over the bowl) | static region (right gripper).
Message: the teacher already loses detail relative to the ground truth; the 3-step student matches the teacher;
only the 1-step student dissolves the moving arm, and even there the static region stays sharp.
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SRC = ("/private/tmp/claude-501/-Users-hongtaeyang-Desktop-NAIS/4d37aec7-cae7-40a3-965b-4ed2a4ecc968/"
       "scratchpad/repo/results/qualitative/dmd6a_qual_lr/rgb_left.png")
OUT = ("/private/tmp/claude-501/-Users-hongtaeyang-Desktop-NAIS/4d37aec7-cae7-40a3-965b-4ed2a4ecc968/"
       "scratchpad/paper/figures/fig9_zoom_qualitative.png")

TILE_W, TILE_H = 320, 256
CROP_TOP = 18                       # burned-in "t=7" label box
FRAME = 7
ROIS = [((55, 25, 175, 125), "moving: left arm + gripper"),
        ((185, 70, 285, 170), "static: right gripper")]
# sharpness = variance of the Laplacian (Table 1, 20 samples); the ground-truth value comes from the
# blur-hypothesis run (2 samples x 4 seeds), the only run in which ground-truth sharpness was measured.
ROWS = [("Ground truth", "whole 0.0197", "moving 1461   static 1326"),
        ("Teacher, 25 steps", "whole 0.0134  (−32% vs GT)", "moving 692   static 779"),
        ("Student, 3 steps (ours)", "whole 0.0136  (+2% vs teacher)", "moving 413 (−40%)   static 807 (+4%)"),
        ("Student, 1 step (rejected)", "whole 0.0107  (−20% vs teacher)", "moving 117 (−83%)   static 631 (−19%)")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def main():
    grid = np.asarray(Image.open(SRC).convert("RGB"))
    assert grid.shape == (4 * TILE_H, 10 * TILE_W, 3), grid.shape
    tiles = [grid[r * TILE_H:(r + 1) * TILE_H, FRAME * TILE_W:(FRAME + 1) * TILE_W] for r in range(4)]

    n_col = 1 + len(ROIS)
    fig_w, fig_h = 10.6, 8.4
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    left, top, gap_x, gap_y = 2.55, 0.42, 0.07, 0.30
    label_h = 0.20
    cell_w = (fig_w - left - 0.12 - (n_col - 1) * gap_x) / n_col
    cell_h = (fig_h - top - 0.30 - 4 * gap_y) / 4

    for r, (name, sharp, roi) in enumerate(ROWS):
        y = fig_h - top - (r + 1) * cell_h - r * gap_y
        fig.text(left / fig_w - 0.02, (y + cell_h * 0.62) / fig_h, name, ha="right", va="center",
                 fontsize=10, fontweight="bold" if r == 2 else "normal")
        fig.text(left / fig_w - 0.02, (y + cell_h * 0.40) / fig_h, sharp, ha="right",
                 va="center", fontsize=8, color="#444444")
        fig.text(left / fig_w - 0.02, (y + cell_h * 0.24) / fig_h, roi, ha="right",
                 va="center", fontsize=8, color="#b3541e")
        for c in range(n_col):
            x = left + c * (cell_w + gap_x)
            ax = fig.add_axes([x / fig_w, y / fig_h, cell_w / fig_w, cell_h / fig_h])
            if c == 0:
                img = tiles[r][CROP_TOP:]
                for (x0, y0, x1, y1), _ in ROIS:                       # mark the crop regions
                    ax.add_patch(Rectangle((x0, y0 - CROP_TOP), x1 - x0, y1 - y0,
                                           fill=False, edgecolor="#e8b100", lw=1.1))
            else:
                (x0, y0, x1, y1), _ = ROIS[c - 1]
                img = tiles[r][y0:y1, x0:x1]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.6)
            if r == 0:
                title = "full frame (predicted frame 8)" if c == 0 else ROIS[c - 1][1]
                ax.set_title(title, fontsize=9, pad=4)

    fig.text(0.5, 0.022,
             "Crops are native-resolution pixels enlarged without interpolation; sharpness is the variance of the Laplacian, averaged "
             "over the 10 predicted frames.\nWhole-image sharpness is background-dominated and reports parity (+2%), but the two regions "
             "move in opposite directions: the 3-step student\nmatches the teacher on the static gripper (+4%) and loses 40% on the "
             "moving arm. Single sample; region-wise numbers are preliminary.",
             ha="center", va="bottom", fontsize=8.5, color="#333333")
    fig.savefig(OUT, dpi=200)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
