"""Figure 6: speed-quality trade-off (1x3 panels).

All numbers are copied from FACTS.md Section 7 (main table, 20 samples = 40
view-samples, fake pixels excluded; seed diversity = 3 samples x 4 seeds).
x = pure inference time (s, log scale); y = AbsRel (a), LPIPS (b), seed
diversity (c).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "fig6_speed_quality.png")

# ---------------------------------------------------------------------------
# Data (FACTS.md Section 7, main table)
#   key: (pure inference time [s], AbsRel, LPIPS, seed diversity)
# ---------------------------------------------------------------------------
DATA = {
    "T25": (21.49, 0.066, 0.118, 0.0227),  # Geo4D teacher, Euler, 25 steps
    "T4":  (4.84,  0.064, 0.132, 0.0131),  # teacher, Euler, 4 steps
    "T3r": (2.81,  0.066, 0.132, 0.0120),  # teacher, re-noising, 3 steps
    "S3":  (1.64,  0.082, 0.136, 0.0224),  # DMD student, 3 steps + per-view anchor, bf16
    "S1":  (1.65,  0.116, 0.177, 0.0117),  # DMD student, 1 step (fp32 time)
}

# Validated colorblind-safe categorical set (blue, aqua, violet, orange) plus a
# neutral dark gray for the fifth series, which is drawn hollow.
STYLE = {
    "T25": dict(color="#2a78d6", marker="o", ms=7.5, fill=True),
    "T4":  dict(color="#1baf7a", marker="s", ms=7.0, fill=True),
    "T3r": dict(color="#4a3aa7", marker="D", ms=6.5, fill=True),
    "S3":  dict(color="#eb6834", marker="*", ms=15.0, fill=True),
    "S1":  dict(color="#52514e", marker="v", ms=8.0, fill=False),
}

LEGEND_NAMES = {
    "T25": "T25: Geo4D teacher, 25 Euler steps",
    "T4":  "T4: teacher, 4 Euler steps",
    "T3r": "T3r: teacher, 3 re-noising steps",
    "S3":  "S3: DMD student, 3 steps + per-view anchor, bf16 (ours)",
    "S1":  "S1: DMD student, 1 step (fp32 time)",
}

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BUDGET_S = 2.0

# Per-panel label offsets (points) and alignment, tuned to avoid collisions.
#   (dx, dy, ha, va)
OFFSETS = {
    # panel (a) AbsRel
    0: {
        "T25": (0, 8, "center", "bottom"),
        "T4":  (0, -8, "center", "top"),
        "T3r": (0, 8, "center", "bottom"),
        "S3":  (-11, 0, "right", "center"),
        "S1":  (-8, 0, "right", "center"),
    },
    # panel (b) LPIPS
    1: {
        "T25": (0, 8, "center", "bottom"),
        "T4":  (0, -8, "center", "top"),
        "T3r": (0, -8, "center", "top"),
        "S3":  (-11, 0, "right", "center"),
        "S1":  (-8, 0, "right", "center"),
    },
    # panel (c) seed diversity
    2: {
        "T25": (0, 8, "center", "bottom"),
        "T4":  (0, 8, "center", "bottom"),
        "T3r": (0, 8, "center", "bottom"),
        "S3":  (-9, -7, "right", "top"),   # below the teacher reference line
        "S1":  (-8, 0, "right", "center"),
    },
}

PANELS = [
    # (column index in DATA tuple, y label, y limits)
    (1, "AbsRel (depth, lower is better)", (0.050, 0.135)),
    (2, "LPIPS (lower is better)", (0.100, 0.195)),
    (3, "seed diversity (std over 4 seeds)", (0.0080, 0.0265)),
]
LETTERS = ["(a)", "(b)", "(c)"]


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    for i, (col, ylabel, ylim) in enumerate(PANELS):
        ax = axes[i]
        teacher_y = DATA["T25"][col]

        # Teacher reference level (dashed line in the teacher color).
        ax.axhline(teacher_y, color=STYLE["T25"]["color"], lw=1.0,
                   ls=(0, (5, 3)), alpha=0.7, zorder=1)

        # Quasi-static budget (2 s).
        ax.axvline(BUDGET_S, color=INK, lw=1.0, ls=(0, (4, 3)), alpha=0.8,
                   zorder=1)
        ax.text(BUDGET_S * 1.06, ylim[1] - 0.02 * (ylim[1] - ylim[0]),
                "quasi-static budget (2 s)", ha="left", va="top",
                fontsize=9, color=INK, zorder=4)

        # Points.
        for key, vals in DATA.items():
            st = STYLE[key]
            x, y = vals[0], vals[col]
            emph = key == "S3"
            if st["fill"]:
                ax.plot(x, y, linestyle="none", marker=st["marker"],
                        ms=st["ms"], mfc=st["color"],
                        mec=INK if emph else "white",
                        mew=0.9 if emph else 1.2,
                        zorder=6 if emph else 5)
            else:
                ax.plot(x, y, linestyle="none", marker=st["marker"],
                        ms=st["ms"], mfc="white", mec=st["color"], mew=1.5,
                        zorder=5)

            dx, dy, ha, va = OFFSETS[i][key]
            label = "S3\n(ours)" if emph else key
            ax.annotate(label, (x, y), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va,
                        fontsize=9, color=INK, multialignment="right",
                        fontweight="bold" if emph else "normal", zorder=7)

        # Axes cosmetics.
        ax.set_xscale("log")
        ax.set_xlim(0.78, 40)
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(*ylim)
        ax.set_xlabel("pure inference time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color=GRID, lw=0.8, ls="-", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=3, width=0.8)

        ax.text(0.02, 0.97, LETTERS[i], transform=ax.transAxes,
                ha="left", va="top", fontsize=10, fontweight="bold",
                color=INK, zorder=8)

    # Shared legend (two rows) above the panels.
    handles = []
    for key in ["T25", "T4", "T3r", "S3", "S1"]:
        st = STYLE[key]
        emph = key == "S3"
        if st["fill"]:
            h = Line2D([], [], linestyle="none", marker=st["marker"],
                       ms=st["ms"] * (0.85 if emph else 1.0),
                       mfc=st["color"], mec=INK if emph else "white",
                       mew=0.9 if emph else 1.0, label=LEGEND_NAMES[key])
        else:
            h = Line2D([], [], linestyle="none", marker=st["marker"],
                       ms=st["ms"], mfc="white", mec=st["color"], mew=1.5,
                       label=LEGEND_NAMES[key])
        handles.append(h)
    handles.append(Line2D([], [], color=STYLE["T25"]["color"], lw=1.0,
                          ls=(0, (5, 3)), alpha=0.7,
                          label="teacher (T25) reference level"))

    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.5, columnspacing=1.6,
               borderaxespad=0.0)

    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.2)
    fig.savefig(OUT_PNG, dpi=200)
    print("saved", OUT_PNG)


if __name__ == "__main__":
    main()
