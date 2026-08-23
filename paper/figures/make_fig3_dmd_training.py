"""Figure 3: DMD training diagnostic (std ratio vs. training step).

Parses lines of the form
    [DIAG step N] ... std비 full=0.791 1step=0.695 ...
from the DMD run log (dmd_6a.log) and plots the std ratio
std(student x0) / std(teacher latent) against the training step for the
student's full sampling schedule (3 steps) and for a single generator call.

Run with the paper venv python:
    .../scratchpad/venv/bin/python make_fig3_dmd_training.py
"""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.abspath(os.path.join(HERE, "..", ".."))
LOG = os.path.join(SCRATCH, "repo", "results", "quantitative", "logs", "dmd_6a.log")
OUT = os.path.join(HERE, "fig3_dmd_training.png")

# Values stated in FACTS.md / the spec for the annotations (rounded log values).
SELECTED_STEP = 1600
SELECTED_RATIO_TXT = "1.06"
OVERSHOOT_STEP = 2000
OVERSHOOT_RATIO_TXT = "1.11"

# Colors (validated with the dataviz palette validator, light surface):
# blue categorical slot 1 for the primary series, lighter blue (ordinal ramp)
# for the secondary series; text/grid in neutral ink tokens.
C_FULL = "#2a78d6"
C_1STEP = "#86b6ef"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#e6e5e1"

PAT = re.compile(r"\[DIAG step (\d+)\].*?std비 full=([0-9.]+) 1step=([0-9.]+)")


def parse(path):
    steps, full, one = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = PAT.search(line)
            if m:
                steps.append(int(m.group(1)))
                full.append(float(m.group(2)))
                one.append(float(m.group(3)))
    return steps, full, one


def main():
    steps, full, one = parse(LOG)
    if not steps:
        sys.exit(f"no DIAG lines found in {LOG}")
    print("parsed %d points" % len(steps))
    for s, a, b in zip(steps, full, one):
        print(f"step {s:5d}  full={a:.3f}  1step={b:.3f}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": C_TEXT2,
        "axes.labelcolor": C_TEXT,
        "xtick.color": C_TEXT2,
        "ytick.color": C_TEXT2,
        "text.color": C_TEXT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    # Recessive grid and axes.
    ax.grid(axis="y", color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)

    # Teacher level (ratio 1.0).
    ax.axhline(1.0, color=C_TEXT2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(20, 1.008, "teacher level (ratio 1.0)", color=C_TEXT2, fontsize=9,
            ha="left", va="bottom", zorder=4)

    # Series.
    ax.plot(steps, full, color=C_FULL, linewidth=1.8, marker="o", markersize=3.5,
            markeredgewidth=0, label="full schedule (3 steps)", zorder=3)
    ax.plot(steps, one, color=C_1STEP, linewidth=1.5, linestyle=(0, (5, 2.5)),
            marker="o", markersize=3.6, markeredgewidth=0, label="single step", zorder=2)

    # Selected checkpoint marker.
    sel_ratio = full[steps.index(SELECTED_STEP)]
    ax.axvline(SELECTED_STEP, color=C_TEXT, linewidth=0.9, linestyle=(0, (1.5, 2.5)), zorder=1)
    ax.plot([SELECTED_STEP], [sel_ratio], marker="o", markersize=8, markerfacecolor="white",
            markeredgecolor=C_FULL, markeredgewidth=1.6, linestyle="none", zorder=5)
    ax.text(SELECTED_STEP - 30, 0.672,
            f"selected checkpoint\n(step {SELECTED_STEP}, ratio {SELECTED_RATIO_TXT})",
            ha="right", va="bottom", fontsize=9, color=C_TEXT, zorder=4)

    # Overshoot annotation at the last step.
    ov_ratio = full[steps.index(OVERSHOOT_STEP)]
    ax.annotate(f"overshoot (ratio {OVERSHOOT_RATIO_TXT}),\nworse in evaluation",
                xy=(OVERSHOOT_STEP, ov_ratio), xytext=(1760, 1.205),
                ha="right", va="top", fontsize=9, color=C_TEXT,
                arrowprops=dict(arrowstyle="-|>", color=C_TEXT2, linewidth=0.9,
                                shrinkA=2, shrinkB=5, mutation_scale=9),
                zorder=6)

    ax.set_xlim(-40, 2060)
    ax.set_ylim(0.65, 1.215)
    ax.set_xticks(range(0, 2001, 400))
    ax.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    ax.set_xlabel("training step (generator updates)")
    ax.set_ylabel("std ratio, std(student $x_0$) / std(teacher latent)")

    ax.legend(loc="upper left", frameon=False, handlelength=2.6, borderaxespad=0.4,
              labelcolor=C_TEXT)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=200)
    print("saved", OUT)


if __name__ == "__main__":
    main()
