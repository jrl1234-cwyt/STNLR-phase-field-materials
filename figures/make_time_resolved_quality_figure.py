#!/usr/bin/env python3
"""Plot time-resolved Allen--Cahn accuracy and active capacity over all 12 pairs."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
OUT = HERE / "artwork" / "allen_cahn_time_resolved_quality.pdf"
SOURCE = BUNDLE / "results/figure_metadata/allen_cahn_time_resolved_quality.json"
TARGETS = ["eps0022_lam14", "eps0022_lam16", "eps0028_lam14", "eps0028_lam16"]
TIMES = np.arange(1, 11) * 0.06
TEAL = "#287A68"
BLUE = "#3E6C8A"
GOLD = "#D39B35"
GRID = "#D8DEE2"
TEXT = "#26343D"


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(SOURCE.read_text())
    return (
        np.asarray(payload["static_rank7_relative_l2"]),
        np.asarray(payload["stnlr_relative_l2"]),
        np.asarray(payload["stnlr_active_rank"]),
    )


def band(ax, x, rows, color, label):
    mean = rows.mean(0)
    sem = rows.std(0, ddof=1) / np.sqrt(len(rows))
    ax.plot(x, mean, marker="o", markersize=4.8, linewidth=2.25, color=color, label=label)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
    return mean


def main() -> None:
    static, nested, ranks = load()
    plt.rcParams.update({
        "font.size": 9.7,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "pdf.fonttype": 42,
        "mathtext.fontset": "stixsans",
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.15), constrained_layout=True)

    ax = axes[0]
    static_mean = band(ax, TIMES, static, BLUE, "Static rank 7")
    nested_mean = band(ax, TIMES, nested, TEAL, "ST-NLR")
    ax.set_xlabel("Physical time $\\tau$")
    ax.set_ylabel("Relative $L_2$ (lower is better)")
    ax.set_title("(a) Forecast error", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    reduction = 100 * (static_mean - nested_mean) / static_mean
    ax.plot(TIMES, reduction, marker="o", markersize=4.8, linewidth=2.25, color=TEAL)
    ax.fill_between(TIMES, 0, reduction, color=TEAL, alpha=0.13)
    ax.axhline(0, color="#6E7B83", linewidth=0.9)
    ax.set_ylim(0, max(reduction) * 1.22)
    ax.set_xlabel("Physical time $\\tau$")
    ax.set_ylabel("Error reduction vs. rank 7 (%)")
    ax.set_title("(b) Gain at every forecast time", fontweight="bold")
    ax.text(0.96, 0.10, "7.99--10.48% lower",
            transform=ax.transAxes, ha="right", va="bottom",
            color=TEAL, fontweight="bold")

    ax = axes[2]
    rank_mean = ranks.mean(0)
    rank_sem = ranks.std(0, ddof=1) / np.sqrt(len(ranks))
    ax.plot(TIMES, rank_mean, marker="o", markersize=4.8, linewidth=2.25,
            color=GOLD, label="ST-NLR")
    ax.fill_between(TIMES, rank_mean - rank_sem, rank_mean + rank_sem,
                    color=GOLD, alpha=0.18, linewidth=0)
    ax.axhline(7, color=BLUE, linestyle="--", linewidth=1.4, label="Static rank 7")
    ax.axhline(16, color="#7B8388", linestyle=":", linewidth=1.2, label="Maximum rank 16")
    ax.set_ylim(2.8, 16.8)
    ax.set_xlabel("Physical time $\\tau$")
    ax.set_ylabel("Active rank")
    ax.set_title("(c) Capacity follows physical time", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    for ax in axes:
        ax.set_xticks([0.06, 0.18, 0.30, 0.42, 0.54, 0.60])
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
