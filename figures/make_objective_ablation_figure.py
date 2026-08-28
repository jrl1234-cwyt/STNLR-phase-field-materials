#!/usr/bin/env python3
"""Plot the predeclared objective-ablation outcomes used in the appendix."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULT = HERE.parent / "results/objective_ablation.json"
OUTPUT = HERE / "artwork/objective_ablation_capacity_interface.pdf"

payload = json.loads(RESULT.read_text())
rows = payload["rows"]
order = ["full_stnlr", "no_field_distillation", "no_spectral_distillation", "no_material_objectives"]
labels = ["Full\nST-NLR", "No field\ndistillation", "No spectral\ndistillation", "No material\nobjectives"]
colors = ["#3F8268", "#9AB9D0", "#D9B26F", "#C99A88"]


def values(metric):
    groups = [np.asarray([row[metric] for row in rows if row["variant"] == name]) for name in order]
    return np.asarray([x.mean() for x in groups]), np.asarray([x.std(ddof=1) for x in groups])


plt.rcParams.update({"pdf.fonttype": 42, "mathtext.fontset": "stixsans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
x = np.arange(len(order))

means, errors = values("mean_active_rank")
axes[0].bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="#354A57", linewidth=0.7)
axes[0].set_ylabel("Mean active rank")
axes[0].set_title("Capacity selected by validation", fontweight="bold")
axes[0].set_ylim(0, max(means + errors) * 1.18)

means, errors = values("terminal_interface_fraction_mae")
axes[1].bar(x, 1e3 * means, yerr=1e3 * errors, capsize=3, color=colors,
            edgecolor="#354A57", linewidth=0.7)
axes[1].set_ylabel(r"Terminal interface MAE ($\times 10^{-3}$)")
axes[1].set_title("Material-interface preservation", fontweight="bold")
axes[1].set_ylim(0, max(1e3 * (means + errors)) * 1.18)

for ax in axes:
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
plt.close(fig)
print(OUTPUT)
