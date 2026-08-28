#!/usr/bin/env python3
"""Regenerate the representative-state figures with English labels."""

from pathlib import Path
import importlib.util
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "material_state_inference.py"
WORKSPACE = HERE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
SPEC = importlib.util.spec_from_file_location("state_source", SOURCE)
SRC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SRC)
SRC.OUT = HERE / "artwork"
NAVY = "#29465B"
TEAL = "#3F8268"


def plot_allen(truth, static, nested, times, trace, selected, time_ids, output):
    plt.rcParams.update({"pdf.fonttype": 42, "mathtext.fontset": "stixsans"})
    fig, axes = plt.subplots(2, 6, figsize=(14.3, 5.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.94, 0.94, 0.94]})
    errors_all = []
    for tid in time_ids:
        errors_all += [np.abs(static[selected, tid] - truth[selected, tid]),
                       np.abs(nested[selected, tid] - truth[selected, tid])]
    error_max = max(max(float(np.quantile(v, 0.995)) for v in errors_all), 1e-4)
    differences = [errors_all[i] - errors_all[i + 1] for i in range(0, len(errors_all), 2)]
    diff_max = max(max(float(np.quantile(np.abs(v), 0.995)) for v in differences), 1e-4)
    titles = ["Reference", "Static rank 7", "ST-NLR", "Rank-7\nabsolute error",
              "ST-NLR\nabsolute error", "Error difference\n(blue: improvement)"]
    field_image = error_image = diff_image = None
    for row, tid in enumerate(time_ids):
        fields = [truth[selected, tid], static[selected, tid], nested[selected, tid]]
        errors = [np.abs(fields[1] - fields[0]), np.abs(fields[2] - fields[0])]
        improvement = errors[0] - errors[1]
        static_l2 = np.linalg.norm(fields[1] - fields[0]) / max(np.linalg.norm(fields[0]), 1e-8)
        nested_l2 = np.linalg.norm(fields[2] - fields[0]) / max(np.linalg.norm(fields[0]), 1e-8)
        reduction = 100 * (static_l2 - nested_l2) / max(static_l2, 1e-8)
        for col, value in enumerate(fields + errors + [improvement]):
            if col < 3:
                field_image = axes[row, col].imshow(value, origin="lower", cmap="RdBu_r",
                                                     vmin=-1, vmax=1, interpolation="nearest")
                axes[row, col].contour(value, levels=[0], colors=NAVY, linewidths=0.65)
            elif col < 5:
                error_image = axes[row, col].imshow(value, origin="lower", cmap="magma",
                                                     vmin=0, vmax=error_max, interpolation="nearest")
            else:
                diff_image = axes[row, col].imshow(value, origin="lower", cmap="RdBu",
                                                    vmin=-diff_max, vmax=diff_max, interpolation="nearest")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=13.2, fontweight="bold")
        axes[row, 3].text(0.04, 0.94, rf"$L_2={static_l2:.4f}$",
                          transform=axes[row, 3].transAxes, va="top", fontsize=11,
                          color="white", bbox=dict(boxstyle="round,pad=0.18", facecolor="#202020",
                                                   edgecolor="none", alpha=0.78))
        axes[row, 4].text(0.04, 0.94, rf"$L_2={nested_l2:.4f}$" + "\n" + rf"$\downarrow {reduction:.1f}\%$",
                          transform=axes[row, 4].transAxes, va="top", fontsize=11,
                          color="white", bbox=dict(boxstyle="round,pad=0.18", facecolor="#1D5B45",
                                                   edgecolor="none", alpha=0.84))
        axes[row, 0].set_ylabel(rf"$\tau={times[tid]:.2f}$" + "\n" + rf"active rank $={trace[tid]}$",
                                fontsize=13.2)
    bar = fig.colorbar(field_image, ax=axes[:, (0, 1, 2)], shrink=0.80, pad=0.012)
    bar.set_label(r"Phase field $u$", fontsize=12)
    bar = fig.colorbar(error_image, ax=axes[:, (3, 4)], shrink=0.80, pad=0.012)
    bar.set_label("Absolute error", fontsize=12)
    bar = fig.colorbar(diff_image, ax=axes[:, 5], shrink=0.80, pad=0.012)
    bar.set_label("Rank-7 error $-$ ST-NLR error", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


def plot_dendrite(truth, static, nested, selected, terminal_time, terminal_rank, output):
    plt.rcParams.update({"pdf.fonttype": 42, "mathtext.fontset": "stixsans"})
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.25), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.94, 1, 0.94]})
    titles = ["Reference", "Continued-static", "Static\nabsolute error",
              "ST-NLR", "ST-NLR\nabsolute error"]
    state_ranges = [(-1.0, 1.0, "RdBu_r"), (-0.32, 0.02, "magma")]
    row_labels = [rf"Phase field $\phi$" + "\n" + rf"$\tau={terminal_time:.0f}$, rank $={terminal_rank}$",
                  r"Temperature field $U$"]
    for row in range(2):
        reference = truth[selected, row]
        static_field = static[selected, row]
        nested_field = nested[selected, row]
        static_error = np.abs(static_field - reference)
        nested_error = np.abs(nested_field - reference)
        error_max = max(float(np.quantile(static_error, 0.995)),
                        float(np.quantile(nested_error, 0.995)), 1e-4)
        values = [reference, static_field, static_error, nested_field, nested_error]
        state_image = error_image = None
        for col, value in enumerate(values):
            if col in (0, 1, 3):
                vmin, vmax, cmap = state_ranges[row]
                state_image = axes[row, col].imshow(value, origin="lower", vmin=vmin, vmax=vmax,
                                                     cmap=cmap, interpolation="nearest")
                if row == 0:
                    axes[row, col].contour(value, levels=[0], colors=NAVY, linewidths=0.7)
            else:
                error_image = axes[row, col].imshow(value, origin="lower", vmin=0, vmax=error_max,
                                                     cmap="magma", interpolation="nearest")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=13.0, fontweight="bold")
        static_l2 = np.linalg.norm((static_field - reference).ravel()) / max(np.linalg.norm(reference.ravel()), 1e-8)
        nested_l2 = np.linalg.norm((nested_field - reference).ravel()) / max(np.linalg.norm(reference.ravel()), 1e-8)
        reduction = 100 * (static_l2 - nested_l2) / max(static_l2, 1e-8)
        axes[row, 4].text(0.04, 0.94, rf"$L_2$ error $\downarrow$ {reduction:.1f}\%",
                          transform=axes[row, 4].transAxes, va="top", color="white", fontsize=10.5,
                          fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor=TEAL,
                                                       edgecolor="none", alpha=0.92))
        axes[row, 0].set_ylabel(row_labels[row], fontsize=13)
        fig.colorbar(state_image, ax=axes[row, (0, 1, 3)], shrink=0.82, pad=0.014)
        bar = fig.colorbar(error_image, ax=axes[row, (2, 4)], shrink=0.82, pad=0.014)
        bar.set_label("Absolute error", fontsize=10.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    SRC.plot_comparison = plot_allen
    SRC.plot_dendrite_comparison = plot_dendrite
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SRC.run_task("allen", device)
    SRC.run_dendrite(device)
