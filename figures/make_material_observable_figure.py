#!/usr/bin/env python3
"""Plot free-energy and interface trajectories from fixed Allen--Cahn checkpoints."""

from pathlib import Path
import importlib.util
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = HERE.parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
SOURCE = HERE / "material_state_inference.py"
SPEC = importlib.util.spec_from_file_location("state_source_observables", SOURCE)
SRC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SRC)

DATA = BUNDLE / "data/full/allen_cahn/eps0022_lam14.npz"
STATIC = BUNDLE / "checkpoints/allen_cahn/static_rank7_eps0022_lam14_seed0.pt"
NESTED = BUNDLE / "checkpoints/allen_cahn/stnlr_eps0022_lam14_seed0.pt"
TRACE = BUNDLE / "results/figure_metadata/allen_cahn_rank7_representative_states.json"
OUT = HERE / "artwork" / "allen_cahn_material_observable_trajectories.pdf"
TEAL = "#287A68"
BLUE = "#3E6C8A"
BLACK = "#22272B"
GRID = "#D8DEE2"


def free_energy(fields: np.ndarray, epsilon: float, reaction: float) -> np.ndarray:
    dx = 1.0 / fields.shape[-1]
    gx = (np.roll(fields, -1, axis=-1) - np.roll(fields, 1, axis=-1)) / (2 * dx)
    gy = (np.roll(fields, -1, axis=-2) - np.roll(fields, 1, axis=-2)) / (2 * dx)
    density = 0.5 * epsilon**2 * (gx**2 + gy**2) + 0.25 * reaction * (fields**2 - 1)**2
    return density.mean(axis=(-2, -1))


def interface_fraction(fields: np.ndarray) -> np.ndarray:
    return (np.abs(fields) < 0.2).mean(axis=(-2, -1))


def mean_sem(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rows.mean(0), rows.std(0, ddof=1) / np.sqrt(len(rows))


def curve(ax, x, rows, color, label, marker="o"):
    mean, sem = mean_sem(rows)
    ax.plot(x, mean, color=color, linewidth=2.1, marker=marker, markersize=4.0, label=label)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.13, linewidth=0)
    return mean


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = np.load(DATA)
    selection = slice(120, 160)
    initial = payload["initial"][selection]
    truth = payload["fields"][selection]
    times = payload["times"]
    trace = json.loads(TRACE.read_text())["rank_trace"]

    static_model = SRC.load_model("static", STATIC, device)
    static = SRC.predict(static_model, "static", initial, times, None, device)
    del static_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    nested_model = SRC.load_model("stnlr", NESTED, device)
    nested = SRC.predict(nested_model, "stnlr", initial, times, trace, device)
    del nested_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    epsilon, reaction = 0.022, 1.4
    e_ref = free_energy(truth, epsilon, reaction)
    e_static = free_energy(static, epsilon, reaction)
    e_nested = free_energy(nested, epsilon, reaction)
    i_ref = interface_fraction(truth)
    i_static = interface_fraction(static)
    i_nested = interface_fraction(nested)
    x = times[1:]
    norm = e_ref[:, [0]]

    plt.rcParams.update({"font.size": 9.6, "pdf.fonttype": 42, "mathtext.fontset": "stixsans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.15), constrained_layout=True)

    ax = axes[0]
    curve(ax, x, e_ref[:, 1:] / norm, BLACK, "Reference", marker="")
    curve(ax, x, e_static[:, 1:] / norm, BLUE, "Static rank 7", marker="s")
    curve(ax, x, e_nested[:, 1:] / norm, TEAL, "ST-NLR", marker="o")
    ax.set_xlabel("Physical time $\\tau$")
    ax.set_ylabel("Normalized free energy $\\mathcal{E}(\\tau)/\\mathcal{E}(0)$")
    ax.set_title("(a) Free-energy evolution", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    energy_gain = 100 * (np.mean(np.abs(e_static[:, 1:] - e_ref[:, 1:]))
                         - np.mean(np.abs(e_nested[:, 1:] - e_ref[:, 1:]))) \
                  / np.mean(np.abs(e_static[:, 1:] - e_ref[:, 1:]))
    ax.text(0.04, 0.08, f"Mean trajectory deviation $\\downarrow$ {energy_gain:.1f}%",
            transform=ax.transAxes, color=TEAL, fontweight="bold", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#E7F2EE", edgecolor="none"))

    ax = axes[1]
    curve(ax, x, i_ref[:, 1:], BLACK, "Reference", marker="")
    curve(ax, x, i_static[:, 1:], BLUE, "Static rank 7", marker="s")
    curve(ax, x, i_nested[:, 1:], TEAL, "ST-NLR", marker="o")
    ax.set_xlabel("Physical time $\\tau$")
    ax.set_ylabel("Interface fraction, $|u|<0.2$")
    ax.set_title("(b) Interface evolution", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    interface_gain = 100 * (np.mean(np.abs(i_static[:, 1:] - i_ref[:, 1:]))
                            - np.mean(np.abs(i_nested[:, 1:] - i_ref[:, 1:]))) \
                     / np.mean(np.abs(i_static[:, 1:] - i_ref[:, 1:]))
    ax.text(0.04, 0.08, f"Mean trajectory deviation $\\downarrow$ {interface_gain:.1f}%",
            transform=ax.transAxes, color=TEAL, fontweight="bold", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#E7F2EE", edgecolor="none"))

    for ax in axes:
        ax.set_xticks([0.06, 0.18, 0.30, 0.42, 0.54, 0.60])
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)

    # Store the audit values used to decide whether the figure belongs in the paper.
    audit = {
        "target": "eps0022_lam14",
        "seed": 0,
        "test_count": 40,
        "free_energy_mae_over_forecast_times": {
            "static_rank7": float(np.mean(np.abs(e_static[:, 1:] - e_ref[:, 1:]))),
            "stnlr": float(np.mean(np.abs(e_nested[:, 1:] - e_ref[:, 1:]))),
        },
        "interface_fraction_mae_over_forecast_times": {
            "static_rank7": float(np.mean(np.abs(i_static[:, 1:] - i_ref[:, 1:]))),
            "stnlr": float(np.mean(np.abs(i_nested[:, 1:] - i_ref[:, 1:]))),
        },
    }
    metadata = BUNDLE / "results/figure_metadata/allen_cahn_material_observable_trajectories.json"
    metadata.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
