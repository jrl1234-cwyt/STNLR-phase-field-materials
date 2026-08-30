#!/usr/bin/env python3
"""Generate the English framework, data-budget, and Pareto figures."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
OUT = HERE / "artwork"
TEXT = "#26343D"
MUTED = "#5F6B73"
NAVY = "#34566B"
LIGHT_BLUE = "#78B9E6"
BLUE = "#BBD4ED"
ICE = "#EAEFF8"
PEACH = "#FFDBC5"
YELLOW = "#FFEEBB"
MINT = "#E2F0D9"
PURPLE = "#F4DAF9"
GREEN = "#CFE7CA"
TEAL = "#3F8268"
GRAY = "#98A2A8"
GRID = "#D9DEE2"


def box(ax, x, y, w, h, face, edge=NAVY, lw=2.0, radius=0.015):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, dashed=False, lw=3.2):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18,
        linewidth=lw, color=LIGHT_BLUE,
        linestyle=(0, (2.2, 2.2)) if dashed else "-",
        connectionstyle="arc3,rad=0", shrinkA=1, shrinkB=1,
    ))


def label(ax, x, y, title, subtitle="", size=13.0):
    ax.text(x, y + (0.018 if subtitle else 0), title, ha="center", va="center",
            fontsize=size, fontweight="bold", color=TEXT)
    if subtitle:
        ax.text(x, y - 0.024, subtitle, ha="center", va="center",
                fontsize=size - 2.0, color=MUTED, linespacing=1.25)


def stage(ax, n, y, title, subtitle, face, edge, *, title_size=14.2,
          title_offset=0.130, subtitle_offset=0.075):
    box(ax, 0.018, y, 0.145, 0.24, face, edge=edge, lw=2.1, radius=0.02)
    ax.add_patch(Circle((0.045, y + 0.196), 0.017, facecolor=edge,
                        edgecolor="white", linewidth=0.8))
    ax.text(0.045, y + 0.196, str(n), ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(0.090, y + title_offset, title, ha="center", va="center",
            color=TEXT, fontsize=title_size, fontweight="bold", linespacing=1.05)
    ax.text(0.090, y + subtitle_offset, subtitle, ha="center", va="center",
            color=edge, fontsize=11.5, fontweight="bold")


def framework() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "mathtext.fontset": "stixsans"})
    fig, ax = plt.subplots(figsize=(15.8, 8.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.975,
            "Physical-time-adaptive nested low-rank transfer to unseen material conditions",
            ha="center", va="top", fontsize=20.5, fontweight="bold", color=TEXT)

    ys = [0.685, 0.372, 0.059]
    stage(ax, 1, ys[0], "Material\ntrajectory", "Define the problem", ICE, "#687CB7")
    stage(ax, 2, ys[1], "Nested low-rank\ntransfer", "Allocate capacity", YELLOW, "#B96D4C")
    stage(ax, 3, ys[2], "Material\nconstrained\ndeployment", "Select the prefix",
          MINT, "#598539", title_size=13.4, title_offset=0.145,
          subtitle_offset=0.052)
    # Vertical dashed connectors are entirely outside the stage cards.
    arrow(ax, (0.090, 0.679), (0.090, 0.623), dashed=True, lw=3.6)
    arrow(ax, (0.090, 0.366), (0.090, 0.310), dashed=True, lw=3.6)

    # Stage 1.
    box(ax, 0.178, ys[0], 0.804, 0.24, ICE, edge="#687CB7", lw=2.1, radius=0.02)
    ax.text(0.196, 0.895, "a. Physical evolution defined by material free energy",
            fontsize=14.5, fontweight="bold", color="#5D6490", va="center")
    items = [
        (0.195, 0.716, 0.150, 0.140, "white", "Target condition", r"$u_0,\ \theta=(\epsilon,\lambda,\ldots)$"),
        (0.375, 0.716, 0.180, 0.140, PEACH, "Free-energy\nfunctional", r"$\mathcal{E}[u;\theta]$"),
        (0.585, 0.716, 0.180, 0.140, YELLOW, "Dissipative flow", r"$\partial_\tau u=-\mathcal{G}\,\delta\mathcal{E}/\delta u$"),
        (0.795, 0.716, 0.160, 0.140, "white", "Microstructure", r"$u_0\rightarrow u_{\tau_1}\rightarrow u_{\tau_2}$"),
    ]
    for x, y, w, h, face, title, sub in items:
        box(ax, x, y, w, h, face, lw=1.8)
        label(ax, x + w/2, y + h/2, title, sub, 12.4)
    for x1, x2 in [(0.346, 0.374), (0.556, 0.584), (0.766, 0.794)]:
        arrow(ax, (x1, 0.786), (x2, 0.786))

    # Stage 2.
    box(ax, 0.178, ys[1], 0.804, 0.24, YELLOW, edge="#B96D4C", lw=2.1, radius=0.02)
    ax.text(0.196, 0.582, "b. Frozen pretrained backbone with a shared nested residual bank",
            fontsize=14.5, fontweight="bold", color="#A55E42", va="center")
    stage2 = [
        (0.195, 0.402, 0.180, 0.145, BLUE, "Multi-PDE\nbackbone", "$\\mathcal{F}_0$ frozen\ntrainable input/output heads"),
        (0.395, 0.397, 0.280, 0.155, "white", "One shared rank-16\nfactor bank", "$r\\in\\{4,8,16\\}$ nested prefixes\nshared directions across time"),
        (0.695, 0.402, 0.130, 0.145, PURPLE, "Physical time", r"$\tau\mapsto r(\tau)$"),
        (0.845, 0.402, 0.110, 0.145, GREEN, "Output", r"$\widehat u_\tau$"),
    ]
    for x, y, w, h, face, title, sub in stage2:
        box(ax, x, y, w, h, face, lw=1.8)
        label(ax, x + w/2, y + h/2, title, sub, 12.2)
    for x1, x2 in [(0.376, 0.394), (0.676, 0.694), (0.826, 0.844)]:
        arrow(ax, (x1, 0.472), (x2, 0.472))

    # Stage 3.
    box(ax, 0.178, ys[2], 0.804, 0.24, MINT, edge="#598539", lw=2.1, radius=0.02)
    ax.text(0.196, 0.269, "c. Smallest feasible prefix selected by material-quality tolerances",
            fontsize=14.5, fontweight="bold", color="#4C7733", va="center")
    stage3 = [
        (0.195, 0.087, 0.205, 0.145, "white", "Material objectives", "field · interface · energy\nconservation · structure\nspectrum"),
        (0.420, 0.087, 0.170, 0.145, BLUE, "Joint calibration", "rank-16 anchor\nrank-8/4 distillation"),
        (0.610, 0.087, 0.175, 0.145, PURPLE, "Validation\ntolerance", "$D_{j,k}(r)\\leq\\varepsilon_k$\n$r_j^*=\\min\\mathcal{F}_j$"),
        (0.805, 0.087, 0.150, 0.145, PEACH, "Material output", "preserved quality\nlower active cost"),
    ]
    for x, y, w, h, face, title, sub in stage3:
        box(ax, x, y, w, h, face, lw=1.8)
        label(ax, x + w/2, y + h/2, title, sub, 12.0)
    for x1, x2 in [(0.401, 0.419), (0.591, 0.609), (0.786, 0.804)]:
        arrow(ax, (x1, 0.157), (x2, 0.157))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "material_stnlr_framework_theory.pdf", bbox_inches="tight",
                facecolor="white", pad_inches=0.03)
    plt.close(fig)


def data_budget() -> None:
    payload = json.loads((BUNDLE / "results/data_budget.json").read_text())
    summary = payload["summary"]
    budgets = np.array([20, 50, 100])
    trajectory = np.array([summary[str(n)]["trajectory_relative_l2_mean"]["stnlr_mean"] for n in budgets])
    trajectory_sd = np.array([summary[str(n)]["trajectory_relative_l2_mean"]["stnlr_sd"] for n in budgets])
    terminal = np.array([summary[str(n)]["terminal_relative_l2_mean"]["stnlr_mean"] for n in budgets])
    terminal_sd = np.array([summary[str(n)]["terminal_relative_l2_mean"]["stnlr_sd"] for n in budgets])
    active_rank = np.array([summary[str(n)]["mean_active_rank"]["mean"] for n in budgets])
    savings = 100 * (1 - active_rank / 16)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(budgets, trajectory, yerr=trajectory_sd, marker="o", linewidth=2.3,
                capsize=3.2, color=TEAL, label="Trajectory")
    ax.errorbar(budgets, terminal, yerr=terminal_sd, marker="s", linewidth=2.3,
                capsize=3.2, color="#426E86", label="Terminal state")
    ax.annotate("20 trajectories\n$\geq$99.3% field-quality retention",
                xy=(20, terminal[0]), xytext=(35, 0.0552), fontsize=9.0,
                color=TEAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TEAL, linewidth=1.2))
    ax.set_xticks(budgets); ax.set_xlabel("Target trajectories")
    ax.set_ylabel("Relative $L_2$ (lower is better)")
    ax.set_title("(a) Field quality across data budgets", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    bars = ax.bar(np.arange(3), savings, width=0.58,
                  color=[TEAL, "#6FA58F", "#8CB9A6"], edgecolor="white")
    ax.set_xticks(np.arange(3), [str(n) for n in budgets])
    ax.set_xlabel("Target trajectories")
    ax.set_ylabel("Active-capacity saving vs. rank 16 (%)")
    ax.set_ylim(0, 75); ax.set_title("(b) Active-capacity reduction", fontweight="bold")
    for bar, value in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width()/2, value + 1.5, f"{value:.2f}%",
                ha="center", fontweight="bold", color=TEAL)
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.7); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "allen_cahn_data_budget_curve.pdf", bbox_inches="tight",
                facecolor="white", pad_inches=0.03)
    plt.close(fig)


def pareto() -> None:
    payload = json.loads((BUNDLE / "results/fixed_prefix_pareto.json").read_text())
    strategies = payload["strategies"]
    trace = np.asarray(payload["validation_selected_rank_trace"], dtype=int)
    feasible_methods = [method for method in ("dynamic", "4", "8", "16")
                        if strategies[method]["validation_feasible_under_timewise_2_percent_rule"]]
    names = [strategies[method]["label"] for method in feasible_methods]
    savings = np.asarray([
        strategies[method]["active_capacity_saving_percent"] for method in feasible_methods
    ])
    ranks = np.asarray([strategies[method]["mean_active_rank"] for method in feasible_methods])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    ax = axes[0]
    labels = [f"{name}\nmean rank {rank:.2f}" for name, rank in zip(names, ranks)]
    colors = [TEAL if method == "dynamic" else "#4C5961" for method in feasible_methods]
    bars = ax.barh(np.arange(len(names)), savings, color=colors, height=0.56)
    ax.set_yticks(np.arange(len(names)), labels); ax.invert_yaxis(); ax.set_xlim(0, 70)
    ax.set_xlabel("Active-capacity saving vs. all rank 16 (%)")
    ax.set_title("(a) Validation-feasible capacity", fontweight="bold")
    for bar, value in zip(bars, savings):
        ax.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.2f}%",
                va="center", fontweight="bold", color=TEAL if value == savings.max() else TEXT)
    ax = axes[1]
    time_index = np.arange(len(trace))
    ax.step(time_index, trace, where="mid", color=TEAL, linewidth=2.6,
            label=f"Dynamic ST-NLR (mean {trace.mean():.2f})")
    ax.axhline(16, color="#4C5961", linewidth=1.8, linestyle="--",
               label="All rank 16")
    ax.scatter(time_index, trace, s=28, color=TEAL, edgecolor="white", zorder=3)
    ax.set_xticks(time_index)
    ax.set_xlabel("Physical-time index")
    ax.set_ylabel("Active rank")
    ax.set_yticks([4, 8, 16])
    ax.set_ylim(2.5, 17.5)
    ax.set_title("(b) Frozen validation-selected schedule", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92,
              fontsize=8.6, loc="center right")
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.7); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fixed_prefix_pareto.pdf", bbox_inches="tight",
                facecolor="white", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    data_budget()
    pareto()
