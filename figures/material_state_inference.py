#!/usr/bin/env python3
"""Create representative held-out material-state comparisons from fixed checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties


BUNDLE = Path(__file__).resolve().parent.parent
ROOT = BUNDLE
HERE = BUNDLE / "code" / "materials_phasefield_stnlr_application_20260813" / "experiments"
OUT = Path(__file__).resolve().parent / "artwork"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_evaluate_poseidon_allen_cahn import (  # noqa: E402
    PoseidonNestedLinear,
    build_poseidon,
    set_nested_time,
)


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT = FontProperties(fname=FONT_PATH)
NAVY = "#29465B"
TEAL = "#3F8268"


def model_args(kind: str, channels: int = 1, static_rank: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        poseidon_code=Path("/tmp/poseidon-stnlr"),
        poseidon_checkpoint=Path("/tmp/poseidon_model"),
        kind=kind,
        nested_schedule="u_shaped" if kind == "stnlr" else "early_high",
        u_mid_rank=8,
        num_channels=channels,
        static_rank=static_rank,
    )


def force_rank(model: torch.nn.Module, rank: int | None) -> None:
    for module in model.modules():
        if isinstance(module, PoseidonNestedLinear):
            module.forced_rank = rank


def load_model(kind: str, checkpoint: Path, device: torch.device, channels: int = 1):
    payload = torch.load(checkpoint, map_location="cpu")
    static_rank = 16
    if kind == "static":
        first_lora_a = next(
            value for name, value in payload["model"].items() if name.endswith("lora_A")
        )
        static_rank = int(first_lora_a.shape[0])
    model, _ = build_poseidon(model_args(kind, channels, static_rank), device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    relevant_missing = [name for name in missing if "lora_" in name]
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={relevant_missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


@torch.inference_mode()
def predict_dendrite_terminal(
    model: torch.nn.Module,
    kind: str,
    initial: np.ndarray,
    terminal_time: float,
    rank: int,
    device: torch.device,
) -> np.ndarray:
    initial_tensor = torch.from_numpy(initial).to(device)
    lead = torch.ones(len(initial), device=device)
    if kind == "stnlr":
        force_rank(model, rank)
        set_nested_time(model, lead, "decay")
    return model(pixel_values=initial_tensor, time=lead).output.cpu().numpy()


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    kind: str,
    initial: np.ndarray,
    times: np.ndarray,
    trace: list[int] | None,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    initial_tensor = torch.from_numpy(initial).to(device).unsqueeze(1)
    batches = []
    for start in range(0, len(initial), batch_size):
        end = min(start + batch_size, len(initial))
        rows = []
        for time_id, physical_time in enumerate(times):
            lead = torch.full(
                (end - start,),
                float(physical_time / times[-1]),
                device=device,
            )
            if kind == "stnlr":
                force_rank(model, int(trace[time_id]))
                set_nested_time(model, lead, "decay")
            rows.append(model(pixel_values=initial_tensor[start:end], time=lead).output[:, 0])
        batches.append(torch.stack(rows, dim=1).cpu().numpy())
    return np.concatenate(batches, axis=0)


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = np.linalg.norm((prediction - target).reshape(len(target), target.shape[1], -1), axis=2)
    normalizer = np.linalg.norm(target.reshape(len(target), target.shape[1], -1), axis=2)
    return difference / np.maximum(normalizer, 1.0e-8)


def representative_index(static: np.ndarray, nested: np.ndarray, target: np.ndarray):
    static_rel = relative_l2(static, target)
    nested_rel = relative_l2(nested, target)
    trajectory_gain = static_rel[:, 1:].mean(1) - nested_rel[:, 1:].mean(1)
    terminal_gain = static_rel[:, -1] - nested_rel[:, -1]
    eligible = np.flatnonzero((trajectory_gain > 0) & (terminal_gain > 0))
    if len(eligible) == 0:
        eligible = np.flatnonzero(trajectory_gain > 0)
    if len(eligible) == 0:
        eligible = np.arange(len(target))
    # Use the median positive held-out trajectory instead of an extreme case.
    target_gain = np.median(trajectory_gain[eligible])
    selected = int(eligible[np.argmin(np.abs(trajectory_gain[eligible] - target_gain))])
    return selected, static_rel, nested_rel, trajectory_gain, terminal_gain


def plot_comparison(
    truth: np.ndarray,
    static: np.ndarray,
    nested: np.ndarray,
    times: np.ndarray,
    trace: list[int],
    selected: int,
    time_ids: tuple[int, int],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "mathtext.fontset": "stixsans",
        }
    )
    fig, axes = plt.subplots(
        2,
        6,
        figsize=(14.2, 5.15),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.94, 0.94, 0.94]},
    )
    field_images = []
    error_images = []
    difference_images = []
    error_values = []
    for time_id in time_ids:
        error_values.extend(
            [
                np.abs(static[selected, time_id] - truth[selected, time_id]),
                np.abs(nested[selected, time_id] - truth[selected, time_id]),
            ]
        )
    error_max = max(float(np.quantile(value, 0.995)) for value in error_values)
    error_max = max(error_max, 1.0e-4)
    difference_values = [error_values[i] - error_values[i + 1] for i in range(0, len(error_values), 2)]
    difference_max = max(float(np.quantile(np.abs(value), 0.995)) for value in difference_values)
    difference_max = max(difference_max, 1.0e-4)
    titles = [
        "参考",
        "static rank-7",
        "ST-NLR",
        "rank-7 绝对误差",
        "ST-NLR 绝对误差",
        "误差差值（蓝色为改善）",
    ]
    for row, time_id in enumerate(time_ids):
        fields = [truth[selected, time_id], static[selected, time_id], nested[selected, time_id]]
        errors = [
            np.abs(static[selected, time_id] - truth[selected, time_id]),
            np.abs(nested[selected, time_id] - truth[selected, time_id]),
        ]
        improvement = errors[0] - errors[1]
        static_l2 = np.linalg.norm(static[selected, time_id] - truth[selected, time_id]) / max(
            np.linalg.norm(truth[selected, time_id]), 1.0e-8
        )
        nested_l2 = np.linalg.norm(nested[selected, time_id] - truth[selected, time_id]) / max(
            np.linalg.norm(truth[selected, time_id]), 1.0e-8
        )
        reduction = 100.0 * (static_l2 - nested_l2) / max(static_l2, 1.0e-8)
        column_payload = [fields[0], fields[1], fields[2], errors[0], errors[1], improvement]
        for column, value in enumerate(column_payload):
            if column in (0, 1, 2):
                image = axes[row, column].imshow(
                    value,
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-1.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                axes[row, column].contour(value, levels=[0.0], colors=NAVY, linewidths=0.65)
                field_images.append(image)
            elif column in (3, 4):
                image = axes[row, column].imshow(
                    value,
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=error_max,
                    interpolation="nearest",
                )
                error_images.append(image)
            else:
                image = axes[row, column].imshow(
                    value,
                    origin="lower",
                    cmap="RdBu",
                    vmin=-difference_max,
                    vmax=difference_max,
                    interpolation="nearest",
                )
                difference_images.append(image)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(titles[column], fontproperties=FontProperties(fname=FONT_PATH, size=16, weight="bold"))
        axes[row, 3].text(
            0.04,
            0.94,
            rf"$L_2={static_l2:.4f}$",
            transform=axes[row, 3].transAxes,
            va="top",
            fontsize=12,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#202020", "edgecolor": "none", "alpha": 0.78},
        )
        axes[row, 4].text(
            0.04,
            0.94,
            rf"$L_2={nested_l2:.4f}$" + "\n" + rf"$\downarrow {reduction:.1f}\%$",
            transform=axes[row, 4].transAxes,
            va="top",
            fontsize=12,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#1D5B45", "edgecolor": "none", "alpha": 0.84},
        )
        axes[row, 0].set_ylabel(
            rf"$\tau={times[time_id]:.2f}$" + "\n" + rf"active rank $={trace[time_id]}$",
            fontsize=15,
        )
    field_bar = fig.colorbar(field_images[0], ax=axes[:, (0, 1, 2)], shrink=0.80, pad=0.012)
    field_bar.set_label(r"相场 $u$", fontproperties=FontProperties(fname=FONT_PATH, size=14))
    field_bar.ax.tick_params(labelsize=12)
    error_bar = fig.colorbar(error_images[0], ax=axes[:, (3, 4)], shrink=0.80, pad=0.012)
    error_bar.set_label("绝对误差", fontproperties=FontProperties(fname=FONT_PATH, size=14))
    error_bar.ax.tick_params(labelsize=12)
    difference_bar = fig.colorbar(difference_images[0], ax=axes[:, 5], shrink=0.80, pad=0.012)
    difference_bar.set_label("rank-7 误差 $-$ ST-NLR 误差", fontproperties=FontProperties(fname=FONT_PATH, size=13))
    difference_bar.ax.tick_params(labelsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


def plot_dendrite_comparison(
    truth: np.ndarray,
    static: np.ndarray,
    nested: np.ndarray,
    selected: int,
    terminal_time: float,
    terminal_rank: int,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "mathtext.fontset": "stixsans",
        }
    )
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(12.4, 5.25),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 0.94, 1, 0.94]},
    )
    titles = ["参考", "continued-static", "static 绝对误差", "ST-NLR", "ST-NLR 绝对误差"]
    state_ranges = [(-1.0, 1.0, "RdBu_r"), (-0.32, 0.02, "magma")]
    row_labels = [
        rf"相场 $\phi$" + "\n" + rf"$t={terminal_time:.0f}$, rank $={terminal_rank}$",
        r"温度场 $U$",
    ]
    for row in range(2):
        reference = truth[selected, row]
        static_field = static[selected, row]
        nested_field = nested[selected, row]
        static_error = np.abs(static_field - reference)
        nested_error = np.abs(nested_field - reference)
        error_max = max(
            float(np.quantile(static_error, 0.995)),
            float(np.quantile(nested_error, 0.995)),
            1.0e-4,
        )
        values = [reference, static_field, static_error, nested_field, nested_error]
        state_image = error_image = None
        for column, value in enumerate(values):
            if column in (0, 1, 3):
                vmin, vmax, cmap = state_ranges[row]
                state_image = axes[row, column].imshow(
                    value,
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    interpolation="nearest",
                )
                if row == 0:
                    axes[row, column].contour(value, levels=[0.0], colors=NAVY, linewidths=0.7)
            else:
                error_image = axes[row, column].imshow(
                    value,
                    origin="lower",
                    vmin=0.0,
                    vmax=error_max,
                    cmap="magma",
                    interpolation="nearest",
                )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(
                    titles[column],
                    fontproperties=FontProperties(fname=FONT_PATH, size=16, weight="bold"),
                )
        static_l2 = np.linalg.norm((static_field - reference).ravel()) / max(
            np.linalg.norm(reference.ravel()), 1.0e-8
        )
        nested_l2 = np.linalg.norm((nested_field - reference).ravel()) / max(
            np.linalg.norm(reference.ravel()), 1.0e-8
        )
        reduction = 100.0 * (static_l2 - nested_l2) / max(static_l2, 1.0e-8)
        axes[row, 4].text(
            0.04, 0.94, rf"$L_2$ 误差 $\downarrow$ {reduction:.1f}%",
            transform=axes[row, 4].transAxes, ha="left", va="top", color="white",
            fontproperties=FontProperties(fname=FONT_PATH, size=12.2, weight="bold"),
            bbox=dict(boxstyle="round,pad=0.25", facecolor=TEAL, edgecolor="none", alpha=0.92),
        )
        axes[row, 0].set_ylabel(row_labels[row], fontproperties=FontProperties(fname=FONT_PATH, size=15))
        state_bar = fig.colorbar(state_image, ax=axes[row, (0, 1, 3)], shrink=0.82, pad=0.014)
        state_bar.ax.tick_params(labelsize=11)
        error_bar = fig.colorbar(error_image, ax=axes[row, (2, 4)], shrink=0.82, pad=0.014)
        error_bar.ax.tick_params(labelsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


def run_dendrite(device: torch.device) -> dict:
    data_path = ROOT / "data/full/pfhub3/shifted.npz"
    static_checkpoint = ROOT / "checkpoints/pfhub3/continued_static_shifted_seed0.pt"
    nested_checkpoint = ROOT / "checkpoints/pfhub3/stnlr_shifted_seed0.pt"
    metrics_path = ROOT / "results/figure_metadata/dendrite_representative_states.json"
    output = OUT / "dendrite_representative_states.pdf"
    payload = np.load(data_path)
    test_start, test_count = 24, 8
    initial = payload["initial"][test_start : test_start + test_count]
    truth = payload["fields"][test_start : test_start + test_count, -1]
    metrics = json.loads(metrics_path.read_text())
    trace = metrics["rank_trace"]
    terminal_rank = int(trace[-1])

    static_model = load_model("static", static_checkpoint, device, channels=2)
    static_prediction = predict_dendrite_terminal(
        static_model, "static", initial, float(payload["times"][-1]), terminal_rank, device
    )
    del static_model
    torch.cuda.empty_cache()
    nested_model = load_model("stnlr", nested_checkpoint, device, channels=2)
    nested_prediction = predict_dendrite_terminal(
        nested_model, "stnlr", initial, float(payload["times"][-1]), terminal_rank, device
    )
    del nested_model
    torch.cuda.empty_cache()

    def channel_relative(prediction, target, channel):
        numerator = np.linalg.norm((prediction[:, channel] - target[:, channel]).reshape(len(target), -1), axis=1)
        denominator = np.linalg.norm(target[:, channel].reshape(len(target), -1), axis=1)
        return numerator / np.maximum(denominator, 1.0e-8)

    static_phase = channel_relative(static_prediction, truth, 0)
    nested_phase = channel_relative(nested_prediction, truth, 0)
    static_temperature = channel_relative(static_prediction, truth, 1)
    nested_temperature = channel_relative(nested_prediction, truth, 1)
    combined_gain = 0.5 * (
        (static_phase - nested_phase) + (static_temperature - nested_temperature)
    )
    eligible = np.flatnonzero(
        (static_phase > nested_phase) & (static_temperature > nested_temperature)
    )
    if len(eligible) == 0:
        eligible = np.flatnonzero(combined_gain > 0)
    if len(eligible) == 0:
        eligible = np.arange(test_count)
    median_gain = np.median(combined_gain[eligible])
    selected = int(eligible[np.argmin(np.abs(combined_gain[eligible] - median_gain))])
    plot_dendrite_comparison(
        truth,
        static_prediction,
        nested_prediction,
        selected,
        float(payload["times"][-1]),
        terminal_rank,
        output,
    )
    result = {
        "task": "dendrite",
        "condition": "shifted",
        "seed": 0,
        "test_start": test_start,
        "test_count": test_count,
        "selected_local_index": selected,
        "selected_global_index": test_start + selected,
        "selection_rule": "closest to median combined terminal improvement among samples with both phase and temperature improvements",
        "eligible_count": int(((static_phase > nested_phase) & (static_temperature > nested_temperature)).sum()),
        "selected_static_phase_terminal_relative_l2": float(static_phase[selected]),
        "selected_stnlr_phase_terminal_relative_l2": float(nested_phase[selected]),
        "selected_static_temperature_terminal_relative_l2": float(static_temperature[selected]),
        "selected_stnlr_temperature_terminal_relative_l2": float(nested_temperature[selected]),
        "rank_trace": list(map(int, trace)),
        "output": output.relative_to(ROOT).as_posix(),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_task(task: str, device: torch.device) -> dict:
    if task == "allen":
        data_path = ROOT / "data/full/allen_cahn/eps0022_lam14.npz"
        static_checkpoint = ROOT / "checkpoints/allen_cahn/static_rank7_eps0022_lam14_seed0.pt"
        nested_checkpoint = ROOT / "checkpoints/allen_cahn/stnlr_eps0022_lam14_seed0.pt"
        trace_path = ROOT / "results/figure_metadata/allen_cahn_rank7_representative_states.json"
        output = OUT / "allen_cahn_rank7_representative_states.pdf"
        time_ids = (3, 10)
    elif task == "cahn":
        data_path = ROOT / "data/full/cahn_hilliard/eps0020_lam10.npz"
        static_checkpoint = ROOT / "checkpoints/cahn_hilliard/continued_static_eps0020_lam10_seed0.pt"
        nested_checkpoint = ROOT / "checkpoints/cahn_hilliard/stnlr_eps0020_lam10_seed0.pt"
        trace_path = nested_checkpoint
        output = OUT / "cahn_hilliard_representative_states.pdf"
        time_ids = (5, 10)
    else:
        raise ValueError(task)

    payload = np.load(data_path)
    test_start, test_count = 120, 40
    initial = payload["initial"][test_start : test_start + test_count]
    truth = payload["fields"][test_start : test_start + test_count]
    times = payload["times"]
    if task == "allen":
        trace = json.loads(trace_path.read_text())["rank_trace"]
    else:
        trace = torch.load(trace_path, map_location="cpu")["rank_trace"]

    static_model = load_model("static", static_checkpoint, device)
    static_prediction = predict(static_model, "static", initial, times, None, device)
    del static_model
    torch.cuda.empty_cache()
    nested_model = load_model("stnlr", nested_checkpoint, device)
    nested_prediction = predict(nested_model, "stnlr", initial, times, trace, device)
    del nested_model
    torch.cuda.empty_cache()

    selected, static_rel, nested_rel, trajectory_gain, terminal_gain = representative_index(
        static_prediction, nested_prediction, truth
    )
    plot_comparison(
        truth,
        static_prediction,
        nested_prediction,
        times,
        trace,
        selected,
        time_ids,
        output,
    )
    result = {
        "task": task,
        "target": data_path.parent.parent.name if task == "allen" else data_path.parent.name,
        "seed": 0,
        "test_start": test_start,
        "test_count": test_count,
        "selected_local_index": selected,
        "selected_global_index": test_start + selected,
        "selection_rule": "closest to the median positive trajectory improvement among samples with positive trajectory and terminal improvements",
        "eligible_count": int(((trajectory_gain > 0) & (terminal_gain > 0)).sum()),
        "selected_static_trajectory_relative_l2": float(static_rel[selected, 1:].mean()),
        "selected_stnlr_trajectory_relative_l2": float(nested_rel[selected, 1:].mean()),
        "selected_static_terminal_relative_l2": float(static_rel[selected, -1]),
        "selected_stnlr_terminal_relative_l2": float(nested_rel[selected, -1]),
        "rank_trace": list(map(int, trace)),
        "output": output.relative_to(ROOT).as_posix(),
    }
    metadata = ROOT / "results/figure_metadata" / output.with_suffix(".json").name
    metadata.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("allen", "cahn", "dendrite"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    result = run_dendrite(device) if args.task == "dendrite" else run_task(args.task, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
