#!/usr/bin/env python3
"""Hierarchical paired bootstrap for the final Allen--Cahn main comparison.

The hierarchy follows target condition -> training seed -> paired test
trajectory.  Predictions are recomputed from the exact final checkpoints and
the frozen validation rank traces used in the manuscript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_allen_cahn_conditional_trajectory import free_energy
from train_evaluate_poseidon_allen_cahn import build_poseidon, load_data, set_nested_time
from train_poseidon_physics_calibrated_nested import force_rank


TARGETS = ("eps0022_lam14", "eps0022_lam16", "eps0028_lam14", "eps0028_lam16")
SEEDS = (0, 1, 2)
METRICS = (
    "trajectory_relative_l2",
    "terminal_relative_l2",
    "terminal_interface_fraction_mae",
    "terminal_free_energy_relative_error",
)


def data_path(repo: Path, target: str) -> Path:
    if target == "eps0022_lam16":
        return repo / "results_allen_cahn_conditional_trajectory_20260816/data/target_trajectories.npz"
    return repo / f"results_allen_cahn_multitarget_transfer_20260817/targets/{target}/data/target_trajectories.npz"


def build(kind: str, device: torch.device):
    config = SimpleNamespace(
        kind=kind,
        nested_schedule="u_shaped",
        u_mid_rank=8,
        poseidon_code=Path("/tmp/poseidon-stnlr"),
        poseidon_checkpoint=Path("/tmp/poseidon_model"),
    )
    model, _ = build_poseidon(config, device)
    return model


def load_model(kind: str, checkpoint: Path, device: torch.device):
    model = build(kind, device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict(model, initial, times, batch_size: int, rank_trace: list[int] | None):
    outputs = []
    nested = rank_trace is not None
    for begin in range(0, initial.shape[0], batch_size):
        end = min(initial.shape[0], begin + batch_size)
        rows = []
        for time_id, physical_time in enumerate(times):
            lead = (physical_time / times[-1]).expand(end - begin)
            if nested:
                set_nested_time(model, lead, "decay")
                force_rank(model, rank_trace[time_id])
            rows.append(model(pixel_values=initial[begin:end], time=lead).output)
        outputs.append(torch.stack(rows, dim=1))
    return torch.cat(outputs)


def per_sample_metrics(predicted, target, epsilon, reaction):
    error = (predicted - target).flatten(2)
    target_flat = target.flatten(2)
    relative = torch.linalg.vector_norm(error, dim=2) / torch.linalg.vector_norm(target_flat, dim=2).clamp_min(1.0e-8)
    interface_pred = (predicted.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    interface_target = (target.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    energy_pred = free_energy(predicted[:, -1], epsilon, reaction)
    energy_target = free_energy(target[:, -1], epsilon, reaction)

    return {
        "trajectory_relative_l2": relative.mean(1).cpu().numpy(),
        "terminal_relative_l2": relative[:, -1].cpu().numpy(),
        "terminal_interface_fraction_mae": (interface_pred[:, -1] - interface_target[:, -1]).abs().cpu().numpy(),
        "terminal_free_energy_relative_error": ((energy_pred - energy_target).abs() / energy_target.abs().clamp_min(1.0e-8)).cpu().numpy(),
    }


def collect(repo: Path, static_root: Path, device: torch.device, batch_size: int):
    records = {}
    strict_root = repo / "results_poseidon_t_allen_cahn_formal_20260817/physics_calibrated_selfdistill_test_strict2"
    trained_root = repo / "results_poseidon_t_allen_cahn_formal_20260817/physics_calibrated_selfdistill"
    for target in TARGETS:
        data = load_data(data_path(repo, target))
        start, count = 120, 40
        initial = torch.from_numpy(data["initial"][start:start + count]).to(device).unsqueeze(1)
        truth = torch.from_numpy(data["fields"][start:start + count]).to(device).unsqueeze(2)
        epsilon = torch.from_numpy(data["epsilon"][start:start + count]).to(device)
        reaction = torch.from_numpy(data["reaction"][start:start + count]).to(device)
        times = torch.from_numpy(data["times"]).to(device)
        for seed in SEEDS:
            strict = json.loads((strict_root / target / f"seed{seed}/metrics.json").read_text())
            trace = [int(rank) for rank in strict["rank_trace"]]
            static = load_model("static", static_root / target / f"seed{seed}/final.pt", device)
            static_prediction = predict(static, initial, times, batch_size, None)
            del static
            torch.cuda.empty_cache()
            nested = load_model("stnlr", trained_root / target / f"seed{seed}/final.pt", device)
            nested_prediction = predict(nested, initial, times, batch_size, trace)
            del nested
            torch.cuda.empty_cache()
            records[f"{target}/seed{seed}"] = {
                "target": target,
                "seed": seed,
                "rank_trace": trace,
                "continued_static": {
                    key: values.tolist() for key, values in per_sample_metrics(
                        static_prediction, truth, epsilon, reaction
                    ).items()
                },
                "stnlr": {
                    key: values.tolist() for key, values in per_sample_metrics(
                        nested_prediction, truth, epsilon, reaction
                    ).items()
                },
            }
            print(f"evaluated {target} seed{seed}", flush=True)
    return records


def hierarchical_bootstrap(records: dict, repetitions: int, random_seed: int):
    rng = np.random.default_rng(random_seed)
    reductions = {metric: np.empty(repetitions, dtype=np.float64) for metric in METRICS}
    observed = {}
    pair_rows = []
    for target in TARGETS:
        for seed in SEEDS:
            row = records[f"{target}/seed{seed}"]
            pair = {"target": target, "seed": seed}
            for metric in METRICS:
                baseline = np.mean(row["continued_static"][metric])
                stnlr = np.mean(row["stnlr"][metric])
                pair[metric] = {
                    "continued_static": float(baseline),
                    "stnlr": float(stnlr),
                    "relative_reduction_percent": float(100.0 * (baseline - stnlr) / baseline),
                }
            pair_rows.append(pair)
    for metric in METRICS:
        baseline = np.mean([row[metric]["continued_static"] for row in pair_rows])
        stnlr = np.mean([row[metric]["stnlr"] for row in pair_rows])
        differences = np.asarray([row[metric]["continued_static"] - row[metric]["stnlr"] for row in pair_rows])
        observed[metric] = {
            "continued_static_mean": float(baseline),
            "stnlr_mean": float(stnlr),
            "relative_reduction_percent": float(100.0 * (baseline - stnlr) / baseline),
            "paired_wins_out_of_12": int(np.sum(differences > 0)),
            "standardized_paired_effect": float(differences.mean() / differences.std(ddof=1)),
        }

    for rep in range(repetitions):
        selected_targets = rng.integers(0, len(TARGETS), size=len(TARGETS))
        sums = {metric: [0.0, 0.0] for metric in METRICS}
        cells = 0
        for target_id in selected_targets:
            target = TARGETS[int(target_id)]
            selected_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
            for seed_id in selected_seeds:
                row = records[f"{target}/seed{SEEDS[int(seed_id)]}"]
                sample_count = len(row["stnlr"][METRICS[0]])
                selected_samples = rng.integers(0, sample_count, size=sample_count)
                for metric in METRICS:
                    baseline = np.asarray(row["continued_static"][metric])[selected_samples].mean()
                    stnlr = np.asarray(row["stnlr"][metric])[selected_samples].mean()
                    sums[metric][0] += baseline
                    sums[metric][1] += stnlr
                cells += 1
        for metric in METRICS:
            baseline = sums[metric][0] / cells
            stnlr = sums[metric][1] / cells
            reductions[metric][rep] = 100.0 * (baseline - stnlr) / baseline

    for metric in METRICS:
        low, high = np.quantile(reductions[metric], [0.025, 0.975])
        observed[metric]["hierarchical_bootstrap_95ci_percent"] = [float(low), float(high)]
        observed[metric]["bootstrap_probability_of_improvement"] = float(np.mean(reductions[metric] > 0.0))
    return observed, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--static-root",
        type=Path,
        default=Path("results_poseidon_static_continuation_20260823"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()
    device = torch.device(args.device)
    records = collect(args.repo, args.static_root, device, args.batch_size)
    inference, pair_rows = hierarchical_bootstrap(records, args.bootstrap_repetitions, args.bootstrap_seed)
    result = {
        "protocol": {
            "comparison": "selected continued-static rank-16 root vs final validation-frozen ST-NLR",
            "static_root": str(args.static_root),
            "hierarchy": "target condition -> training seed -> paired test trajectory",
            "targets": list(TARGETS),
            "training_seeds": list(SEEDS),
            "test_trajectories_per_cell": 40,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "inference": inference,
        "pair_rows": pair_rows,
        "per_sample_records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(inference, indent=2))


if __name__ == "__main__":
    main()
