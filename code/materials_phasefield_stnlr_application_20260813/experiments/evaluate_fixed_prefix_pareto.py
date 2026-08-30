#!/usr/bin/env python3
"""Reproduce the fixed-prefix quality--capacity audit in manuscript Figure 5.

The audit uses the released Allen--Cahn checkpoint for the representative
condition and seed.  The physical-time rank trace is read from the archived
validation-frozen main-experiment record.  Rank 4, rank 8, rank 16, and the
frozen dynamic trace are then evaluated on both the validation and test splits.
No model parameters are updated by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_hierarchical_bootstrap_allen_cahn import (
    METRICS,
    load_model,
    per_sample_metrics,
    predict,
)
from evaluate_allen_cahn_conditional_trajectory import free_energy
from train_evaluate_poseidon_allen_cahn import load_data


RANKS = (4, 8, 16)
ADDITIVE_CONSTANTS = {
    "relative_l2": 1.0e-4,
    "interface_mae": 1.0e-4,
    "energy_relative_error": 2.0e-4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timewise_metrics(predicted, target, epsilon, reaction):
    difference = (predicted - target).flatten(2)
    denominator = torch.linalg.vector_norm(target.flatten(2), dim=2).clamp_min(1.0e-8)
    relative_l2 = torch.linalg.vector_norm(difference, dim=2) / denominator
    predicted_interface = (predicted.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    target_interface = (target.abs() < 0.2).float().mean(dim=(-3, -2, -1))
    interface_mae = (predicted_interface - target_interface).abs()
    energy_error = []
    for time_id in range(predicted.shape[1]):
        predicted_energy = free_energy(predicted[:, time_id], epsilon, reaction)
        target_energy = free_energy(target[:, time_id], epsilon, reaction)
        energy_error.append(
            (predicted_energy - target_energy).abs()
            / target_energy.abs().clamp_min(1.0e-8)
        )
    return {
        "relative_l2": relative_l2.mean(0),
        "interface_mae": interface_mae.mean(0),
        "energy_relative_error": torch.stack(energy_error, dim=1).mean(0),
    }


def aggregate(predicted, target, epsilon, reaction):
    per_sample = per_sample_metrics(predicted, target, epsilon, reaction)
    return {
        "mean": {metric: float(np.mean(per_sample[metric])) for metric in METRICS},
        "per_sample": {metric: values.tolist() for metric, values in per_sample.items()},
    }


def select_trace(validation_by_rank):
    reference = validation_by_rank[16]
    trace = []
    feasibility = []
    for time_id in range(len(reference["relative_l2"])):
        row = {"time_index": time_id, "candidates": {}}
        selected = 16
        for rank in RANKS:
            checks = {}
            for metric, additive in ADDITIVE_CONSTANTS.items():
                candidate = float(validation_by_rank[rank][metric][time_id])
                baseline = float(reference[metric][time_id])
                limit = 1.02 * baseline + additive
                checks[metric] = {
                    "value": candidate,
                    "rank16_reference": baseline,
                    "limit": limit,
                    "feasible": candidate <= limit,
                }
            feasible = all(item["feasible"] for item in checks.values())
            row["candidates"][str(rank)] = {"metrics": checks, "feasible": feasible}
            if feasible and selected == 16:
                selected = rank
        row["selected_rank"] = selected
        feasibility.append(row)
        trace.append(selected)
    return trace, feasibility


def main() -> None:
    parser = argparse.ArgumentParser()
    bundle = Path(__file__).resolve().parents[3]
    parser.add_argument("--bundle", type=Path, default=bundle)
    parser.add_argument("--target", default="eps0022_lam14")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data_path = args.bundle / f"data/full/allen_cahn/{args.target}.npz"
    checkpoint_path = (
        args.bundle / f"checkpoints/allen_cahn/stnlr_{args.target}_seed{args.seed}.pt"
    )
    archived_path = args.bundle / "results/allen_cahn_bootstrap.json"
    output_path = args.output or args.bundle / "results/fixed_prefix_pareto.json"

    data = load_data(data_path)
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    record_key = f"{args.target}/seed{args.seed}"
    archived_trace = [
        int(rank) for rank in archived["per_sample_records"][record_key]["rank_trace"]
    ]

    device = torch.device(args.device)
    model = load_model("stnlr", checkpoint_path, device)
    times = torch.from_numpy(data["times"]).to(device)
    predictions = {"validation": {}, "test": {}}
    split_spec = {"validation": (100, 20), "test": (120, 40)}
    targets = {}
    material_parameters = {}

    for split, (start, count) in split_spec.items():
        initial = torch.from_numpy(data["initial"][start:start + count]).to(device).unsqueeze(1)
        targets[split] = torch.from_numpy(data["fields"][start:start + count]).to(device).unsqueeze(2)
        material_parameters[split] = (
            torch.from_numpy(data["epsilon"][start:start + count]).to(device),
            torch.from_numpy(data["reaction"][start:start + count]).to(device),
        )
        for rank in RANKS:
            predictions[split][str(rank)] = predict(
                model, initial, times, args.batch_size, [rank] * len(times)
            )
        predictions[split]["dynamic"] = predict(
            model, initial, times, args.batch_size, archived_trace
        )

    validation_timewise = {
        rank: timewise_metrics(
            predictions["validation"][str(rank)],
            targets["validation"],
            *material_parameters["validation"],
        )
        for rank in RANKS
    }
    recomputed_trace, feasibility = select_trace(validation_timewise)
    if recomputed_trace != archived_trace:
        raise RuntimeError(
            "The recomputed validation trace does not match the archived frozen trace: "
            f"{recomputed_trace} != {archived_trace}"
        )

    split_results = {}
    for split in ("validation", "test"):
        split_results[split] = {}
        for method in ("4", "8", "16", "dynamic"):
            split_results[split][method] = aggregate(
                predictions[split][method],
                targets[split],
                *material_parameters[split],
            )

    test_reference = split_results["test"]["16"]["mean"]
    strategies = {}
    for method, label, mean_rank in (
        ("dynamic", "Dynamic ST-NLR", float(np.mean(archived_trace))),
        ("4", "All rank 4", 4.0),
        ("8", "All rank 8", 8.0),
        ("16", "All rank 16", 16.0),
    ):
        values = split_results["test"][method]["mean"]
        increases = {
            metric: 100.0 * (values[metric] / test_reference[metric] - 1.0)
            for metric in METRICS
        }
        maximum_increase = max(increases.values())
        if method == "dynamic":
            validation_checks = [
                row["candidates"][str(archived_trace[row["time_index"]])]["metrics"]
                for row in feasibility
            ]
        else:
            validation_checks = [row["candidates"][method]["metrics"] for row in feasibility]
        maximum_tolerance_utilization = max(
            metric_row[metric]["value"] / metric_row[metric]["limit"]
            for metric_row in validation_checks
            for metric in ADDITIVE_CONSTANTS
        )
        strategies[method] = {
            "label": label,
            "mean_active_rank": mean_rank,
            "active_capacity_saving_percent": 100.0 * (1.0 - mean_rank / 16.0),
            "test_metrics": values,
            "test_error_change_vs_rank16_percent": increases,
            "worst_case_error_increase_percent": maximum_increase,
            "worst_case_material_quality_retention_percent": 100.0 - maximum_increase,
            "maximum_validation_tolerance_utilization_percent": (
                100.0 * maximum_tolerance_utilization
            ),
            "validation_feasible_under_timewise_2_percent_rule": (
                method == "16"
                or method == "dynamic"
                or all(
                    row["candidates"][method]["feasible"] for row in feasibility
                )
            ),
        }

    result = {
        "status": "complete",
        "protocol": {
            "purpose": "Fixed-prefix and dynamic-prefix audit for manuscript Figure 5",
            "target": args.target,
            "training_seed": args.seed,
            "candidate_ranks": list(RANKS),
            "adapt_split": [0, 100],
            "validation_split": [100, 120],
            "test_split": [120, 160],
            "selection_tolerance": 0.02,
            "selection_additive_constants": ADDITIVE_CONSTANTS,
            "quality_retention_definition": (
                "100 minus the largest test-error increase in percent relative to all rank 16 "
                "over trajectory L2, terminal L2, terminal interface MAE, and terminal free-energy error"
            ),
            "checkpoint": str(checkpoint_path.relative_to(args.bundle)),
            "checkpoint_sha256": sha256(checkpoint_path),
            "data": str(data_path.relative_to(args.bundle)),
            "data_sha256": sha256(data_path),
            "archived_rank_trace_source": str(archived_path.relative_to(args.bundle)),
        },
        "validation_selected_rank_trace": archived_trace,
        "mean_active_rank": float(np.mean(archived_trace)),
        "validation_timewise_feasibility": feasibility,
        "split_results": split_results,
        "strategies": strategies,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"rank_trace": archived_trace, "strategies": strategies}, indent=2))


if __name__ == "__main__":
    main()
