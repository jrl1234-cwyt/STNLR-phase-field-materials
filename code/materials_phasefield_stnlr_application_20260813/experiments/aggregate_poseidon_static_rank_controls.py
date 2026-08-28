#!/usr/bin/env python3
"""Aggregate the predeclared static rank-4/7/8 Allen--Cahn controls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TARGETS = (
    "eps0022_lam14",
    "eps0022_lam16",
    "eps0028_lam14",
    "eps0028_lam16",
)
SEEDS = (0, 1, 2)
RANKS = (4, 7, 8)
METRICS = (
    "trajectory_relative_l2_mean",
    "terminal_relative_l2_mean",
    "terminal_interface_fraction_mae",
    "terminal_free_energy_relative_error_mean",
    "trajectory_pde_residual_relative_rms",
)


def seed_aggregate(rows, metric):
    values = []
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        values.append(statistics.mean(row["metrics"][metric] for row in selected))
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "per_seed_target_mean": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--stnlr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_method = {}
    for rank in RANKS:
        rows = []
        for target in TARGETS:
            for seed in SEEDS:
                payload = json.loads(
                    (args.control_root / f"static_rank{rank}" / target /
                     f"seed{seed}" / "metrics.json").read_text()
                )
                rows.append({
                    "target": target,
                    "seed": seed,
                    "metrics": payload["after_metrics"],
                    "trainable_parameters": payload["trainable_parameters"],
                })
        rows_by_method[f"static_rank{rank}"] = rows

    stnlr_rows = []
    for target in TARGETS:
        for seed in SEEDS:
            payload = json.loads(
                (args.stnlr_root / target / f"seed{seed}" / "metrics.json").read_text()
            )
            stnlr_rows.append({
                "target": target,
                "seed": seed,
                "metrics": payload["metrics"],
                "mean_active_rank": payload["mean_active_rank"],
            })
    rows_by_method["stnlr"] = stnlr_rows

    aggregate = {
        method: {metric: seed_aggregate(rows, metric) for metric in METRICS}
        for method, rows in rows_by_method.items()
    }
    comparisons = {}
    st_lookup = {
        (row["target"], row["seed"]): row["metrics"] for row in stnlr_rows
    }
    for rank in RANKS:
        method = f"static_rank{rank}"
        base_lookup = {
            (row["target"], row["seed"]): row["metrics"]
            for row in rows_by_method[method]
        }
        comparisons[method] = {}
        for metric in METRICS:
            base_mean = aggregate[method][metric]["mean"]
            st_mean = aggregate["stnlr"][metric]["mean"]
            comparisons[method][metric] = {
                "stnlr_relative_change_percent": 100.0 * (st_mean / base_mean - 1.0),
                "stnlr_paired_wins_out_of_12": sum(
                    st_lookup[key][metric] < base_lookup[key][metric]
                    for key in base_lookup
                ),
            }
        comparisons[method]["all_five_paired_dominance_out_of_12"] = sum(
            all(st_lookup[key][metric] < base_lookup[key][metric] for metric in METRICS)
            for key in base_lookup
        )

    result = {
        "protocol": {
            "targets": list(TARGETS),
            "seeds": list(SEEDS),
            "static_ranks": list(RANKS),
            "pairs": len(TARGETS) * len(SEEDS),
            "steps": 600,
            "reporting": "all predeclared ranks, targets, seeds, and metrics",
        },
        "trainable_parameters": {
            method: rows[0].get("trainable_parameters")
            for method, rows in rows_by_method.items()
            if method.startswith("static_rank")
        },
        "stnlr_mean_active_rank": {
            "mean": statistics.mean(row["mean_active_rank"] for row in stnlr_rows),
            "seed_sd": statistics.stdev(
                statistics.mean(
                    row["mean_active_rank"] for row in stnlr_rows
                    if row["seed"] == seed
                )
                for seed in SEEDS
            ),
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "rows": rows_by_method,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "trainable_parameters": result["trainable_parameters"],
        "stnlr_mean_active_rank": result["stnlr_mean_active_rank"],
        "aggregate": aggregate,
        "comparisons": comparisons,
    }, indent=2))


if __name__ == "__main__":
    main()
