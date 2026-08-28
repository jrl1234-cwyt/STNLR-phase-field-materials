#!/usr/bin/env python3
"""Aggregate four-condition, three-seed Cahn--Hilliard transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGETS = ("eps0020_lam10", "eps0020_lam14", "eps0026_lam10", "eps0026_lam14")
METRICS = (
    "trajectory_relative_l2_mean",
    "terminal_relative_l2_mean",
    "trajectory_mass_drift_mae",
    "maximum_mass_drift_mean",
    "terminal_free_energy_relative_error_mean",
    "terminal_structure_factor_centroid_relative_error_mean",
    "trajectory_pde_residual_relative_rms",
)


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def seed_macro(rows, method, metric):
    return mean_sd([
        np.mean([row[method][metric] for row in rows if row["seed"] == seed])
        for seed in range(3)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for target in TARGETS:
        for seed in range(3):
            static = json.loads((args.root / target / "static_rank16" / f"seed{seed}" / "metrics.json").read_text())
            nested = json.loads((args.root / target / "stnlr_nested" / f"seed{seed}" / "metrics.json").read_text())
            rows.append({
                "target": target,
                "seed": seed,
                "static": static["test_metrics"],
                "stnlr": nested["test_calibrated_metrics"],
                "stnlr_rank16": nested["test_rank16_metrics"],
                "rank_trace": nested["validation_selected_rank_trace"],
                "mean_active_rank": nested["mean_active_rank"],
            })
    methods = ("static", "stnlr_rank16", "stnlr")
    summary = {method: {metric: seed_macro(rows, method, metric) for metric in METRICS} for method in methods}
    result = {
        "pairs": len(rows),
        "summary": summary,
        "stnlr_vs_static": {
            metric: {
                "relative_change_percent": float(100.0 * (summary["stnlr"][metric]["mean"] - summary["static"][metric]["mean"]) / summary["static"][metric]["mean"]),
                "paired_wins": int(sum(row["stnlr"][metric] < row["static"][metric] for row in rows)),
            }
            for metric in METRICS
        },
        "mean_active_rank": mean_sd([row["mean_active_rank"] for row in rows]),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
