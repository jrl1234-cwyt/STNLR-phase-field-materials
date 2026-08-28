#!/usr/bin/env python3
"""Aggregate the two-condition, three-seed dendrite transfer experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "phase_trajectory_relative_l2",
    "phase_terminal_relative_l2",
    "temperature_trajectory_relative_l2",
    "temperature_terminal_relative_l2",
    "solid_fraction_mae",
    "free_energy_relative_error",
    "tip_position_mae",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for target in ("standard", "shifted"):
        for seed in range(3):
            base = args.root / target / f"seed{seed}"
            static = json.loads((base / "static_material_v2_600/metrics.json").read_text())
            nested = json.loads((base / "stnlr_material_v2_600/metrics.json").read_text())
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "static": static["test_metrics"],
                    "stnlr": nested["test_calibrated_metrics"],
                    "rank_trace": nested["validation_selected_rank_trace"],
                    "mean_active_rank": nested["mean_active_rank"],
                }
            )

    summary = {}
    for metric in METRICS:
        static_seed_means = []
        nested_seed_means = []
        for seed in range(3):
            selected = [row for row in rows if row["seed"] == seed]
            static_seed_means.append(np.mean([row["static"][metric] for row in selected]))
            nested_seed_means.append(np.mean([row["stnlr"][metric] for row in selected]))
        static_values = np.asarray(static_seed_means)
        nested_values = np.asarray(nested_seed_means)
        paired = [(row["static"][metric], row["stnlr"][metric]) for row in rows]
        summary[metric] = {
            "continued_static_mean": float(static_values.mean()),
            "continued_static_sd": float(static_values.std(ddof=1)),
            "stnlr_mean": float(nested_values.mean()),
            "stnlr_sd": float(nested_values.std(ddof=1)),
            "relative_change_percent": float(100.0 * (nested_values.mean() / static_values.mean() - 1.0)),
            "paired_wins_out_of_6": int(sum(nested < static for static, nested in paired)),
        }
    ranks_by_seed = []
    for seed in range(3):
        ranks_by_seed.append(np.mean([row["mean_active_rank"] for row in rows if row["seed"] == seed]))
    result = {
        "protocol": {
            "targets": ["PFHub-3 standard", "parameter-shifted anisotropic dendrite"],
            "seeds": [0, 1, 2],
            "pairs": 6,
            "comparison": "same static checkpoint plus matched 600-step material calibration",
        },
        "metrics": summary,
        "mean_active_rank": {
            "mean": float(np.mean(ranks_by_seed)),
            "sd": float(np.std(ranks_by_seed, ddof=1)),
            "static": 16.0,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"metrics": summary, "mean_active_rank": result["mean_active_rank"]}, indent=2))


if __name__ == "__main__":
    main()
