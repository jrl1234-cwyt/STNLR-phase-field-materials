#!/usr/bin/env python3
"""Aggregate the predeclared three-seed Allen--Cahn objective ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ("full_stnlr", "no_field_distillation", "no_spectral_distillation", "no_material_objectives")
METRICS = (
    "trajectory_relative_l2_mean",
    "terminal_relative_l2_mean",
    "terminal_interface_fraction_mae",
    "terminal_free_energy_relative_error_mean",
    "trajectory_pde_residual_relative_rms",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    target = "eps0022_lam14"
    rows = []
    for seed in range(3):
        paths = {}
        for variant in VARIANTS:
            paths[variant] = args.result_root / "objective_ablation" / target / f"seed{seed}" / variant / "metrics.json"
        if not all(path.is_file() for path in paths.values()):
            print(f"aggregate deferred: incomplete seed {seed}")
            return
        for variant, path in paths.items():
            payload = json.loads(path.read_text())
            rows.append({
                "seed": seed,
                "variant": variant,
                "mean_active_rank": float(payload["mean_active_rank"]),
                **{metric: float(payload["calibrated_metrics"][metric]) for metric in METRICS},
            })
    summary = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            key: {
                "mean": float(np.mean([row[key] for row in selected])),
                "std": float(np.std([row[key] for row in selected], ddof=1)),
            }
            for key in ("mean_active_rank", *METRICS)
        }
    comparisons = {}
    full = [row for row in rows if row["variant"] == "full_stnlr"]
    for variant in VARIANTS[1:]:
        other = [row for row in rows if row["variant"] == variant]
        comparisons[f"full_stnlr_vs_{variant}"] = {
            metric: {
                "relative_reduction_percent": float(100.0 * (1.0 - np.mean([row[metric] for row in full]) / np.mean([row[metric] for row in other]))),
                "paired_wins_out_of_3": int(sum(a[metric] < b[metric] for a, b in zip(full, other))),
            }
            for metric in METRICS
        }
    result = {"protocol": "configs/experiment_protocols.yaml#objective_ablation", "summary": summary, "comparisons": comparisons, "rows": rows}
    output = args.result_root / "objective_ablation" / "aggregate.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary": summary, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
