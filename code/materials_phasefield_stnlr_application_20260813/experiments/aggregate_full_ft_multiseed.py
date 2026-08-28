#!/usr/bin/env python3
"""Aggregate four-target, three-seed full fine-tuning and nested PEFT results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGETS = ("eps0022_lam14", "eps0022_lam16", "eps0028_lam14", "eps0028_lam16")
METRICS = (
    "trajectory_relative_l2_mean",
    "terminal_relative_l2_mean",
    "terminal_interface_fraction_mae",
    "terminal_free_energy_relative_error_mean",
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
    parser.add_argument("--seed0-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--stnlr-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for target in TARGETS:
        for seed in range(3):
            full_path = (args.seed0_root / target / "seed0" / "metrics.json") if seed == 0 else (args.extension_root / target / f"seed{seed}" / "metrics.json")
            nested_path = args.stnlr_root / target / f"seed{seed}" / "metrics.json"
            full_payload = json.loads(full_path.read_text())
            nested_payload = json.loads(nested_path.read_text())
            rows.append({
                "target": target,
                "seed": seed,
                "full": full_payload["metrics"],
                "stnlr": nested_payload["metrics"],
                "full_trainable_parameters": full_payload["trainable_parameters"],
                "stnlr_mean_active_rank": nested_payload["mean_active_rank"],
            })
    summary = {
        method: {metric: seed_macro(rows, method, metric) for metric in METRICS}
        for method in ("full", "stnlr")
    }
    result = {
        "pairs": len(rows),
        "summary": summary,
        "stnlr_pairwise_wins": {metric: int(sum(row["stnlr"][metric] < row["full"][metric] for row in rows)) for metric in METRICS},
        "full_trainable_parameters": int(rows[0]["full_trainable_parameters"]),
        "stnlr_mean_active_rank": mean_sd([row["stnlr_mean_active_rank"] for row in rows]),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
