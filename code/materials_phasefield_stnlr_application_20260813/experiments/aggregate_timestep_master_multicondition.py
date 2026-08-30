#!/usr/bin/env python3
"""Aggregate four-condition paired ST-NLR vs TimeStep Master inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGETS = ("eps0022_lam14", "eps0022_lam16", "eps0028_lam14", "eps0028_lam16")
SEEDS = (0, 1, 2)
METRICS = (
    "trajectory_relative_l2",
    "terminal_relative_l2",
    "terminal_interface_fraction_mae",
    "terminal_free_energy_relative_error",
)


def load_tsm(root: Path, target: str, seed: int) -> dict[str, list[float]]:
    cell = root / "timestep_master" / target / f"seed{seed}"
    candidates = (
        cell / "metrics.json",
        cell / "paired_audit.json",
        root / "cells" / target / f"seed{seed}.json",
    )
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("per_sample_records")
            if records and all(metric in records for metric in METRICS):
                return records
    raise FileNotFoundError(f"no per-sample TimeStep Master metrics in {cell}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stnlr-bootstrap", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()

    stnlr_payload = json.loads(args.stnlr_bootstrap.read_text(encoding="utf-8"))
    cells = {}
    missing = []
    for target in TARGETS:
        for seed in SEEDS:
            key = f"{target}/seed{seed}"
            try:
                tsm = load_tsm(args.root, target, seed)
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            stnlr = stnlr_payload["per_sample_records"][key]["stnlr"]
            cells[key] = {"target": target, "seed": seed, "timestep_master": tsm, "stnlr": stnlr}

    result = {
        "status": "incomplete" if missing else "complete",
        "protocol": {
            "hierarchy": "target condition -> training seed -> paired test trajectory",
            "targets": list(TARGETS),
            "training_seeds": list(SEEDS),
            "test_trajectories_per_cell": 40,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "completed_cells": sorted(cells),
        "missing": missing,
    }
    if missing:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    pair_rows = []
    for target in TARGETS:
        for seed in SEEDS:
            row = cells[f"{target}/seed{seed}"]
            pair = {"target": target, "seed": seed}
            for metric in METRICS:
                tsm = float(np.mean(row["timestep_master"][metric]))
                stnlr = float(np.mean(row["stnlr"][metric]))
                pair[metric] = {
                    "timestep_master": tsm,
                    "stnlr": stnlr,
                    "relative_reduction_percent": 100.0 * (tsm - stnlr) / tsm,
                }
            pair_rows.append(pair)

    rng = np.random.default_rng(args.bootstrap_seed)
    draws = {metric: np.empty(args.bootstrap_repetitions, dtype=np.float64) for metric in METRICS}
    observed = {}
    for metric in METRICS:
        tsm = np.mean([row[metric]["timestep_master"] for row in pair_rows])
        stnlr = np.mean([row[metric]["stnlr"] for row in pair_rows])
        observed[metric] = {
            "timestep_master_mean": float(tsm),
            "stnlr_mean": float(stnlr),
            "relative_reduction_percent": float(100.0 * (tsm - stnlr) / tsm),
            "paired_wins_out_of_12": int(sum(row[metric]["stnlr"] < row[metric]["timestep_master"] for row in pair_rows)),
        }

    for repetition in range(args.bootstrap_repetitions):
        target_ids = rng.integers(0, len(TARGETS), size=len(TARGETS))
        sums = {metric: [0.0, 0.0] for metric in METRICS}
        count = 0
        for target_id in target_ids:
            target = TARGETS[int(target_id)]
            seed_ids = rng.integers(0, len(SEEDS), size=len(SEEDS))
            for seed_id in seed_ids:
                row = cells[f"{target}/seed{SEEDS[int(seed_id)]}"]
                sample_count = len(row["stnlr"][METRICS[0]])
                sample_ids = rng.integers(0, sample_count, size=sample_count)
                for metric in METRICS:
                    sums[metric][0] += np.asarray(row["timestep_master"][metric])[sample_ids].mean()
                    sums[metric][1] += np.asarray(row["stnlr"][metric])[sample_ids].mean()
                count += 1
        for metric in METRICS:
            tsm = sums[metric][0] / count
            stnlr = sums[metric][1] / count
            draws[metric][repetition] = 100.0 * (tsm - stnlr) / tsm

    for metric in METRICS:
        low, high = np.quantile(draws[metric], [0.025, 0.975])
        observed[metric]["hierarchical_bootstrap_95ci_percent"] = [float(low), float(high)]
        observed[metric]["bootstrap_probability_of_improvement"] = float(np.mean(draws[metric] > 0.0))
    result.update({"inference": observed, "pair_rows": pair_rows})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
