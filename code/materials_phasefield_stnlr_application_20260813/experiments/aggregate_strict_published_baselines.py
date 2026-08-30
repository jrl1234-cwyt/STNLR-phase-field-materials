#!/usr/bin/env python3
"""Aggregate the paired AdaLoRA and TimeStep Master baseline runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


METRICS = (
    "trajectory_relative_l2_mean",
    "terminal_relative_l2_mean",
    "terminal_interface_fraction_mae",
    "terminal_free_energy_relative_error_mean",
    "trajectory_pde_residual_relative_rms",
)


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    methods = {}
    missing = []
    for method in ("adalora", "timestep_master"):
        seeds = []
        for seed in (0, 1, 2):
            path = args.root / method / f"seed{seed}" / "metrics.json"
            if not path.is_file():
                missing.append(str(path.relative_to(args.root)))
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            row["seed"] = seed
            seeds.append(row)
        aggregate = {
            "completed_seeds": len(seeds),
            "metrics": {
                metric: mean_std([seed["metrics"][metric] for seed in seeds])
                for metric in METRICS
            } if seeds else {},
            "training_seconds": mean_std([seed["training_seconds"] for seed in seeds]) if seeds else {},
        }
        if seeds and method == "adalora":
            aggregate["mean_active_rank"] = mean_std([seed["mean_active_rank"] for seed in seeds])
        if seeds and method == "timestep_master":
            aggregate["stored_adapter_rank_equivalent"] = seeds[0]["stored_adapter_rank_equivalent"]
            aggregate["active_adapter_rank_equivalent"] = seeds[0]["active_adapter_rank_equivalent"]
            aggregate["router_parameters"] = seeds[0]["router_parameters"]
        methods[method] = {"seeds": seeds, "aggregate": aggregate}
    payload = {
        "complete": not missing,
        "protocol": "PROTOCOL.md",
        "methods": methods,
        "missing": missing,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state = "completed successfully" if not missing else "running"
    lines = [
        "# Strict published-baseline status",
        "",
        f"- Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- State: {state}",
        f"- Completed runs: {6-len(missing)} / 6",
        "- Methods: AdaLoRA (ICLR 2023) and TimeStep Master (ICML 2025)",
        "- Manuscript modified: no",
        "- Final aggregate: `aggregate.json`",
        "",
    ]
    if missing:
        lines.extend(["## Pending", "", *[f"- `{item}`" for item in missing], ""])
    else:
        lines.extend(["Both GPU workers have exited and all six metric files are present.", ""])
    (args.root / "RUN_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": not missing, "completed": 6-len(missing), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
