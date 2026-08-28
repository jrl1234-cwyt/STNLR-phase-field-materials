#!/usr/bin/env python3
"""Select the Poseidon-T material schedule using validation metrics only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TARGETS = ("eps0022_lam14", "eps0022_lam16", "eps0028_lam14", "eps0028_lam16")
METHODS = ("stnlr_early_high", "stnlr_u_mid4", "stnlr_u_mid8")
METRICS = (
    "trajectory_relative_l2_mean",
    "terminal_relative_l2_mean",
    "terminal_interface_fraction_mae",
    "terminal_free_energy_relative_error_mean",
    "trajectory_pde_residual_relative_rms",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    scores, details = {}, {}
    for method in METHODS:
        log_ratios = []
        details[method] = {}
        for target in TARGETS:
            base = json.loads((args.root / "validation" / target / "static_rank16" / "seed0" / "metrics.json").read_text())["metrics"]
            row = json.loads((args.root / "validation" / target / method / "seed0" / "metrics.json").read_text())["metrics"]
            ratios = {key: row[key] / max(base[key], 1e-12) for key in METRICS}
            details[method][target] = ratios
            log_ratios.extend(math.log(max(value, 1e-12)) for value in ratios.values())
        scores[method] = math.exp(sum(log_ratios) / len(log_ratios))
    summary = {
        "selection_rule": "lowest geometric mean of five lower-is-better metric ratios to static rank-16 over four validation targets",
        "scores": scores,
        "selected_u_schedule": min(("stnlr_u_mid4", "stnlr_u_mid8"), key=scores.get),
        "selected_dynamic_schedule": min(METHODS, key=scores.get),
        "metric_ratios": details,
    }
    (args.root / "validation" / "selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
