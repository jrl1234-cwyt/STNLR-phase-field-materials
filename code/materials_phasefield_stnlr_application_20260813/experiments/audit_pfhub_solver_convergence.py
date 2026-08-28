#!/usr/bin/env python3
"""Paired spatial/time refinement audit for the short-horizon PFHub solver."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from generate_pfhub_dendrite_trajectories import generate
from train_evaluate_poseidon_dendrite import summarize


CONDITIONS = {
    "standard": {"diffusion": 10.0, "undercooling": -0.3, "epsilon4": 0.05},
    "shifted": {"diffusion": 12.0, "undercooling": -0.32, "epsilon4": 0.065},
}


def configuration(condition: str, grid: int, dx: float, dt: float, device: str) -> Namespace:
    values = CONDITIONS[condition]
    return Namespace(
        trajectories=1,
        grid=grid,
        save_grid=96,
        dx=dx,
        dt=dt,
        t_end=300.0,
        outputs=9,
        epsilon4=values["epsilon4"],
        theta0=0.0,
        diffusion=values["diffusion"],
        undercooling=values["undercooling"],
        w0=1.0,
        tau0=1.0,
        seed_radius=8.0,
        seed_jitter=0.0,
        seed_shape_noise=0.0,
        temperature_noise=0.0,
        exact_index=0,
        seed=20260825,
        device=device,
    )


def compare(candidate: np.ndarray, reference: np.ndarray, condition: str) -> dict[str, float]:
    values = CONDITIONS[condition]
    candidate_t = torch.from_numpy(candidate)
    reference_t = torch.from_numpy(reference)
    return summarize(
        candidate_t,
        reference_t,
        values["diffusion"],
        values["epsilon4"],
        1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "production_192_dx05_dt005": configuration(args.condition, 192, 0.5, 0.005, args.device),
        "time_refined_192_dx05_dt0025": configuration(args.condition, 192, 0.5, 0.0025, args.device),
        "grid_refined_256_dx0375_dt0025": configuration(args.condition, 256, 0.375, 0.0025, args.device),
        "reference_256_dx0375_dt00125": configuration(args.condition, 256, 0.375, 0.00125, args.device),
    }
    trajectories = {}
    for name, config in cases.items():
        print(f"running {name}", flush=True)
        trajectories[name] = generate(config)["fields"]
    reference = trajectories["reference_256_dx0375_dt00125"]
    comparisons = {
        name: compare(trajectory, reference, args.condition)
        for name, trajectory in trajectories.items()
        if name != "reference_256_dx0375_dt00125"
    }
    result = {
        "protocol": {
            "condition": args.condition,
            "same_initial_condition": "exact radius-8 unperturbed seed, no temperature noise",
            "fixed_physical_extent": 96.0,
            "comparison_grid": "all fields bilinearly sampled to 96x96 by the production generator",
            "terminal_time": 300.0,
            "outputs": 9,
        },
        "comparisons_to_reference": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
