#!/usr/bin/env python3
"""Frozen-checkpoint audit on an alternate discretization and noisy initials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_hierarchical_bootstrap_allen_cahn import (
    SEEDS,
    TARGETS,
    data_path,
    hierarchical_bootstrap,
    load_model,
    per_sample_metrics,
    predict,
)
from audit_reference_solver_convergence import periodic_resample
from generate_allen_cahn_trajectories import integrate_trajectories
from train_evaluate_poseidon_allen_cahn import load_data


PARAMETERS = {
    "eps0022_lam14": (0.022, 1.4),
    "eps0022_lam16": (0.022, 1.6),
    "eps0028_lam14": (0.028, 1.4),
    "eps0028_lam16": (0.028, 1.6),
}


def integrate_finite_difference(initial, epsilon, reaction, times, solver_dt):
    """Semi-implicit solver using the second-order finite-difference symbol."""
    batch, grid, _ = initial.shape
    dx = 1.0 / grid
    frequency = np.fft.fftfreq(grid)
    sx, sy = np.meshgrid(np.sin(np.pi * frequency), np.sin(np.pi * frequency), indexing="ij")
    minus_laplacian = 4.0 * (sx**2 + sy**2) / dx**2
    field = initial.astype(np.float64, copy=True)
    output = np.empty((batch, len(times), grid, grid), dtype=np.float32)
    output[:, 0] = field
    step = 0
    for time_id in range(1, len(times)):
        target_step = int(round(float(times[time_id]) / solver_dt))
        while step < target_step:
            rhs = field + solver_dt * reaction[:, None, None] * (field - field**3)
            rhs_hat = np.fft.fft2(rhs, axes=(-2, -1))
            denominator = 1.0 + solver_dt * epsilon[:, None, None] ** 2 * minus_laplacian[None]
            field = np.fft.ifft2(rhs_hat / denominator, axes=(-2, -1)).real
            field = np.clip(field, -1.05, 1.05)
            step += 1
        output[:, time_id] = field
    return output


def build_variants(repo: Path, noise_seed: int):
    variants = {}
    agreement = {}
    for target_id, target in enumerate(TARGETS):
        epsilon, reaction = PARAMETERS[target]
        data = load_data(data_path(repo, target))
        initial = data["initial"][120:160].astype(np.float64)
        times = data["times"].astype(np.float64)
        eps = np.full(40, epsilon, dtype=np.float64)
        lam = np.full(40, reaction, dtype=np.float64)
        initial128 = periodic_resample(initial, 128)
        fd128 = integrate_finite_difference(initial128, eps, lam, times, 0.00125)
        fd128 = periodic_resample(fd128, 64).astype(np.float32)
        production = data["fields"][120:160]
        numerator = np.linalg.norm((production - fd128).reshape(40, 11, -1), axis=-1)
        denominator = np.linalg.norm(fd128.reshape(40, 11, -1), axis=-1)
        agreement[target] = {
            "trajectory_relative_l2_mean": float(np.mean(numerator / np.maximum(denominator, 1.0e-12))),
            "terminal_relative_l2_mean": float(np.mean(numerator[:, -1] / np.maximum(denominator[:, -1], 1.0e-12))),
        }
        variants.setdefault("finite_difference_128_clean", {})[target] = {
            "initial": initial.astype(np.float32),
            "target": fd128,
        }
        rng = np.random.default_rng(noise_seed + target_id)
        unit_noise = rng.standard_normal(initial.shape)
        for sigma in (0.02, 0.05):
            noisy = np.clip(initial + sigma * unit_noise, -0.85, 0.85)
            truth = integrate_trajectories(noisy, eps, lam, times, 0.00125)
            key = f"spectral_noisy_initial_sigma{sigma:.2f}".replace(".", "p")
            variants.setdefault(key, {})[target] = {
                "initial": noisy.astype(np.float32),
                "target": truth.astype(np.float32),
            }
    return variants, agreement


def evaluate_variants(
    repo: Path,
    variants: dict,
    static_root: Path,
    device: torch.device,
    batch_size: int,
):
    all_records = {variant: {} for variant in variants}
    strict_root = repo / "results_poseidon_t_allen_cahn_formal_20260817/physics_calibrated_selfdistill_test_strict2"
    trained_root = repo / "results_poseidon_t_allen_cahn_formal_20260817/physics_calibrated_selfdistill"
    for target in TARGETS:
        data = load_data(data_path(repo, target))
        epsilon = torch.from_numpy(data["epsilon"][120:160]).to(device)
        reaction = torch.from_numpy(data["reaction"][120:160]).to(device)
        times = torch.from_numpy(data["times"]).to(device)
        tensors = {
            variant: (
                torch.from_numpy(payload[target]["initial"]).to(device).unsqueeze(1),
                torch.from_numpy(payload[target]["target"]).to(device).unsqueeze(2),
            )
            for variant, payload in variants.items()
        }
        for seed in SEEDS:
            trace = json.loads((strict_root / target / f"seed{seed}/metrics.json").read_text())["rank_trace"]
            static = load_model("static", static_root / target / f"seed{seed}/final.pt", device)
            static_predictions = {
                variant: predict(static, initial, times, batch_size, None)
                for variant, (initial, _) in tensors.items()
            }
            del static
            torch.cuda.empty_cache()
            nested = load_model("stnlr", trained_root / target / f"seed{seed}/final.pt", device)
            nested_predictions = {
                variant: predict(nested, initial, times, batch_size, trace)
                for variant, (initial, _) in tensors.items()
            }
            del nested
            torch.cuda.empty_cache()
            for variant, (_, truth) in tensors.items():
                all_records[variant][f"{target}/seed{seed}"] = {
                    "target": target,
                    "seed": seed,
                    "rank_trace": trace,
                    "continued_static": {
                        key: values.tolist() for key, values in per_sample_metrics(
                            static_predictions[variant], truth, epsilon, reaction
                        ).items()
                    },
                    "stnlr": {
                        key: values.tolist() for key, values in per_sample_metrics(
                            nested_predictions[variant], truth, epsilon, reaction
                        ).items()
                    },
                }
            print(f"evaluated alternate inputs {target} seed{seed}", flush=True)
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--static-root",
        type=Path,
        default=Path("results_poseidon_static_continuation_20260823"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--noise-seed", type=int, default=20260825)
    args = parser.parse_args()
    variants, agreement = build_variants(args.repo, args.noise_seed)
    records = evaluate_variants(
        args.repo, variants, args.static_root, torch.device(args.device), args.batch_size
    )
    inference = {}
    pair_rows = {}
    for variant, variant_records in records.items():
        inference[variant], pair_rows[variant] = hierarchical_bootstrap(
            variant_records, args.bootstrap_repetitions, args.noise_seed
        )
    result = {
        "protocol": {
            "frozen_checkpoints_and_rank_traces": True,
            "static_root": str(args.static_root),
            "alternate_solver": "128x128 second-order finite-difference Laplacian, semi-implicit diffusion, dt=0.00125",
            "noisy_initial_levels": [0.02, 0.05],
            "noise": "fixed-seed additive white Gaussian noise; each noisy field is independently evolved to create a physically matched target",
            "hierarchy": "target condition -> training seed -> paired test trajectory",
            "targets": list(TARGETS),
            "training_seeds": list(SEEDS),
            "test_trajectories_per_cell": 40,
        },
        "production_vs_finite_difference_reference": agreement,
        "inference": inference,
        "pair_rows": pair_rows,
        "per_sample_records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(inference, indent=2))


if __name__ == "__main__":
    main()
