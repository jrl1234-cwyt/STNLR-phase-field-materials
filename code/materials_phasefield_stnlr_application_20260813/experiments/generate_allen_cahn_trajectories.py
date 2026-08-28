#!/usr/bin/env python3
"""Generate parameterized periodic Allen--Cahn trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_allen_cahn_dataset import smooth_initial_fields


def integrate_trajectories(initial, epsilon, reaction, times, solver_dt):
    batch, n, _ = initial.shape
    dx = 1.0 / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    u = initial.astype(np.float64, copy=True)
    output = np.empty((batch, len(times), n, n), dtype=np.float32)
    output[:, 0] = u
    step = 0
    for time_index in range(1, len(times)):
        target_step = int(round(float(times[time_index]) / solver_dt))
        while step < target_step:
            rhs = u + solver_dt * reaction[:, None, None] * (u - u**3)
            u_hat = np.fft.fft2(rhs, axes=(-2, -1))
            denom = 1.0 + solver_dt * epsilon[:, None, None] ** 2 * k2[None]
            u = np.fft.ifft2(u_hat / denom, axes=(-2, -1)).real
            u = np.clip(u, -1.05, 1.05)
            step += 1
        output[:, time_index] = u
    return output


def make_split(
    count,
    grid,
    times,
    solver_dt,
    seed,
    domain,
    target_epsilon=0.022,
    target_reaction=1.6,
):
    rng = np.random.default_rng(seed)
    initial = smooth_initial_fields(rng, count, grid).astype(np.float32)
    if domain == "source":
        epsilon = rng.uniform(0.035, 0.055, size=count).astype(np.float32)
        reaction = rng.uniform(0.8, 1.2, size=count).astype(np.float32)
    elif domain == "target":
        epsilon = np.full(count, target_epsilon, dtype=np.float32)
        reaction = np.full(count, target_reaction, dtype=np.float32)
    else:
        raise ValueError(domain)
    fields = integrate_trajectories(initial, epsilon, reaction, times, solver_dt)
    return {
        "fields": fields,
        "initial": initial,
        "epsilon": epsilon,
        "reaction": reaction,
        "times": times.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=512)
    parser.add_argument("--target-count", type=int, default=160)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--max-time", type=float, default=0.6)
    parser.add_argument("--time-points", type=int, default=11)
    parser.add_argument("--solver-dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--target-epsilon", type=float, default=0.022)
    parser.add_argument("--target-reaction", type=float, default=1.6)
    parser.add_argument(
        "--only-target",
        action="store_true",
        help="Generate only target trajectories when reusing an existing source checkpoint.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, args.max_time, args.time_points, dtype=np.float64)
    metadata = {
        "equation": "u_tau = epsilon^2 Laplacian(u) + reaction*(u-u^3)",
        "boundary": "periodic",
        "grid": args.grid,
        "times": times.tolist(),
        "solver_dt": args.solver_dt,
        "source_count": args.source_count,
        "target_count": args.target_count,
        "source_parameters": {"epsilon": [0.035, 0.055], "reaction": [0.8, 1.2]},
        "target_parameters": {
            "epsilon": args.target_epsilon,
            "reaction": args.target_reaction,
        },
        "target_split": {"adapt": [0, 100], "validation": [100, 120], "test": [120, 160]},
    }
    splits = [
        (0, "source", args.source_count),
        (1, "target", args.target_count),
    ]
    if args.only_target:
        splits = splits[1:]
    for offset, domain, count in splits:
        payload = make_split(
            count,
            args.grid,
            times,
            args.solver_dt,
            args.seed + offset,
            domain,
            target_epsilon=args.target_epsilon,
            target_reaction=args.target_reaction,
        )
        np.savez_compressed(args.out_dir / f"{domain}_trajectories.npz", **payload)
        print(domain, payload["fields"].shape, flush=True)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
