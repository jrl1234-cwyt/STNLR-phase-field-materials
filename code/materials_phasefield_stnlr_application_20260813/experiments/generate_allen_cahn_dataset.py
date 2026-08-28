#!/usr/bin/env python3
"""Generate source and target 2-D Allen--Cahn snapshot distributions.

The solver uses a periodic Fourier grid. Diffusion is treated implicitly and
the local double-well reaction explicitly:

    u_{n+1} = (I - dt * eps^2 * Laplacian)^{-1}
              [u_n + dt * reaction * (u_n - u_n^3)].

The source domain contains a family of broad-interface, moderate-reaction
systems.  The target domain is a held-out narrow-interface, stronger-reaction
system.  Every saved sample comes from an independent smooth random initial
condition, so evaluation is permutation invariant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def smooth_initial_fields(rng: np.random.Generator, batch: int, n: int) -> np.ndarray:
    noise = rng.standard_normal((batch, n, n))
    freq = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    radius2 = kx**2 + ky**2
    cutoff = rng.uniform(3.0, 6.0, size=(batch, 1, 1))
    spectral_filter = np.exp(-0.5 * radius2[None] / cutoff**2)
    field = np.fft.ifft2(np.fft.fft2(noise, axes=(-2, -1)) * spectral_filter, axes=(-2, -1)).real
    field -= field.mean(axis=(-2, -1), keepdims=True)
    field /= field.std(axis=(-2, -1), keepdims=True) + 1.0e-8
    bias = rng.uniform(-0.12, 0.12, size=(batch, 1, 1))
    amplitude = rng.uniform(0.22, 0.38, size=(batch, 1, 1))
    return np.clip(amplitude * field + bias, -0.85, 0.85)


def integrate_batch(
    initial: np.ndarray,
    epsilon: np.ndarray,
    reaction: np.ndarray,
    final_time: np.ndarray,
    dt: float,
) -> np.ndarray:
    batch, n, _ = initial.shape
    dx = 1.0 / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    u = initial.astype(np.float64, copy=True)
    max_steps = int(np.ceil(float(final_time.max()) / dt))
    active_until = np.ceil(final_time / dt).astype(np.int64)
    denom = 1.0 + dt * epsilon[:, None, None] ** 2 * k2[None]
    for step in range(max_steps):
        active = step < active_until
        if not np.any(active):
            break
        rhs = u[active] + dt * reaction[active, None, None] * (u[active] - u[active] ** 3)
        u_hat = np.fft.fft2(rhs, axes=(-2, -1))
        u[active] = np.fft.ifft2(u_hat / denom[active], axes=(-2, -1)).real
        u[active] = np.clip(u[active], -1.05, 1.05)
    return u.astype(np.float32)


def write_split(
    out_dir: Path,
    count: int,
    n: int,
    batch_size: int,
    seed: int,
    dt: float,
    domain: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    parameter_rows = []
    while written < count:
        batch = min(batch_size, count - written)
        if domain == "source":
            epsilon = rng.uniform(0.035, 0.055, size=batch)
            reaction = rng.uniform(0.8, 1.2, size=batch)
            final_time = rng.uniform(0.45, 0.75, size=batch)
        elif domain == "target":
            epsilon = np.full(batch, 0.022, dtype=np.float64)
            reaction = np.full(batch, 1.6, dtype=np.float64)
            final_time = np.full(batch, 0.60, dtype=np.float64)
        else:
            raise ValueError(domain)

        u0 = smooth_initial_fields(rng, batch, n)
        u1 = integrate_batch(u0, epsilon, reaction, final_time, dt)
        for j in range(batch):
            index = written + j
            path = out_dir / f"sample_{index:06d}.npz"
            np.savez_compressed(
                path,
                field=u1[j].reshape(-1, 1),
                initial=u0[j].astype(np.float32).reshape(-1, 1),
                epsilon=np.float32(epsilon[j]),
                reaction=np.float32(reaction[j]),
                physical_time=np.float32(final_time[j]),
                boundary=np.array("periodic"),
            )
            parameter_rows.append(
                {
                    "file": path.name,
                    "epsilon": float(epsilon[j]),
                    "reaction": float(reaction[j]),
                    "physical_time": float(final_time[j]),
                }
            )
        written += batch
        print(f"{domain}: {written}/{count}", flush=True)

    manifest = {
        "domain": domain,
        "count": count,
        "grid": n,
        "dt": dt,
        "equation": "u_t = epsilon^2 Laplacian(u) + reaction * (u - u^3)",
        "boundary": "periodic",
        "seed": seed,
        "parameters": parameter_rows,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--source-count", type=int, default=3072)
    parser.add_argument("--target-count", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    if args.grid < 16 or args.grid % 4 != 0:
        raise ValueError("--grid must be >=16 and divisible by the DiT patch size 4")
    args.out_root.mkdir(parents=True, exist_ok=True)
    source = write_split(
        args.out_root / "source" / "Processed_Train",
        args.source_count,
        args.grid,
        args.batch_size,
        args.seed,
        args.dt,
        "source",
    )
    target = write_split(
        args.out_root / "target" / "Processed_Train",
        args.target_count,
        args.grid,
        args.batch_size,
        args.seed + 1,
        args.dt,
        "target",
    )
    summary = {"source": source, "target": target}
    with (args.out_root / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
