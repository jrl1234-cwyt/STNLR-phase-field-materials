#!/usr/bin/env python3
"""Generate periodic two-dimensional Cahn--Hilliard trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def initial_fields(
    rng: np.random.Generator,
    count: int,
    grid: int,
    composition_min: float,
    composition_max: float,
    amplitude_min: float,
    amplitude_max: float,
) -> np.ndarray:
    noise = rng.standard_normal((count, grid, grid))
    freq = np.fft.fftfreq(grid) * grid
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    radius2 = kx**2 + ky**2
    cutoff = rng.uniform(4.0, 7.0, size=(count, 1, 1))
    filtered = np.fft.ifft2(
        np.fft.fft2(noise, axes=(-2, -1))
        * np.exp(-0.5 * radius2[None] / cutoff**2),
        axes=(-2, -1),
    ).real
    filtered -= filtered.mean(axis=(-2, -1), keepdims=True)
    filtered /= filtered.std(axis=(-2, -1), keepdims=True) + 1.0e-8
    composition = rng.uniform(composition_min, composition_max, size=(count, 1, 1))
    amplitude = rng.uniform(amplitude_min, amplitude_max, size=(count, 1, 1))
    return np.clip(composition + amplitude * filtered, -0.35, 0.35).astype(np.float32)


def integrate(initial, epsilon, reaction, mobility, times, solver_dt, stabilization_factor=2.0):
    batch, grid, _ = initial.shape
    dx = 1.0 / grid
    wave = 2.0 * np.pi * np.fft.fftfreq(grid, d=dx)
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    k2 = kx**2 + ky**2
    k4 = k2**2
    u = initial.astype(np.float64, copy=True)
    conserved_mean = u.mean(axis=(-2, -1), keepdims=True)
    output = np.empty((batch, len(times), grid, grid), dtype=np.float32)
    output[:, 0] = u
    step = 0
    stabilization = stabilization_factor * reaction
    for time_index in range(1, len(times)):
        target_step = int(round(float(times[time_index]) / solver_dt))
        while step < target_step:
            u_hat = np.fft.fft2(u, axes=(-2, -1))
            nonlinear_hat = np.fft.fft2(u**3 - u, axes=(-2, -1))
            mk2 = mobility[:, None, None] * k2[None]
            numerator = (
                (1.0 + solver_dt * mk2 * stabilization[:, None, None]) * u_hat
                - solver_dt * mk2 * reaction[:, None, None] * nonlinear_hat
            )
            denominator = 1.0 + solver_dt * mk2 * (
                epsilon[:, None, None] ** 2 * k2[None]
                + stabilization[:, None, None]
            )
            u = np.fft.ifft2(numerator / denominator, axes=(-2, -1)).real
            # Remove accumulated floating-point drift without changing morphology.
            u += conserved_mean - u.mean(axis=(-2, -1), keepdims=True)
            u = np.clip(u, -1.05, 1.05)
            step += 1
        output[:, time_index] = u
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--max-time", type=float, default=0.6)
    parser.add_argument("--time-points", type=int, default=11)
    parser.add_argument("--solver-dt", type=float, default=0.002)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--reaction", type=float, required=True)
    parser.add_argument("--mobility", type=float, default=0.01)
    parser.add_argument("--composition-min", type=float, default=-0.12)
    parser.add_argument("--composition-max", type=float, default=0.12)
    parser.add_argument("--amplitude-min", type=float, default=0.045)
    parser.add_argument("--amplitude-max", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    times = np.linspace(0.0, args.max_time, args.time_points, dtype=np.float64)
    initial = initial_fields(
        rng,
        args.count,
        args.grid,
        args.composition_min,
        args.composition_max,
        args.amplitude_min,
        args.amplitude_max,
    )
    epsilon = np.full(args.count, args.epsilon, dtype=np.float32)
    reaction = np.full(args.count, args.reaction, dtype=np.float32)
    mobility = np.full(args.count, args.mobility, dtype=np.float32)
    fields = integrate(initial, epsilon, reaction, mobility, times, args.solver_dt)
    np.savez_compressed(
        args.out_dir / "trajectories.npz",
        fields=fields,
        initial=initial,
        epsilon=epsilon,
        reaction=reaction,
        mobility=mobility,
        times=times.astype(np.float32),
    )
    mass = fields.mean(axis=(-2, -1))
    dx = 1.0 / args.grid
    gradient_x = (np.roll(fields, -1, axis=-1) - np.roll(fields, 1, axis=-1)) / (2.0 * dx)
    gradient_y = (np.roll(fields, -1, axis=-2) - np.roll(fields, 1, axis=-2)) / (2.0 * dx)
    energy = (
        0.5 * args.epsilon**2 * (gradient_x**2 + gradient_y**2)
        + 0.25 * args.reaction * (fields**2 - 1.0) ** 2
    ).mean(axis=(-2, -1))
    maximum_energy_increase = float(np.diff(energy, axis=1).max())
    if not np.isfinite(fields).all():
        raise RuntimeError("Cahn--Hilliard integration produced non-finite values")
    if maximum_energy_increase > 1.0e-6:
        raise RuntimeError(f"discrete free energy increased by {maximum_energy_increase:.3e}")
    metadata = {
        "equation": "u_t = mobility * Laplacian(reaction*(u^3-u) - epsilon^2*Laplacian(u))",
        "boundary": "periodic",
        "grid": args.grid,
        "times": times.tolist(),
        "solver_dt": args.solver_dt,
        "epsilon": args.epsilon,
        "reaction": args.reaction,
        "mobility": args.mobility,
        "initial_composition_range": [args.composition_min, args.composition_max],
        "initial_amplitude_range": [args.amplitude_min, args.amplitude_max],
        "count": args.count,
        "split": {"adapt": [0, 100], "validation": [100, 120], "test": [120, 160]},
        "maximum_absolute_mass_drift": float(np.max(np.abs(mass - mass[:, :1]))),
        "maximum_discrete_energy_increase": maximum_energy_increase,
        "mean_initial_free_energy": float(energy[:, 0].mean()),
        "mean_terminal_free_energy": float(energy[:, -1].mean()),
        "field_range": [float(fields.min()), float(fields.max())],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
