from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

from generate_allen_cahn_trajectories import integrate_trajectories
from generate_cahn_hilliard_trajectories import integrate as integrate_ch


AC_TARGETS = {
    "eps0022_lam14": (0.022, 1.4),
    "eps0022_lam16": (0.022, 1.6),
    "eps0028_lam14": (0.028, 1.4),
    "eps0028_lam16": (0.028, 1.6),
}
CH_TARGETS = {
    "eps0020_lam10": (0.020, 1.0),
    "eps0020_lam14": (0.020, 1.4),
    "eps0026_lam10": (0.026, 1.0),
    "eps0026_lam14": (0.026, 1.4),
}


def periodic_resample(field: np.ndarray, grid: int) -> np.ndarray:
    """Fourier resample the final two periodic axes with amplitude preserved."""
    if field.shape[-1] == grid:
        return field.astype(np.float64, copy=True)
    resized = resample(field, grid, axis=-2)
    resized = resample(resized, grid, axis=-1)
    return np.asarray(resized.real, dtype=np.float64)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm((candidate - reference).reshape(candidate.shape[0], candidate.shape[1], -1), axis=-1)
    denominator = np.linalg.norm(reference.reshape(reference.shape[0], reference.shape[1], -1), axis=-1)
    return numerator / np.maximum(denominator, 1.0e-12)


def free_energy(field: np.ndarray, epsilon: float, reaction: float) -> np.ndarray:
    grid = field.shape[-1]
    dx = 1.0 / grid
    gx = (np.roll(field, -1, axis=-1) - np.roll(field, 1, axis=-1)) / (2.0 * dx)
    gy = (np.roll(field, -1, axis=-2) - np.roll(field, 1, axis=-2)) / (2.0 * dx)
    density = 0.5 * epsilon**2 * (gx**2 + gy**2)
    density += 0.25 * reaction * (field**2 - 1.0) ** 2
    return density.mean(axis=(-2, -1))


def comparison(candidate: np.ndarray, reference: np.ndarray, epsilon: float, reaction: float) -> dict[str, float]:
    rel = relative_l2(candidate, reference)
    candidate_energy = free_energy(candidate[:, -1], epsilon, reaction)
    reference_energy = free_energy(reference[:, -1], epsilon, reaction)
    candidate_interface = (np.abs(candidate[:, -1]) < 0.2).mean(axis=(-2, -1))
    reference_interface = (np.abs(reference[:, -1]) < 0.2).mean(axis=(-2, -1))
    return {
        "trajectory_relative_l2_mean": float(rel.mean()),
        "terminal_relative_l2_mean": float(rel[:, -1].mean()),
        "terminal_free_energy_relative_error_mean": float(
            np.mean(np.abs(candidate_energy - reference_energy) / np.maximum(np.abs(reference_energy), 1.0e-12))
        ),
        "terminal_interface_fraction_mae": float(np.mean(np.abs(candidate_interface - reference_interface))),
    }


def ac_data_path(repo: Path, tag: str) -> Path:
    if tag == "eps0022_lam16":
        return repo / "results_allen_cahn_conditional_trajectory_20260816/data/target_trajectories.npz"
    return repo / f"results_allen_cahn_multitarget_transfer_20260817/targets/{tag}/data/target_trajectories.npz"


def run_allen_cahn(repo: Path, samples: int) -> dict:
    times = np.linspace(0.0, 0.6, 11, dtype=np.float64)
    rows = []
    for tag, (epsilon, reaction) in AC_TARGETS.items():
        with np.load(ac_data_path(repo, tag)) as data:
            initial64 = data["initial"][120 : 120 + samples].astype(np.float64)
            stored = data["fields"][120 : 120 + samples].astype(np.float64)
        parameter_epsilon = np.full(samples, epsilon, dtype=np.float64)
        parameter_reaction = np.full(samples, reaction, dtype=np.float64)
        production = integrate_trajectories(initial64, parameter_epsilon, parameter_reaction, times, 0.005).astype(np.float64)
        time_refined = integrate_trajectories(initial64, parameter_epsilon, parameter_reaction, times, 0.00125).astype(np.float64)
        initial128 = periodic_resample(initial64, 128)
        grid_refined128 = integrate_trajectories(initial128, parameter_epsilon, parameter_reaction, times, 0.0025)
        grid_refined128 = periodic_resample(grid_refined128, 64)
        initial256 = periodic_resample(initial64, 256)
        grid_refined256 = integrate_trajectories(initial256, parameter_epsilon, parameter_reaction, times, 0.005)
        grid_refined256 = periodic_resample(grid_refined256, 64)
        reference256 = integrate_trajectories(initial256, parameter_epsilon, parameter_reaction, times, 0.00125)
        reference256 = periodic_resample(reference256, 64)
        rows.append({
            "target": tag,
            "stored_reproduction": comparison(stored, production, epsilon, reaction),
            "production_64_dt005_vs_reference_256_dt00125": comparison(production, reference256, epsilon, reaction),
            "time_refined_64_dt00125_vs_reference": comparison(time_refined, reference256, epsilon, reaction),
            "grid_refined_128_dt0025_vs_reference": comparison(grid_refined128, reference256, epsilon, reaction),
            "grid_refined_256_dt005_vs_reference": comparison(grid_refined256, reference256, epsilon, reaction),
        })
    return aggregate("Allen--Cahn", rows, samples)


def run_cahn_hilliard(repo: Path, samples: int) -> dict:
    times = np.linspace(0.0, 0.6, 11, dtype=np.float64)
    rows = []
    for tag, (epsilon, reaction) in CH_TARGETS.items():
        path = repo / f"results_cahn_hilliard_stnlr_20260823/data/{tag}/trajectories.npz"
        with np.load(path) as data:
            initial64 = data["initial"][120 : 120 + samples].astype(np.float64)
            stored = data["fields"][120 : 120 + samples].astype(np.float64)
        eps = np.full(samples, epsilon, dtype=np.float64)
        lam = np.full(samples, reaction, dtype=np.float64)
        mobility = np.full(samples, 0.01, dtype=np.float64)
        production = integrate_ch(initial64, eps, lam, mobility, times, 0.002).astype(np.float64)
        time_refined = integrate_ch(initial64, eps, lam, mobility, times, 0.0005).astype(np.float64)
        initial128 = periodic_resample(initial64, 128)
        grid_refined128 = integrate_ch(initial128, eps, lam, mobility, times, 0.002)
        grid_refined128 = periodic_resample(grid_refined128, 64)
        reference128 = integrate_ch(initial128, eps, lam, mobility, times, 0.0005)
        reference128 = periodic_resample(reference128, 64)
        rows.append({
            "target": tag,
            "stored_reproduction": comparison(stored, production, epsilon, reaction),
            "production_64_dt002_vs_reference_128_dt0005": comparison(production, reference128, epsilon, reaction),
            "time_refined_64_dt0005_vs_reference": comparison(time_refined, reference128, epsilon, reaction),
            "grid_refined_128_dt002_vs_reference": comparison(grid_refined128, reference128, epsilon, reaction),
        })
    return aggregate("Cahn--Hilliard", rows, samples)


def aggregate(system: str, rows: list[dict], samples: int) -> dict:
    labels = [key for key in rows[0] if key != "target"]
    macro = {}
    for label in labels:
        metrics = rows[0][label]
        macro[label] = {
            metric: float(np.mean([row[label][metric] for row in rows]))
            for metric in metrics
        }
    return {
        "system": system,
        "paired_initial_conditions_per_target": samples,
        "targets": list(AC_TARGETS if system == "Allen--Cahn" else CH_TARGETS),
        "macro": macro,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    result = {
        "protocol": {
            "same_initial_condition_across_discretizations": True,
            "fine_grid_projection": "periodic Fourier resampling to the 64x64 production grid",
            "admission": "report only if refinement decreases errors and production discretization error is small relative to surrogate error",
        },
        "allen_cahn": run_allen_cahn(args.repo, args.samples),
        "cahn_hilliard": run_cahn_hilliard(args.repo, args.samples),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["allen_cahn"]["macro"], indent=2))
    print(json.dumps(result["cahn_hilliard"]["macro"], indent=2))


if __name__ == "__main__":
    main()
