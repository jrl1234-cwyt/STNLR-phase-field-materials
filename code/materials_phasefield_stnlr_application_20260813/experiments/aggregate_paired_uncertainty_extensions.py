#!/usr/bin/env python3
"""Paired condition--seed bootstrap for the CH and PFHub extensions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get('STNLR_RESULTS_ROOT', Path.cwd())).resolve()
OUT = REPO / 'results_paired_uncertainty_extensions_20260825'


def hierarchical_reductions(a, b, repetitions, seed):
    """Resample condition and then seed, preserving method pairing."""
    rng = np.random.default_rng(seed)
    n_condition, n_seed, _ = a.shape
    samples = np.empty((repetitions, a.shape[-1]))
    for rep in range(repetitions):
        condition_ids = rng.integers(0, n_condition, n_condition)
        aa, bb = [], []
        for condition in condition_ids:
            seed_ids = rng.integers(0, n_seed, n_seed)
            aa.append(a[condition, seed_ids])
            bb.append(b[condition, seed_ids])
        aa = np.concatenate(aa)
        bb = np.concatenate(bb)
        samples[rep] = 100.0 * (aa.mean(0) - bb.mean(0)) / aa.mean(0)
    return samples


def summarize(a, b, metrics, repetitions, seed):
    samples = hierarchical_reductions(a, b, repetitions, seed)
    observed = 100.0 * (a.mean((0, 1)) - b.mean((0, 1))) / a.mean((0, 1))
    result = {}
    for index, metric in enumerate(metrics):
        result[metric] = {
            'continued_static_mean': float(a[:, :, index].mean()),
            'stnlr_mean': float(b[:, :, index].mean()),
            'relative_reduction_percent': float(observed[index]),
            'paired_hierarchical_bootstrap_95ci_percent': [float(x) for x in np.quantile(samples[:, index], [0.025, 0.975])],
            'bootstrap_probability_of_reduction': float((samples[:, index] > 0).mean()),
            'paired_wins': int((b[:, :, index] < a[:, :, index]).sum()),
            'pairs': int(a.shape[0] * a.shape[1]),
        }
    return result


def cahn_hilliard(repetitions):
    root = REPO / 'results_cahn_hilliard_stnlr_20260823/models'
    targets = ('eps0020_lam10', 'eps0020_lam14', 'eps0026_lam10', 'eps0026_lam14')
    metrics = (
        'trajectory_relative_l2_mean', 'terminal_relative_l2_mean',
        'trajectory_mass_drift_mae', 'maximum_mass_drift_mean',
        'terminal_free_energy_relative_error_mean',
        'terminal_structure_factor_centroid_relative_error_mean',
        'trajectory_pde_residual_relative_rms',
    )
    a = np.empty((4, 3, len(metrics)))
    b = np.empty_like(a)
    reference = np.empty((4, 3))
    for target_id, target in enumerate(targets):
        for seed in range(3):
            static = json.loads((root / target / 'static_rank16_continued_600' / f'seed{seed}/metrics.json').read_text())['test_metrics']
            stnlr = json.loads((root / target / 'stnlr_selfdistill_strict2' / f'seed{seed}/metrics.json').read_text())['test_calibrated_metrics']
            a[target_id, seed] = [static[metric] for metric in metrics]
            b[target_id, seed] = [stnlr[metric] for metric in metrics]
            reference[target_id, seed] = static['reference_discretization_residual_relative_rms']
    return {
        'hierarchy': 'condition -> training seed; method pairing preserved',
        'metrics': summarize(a, b, metrics, repetitions, 20260825),
        'reference_discretization_residual_relative_rms': {
            'mean': float(reference.mean()), 'sd': float(reference.std(ddof=1)),
        },
    }


def pfhub(repetitions):
    source = json.loads((REPO / 'results_pfhub_dendrite_stnlr_20260824/formal_pilot/dendrite_three_seed_aggregate.json').read_text())
    rows = source['rows']
    targets = ('standard', 'shifted')
    metrics = tuple(rows[0]['static'])
    a = np.empty((2, 3, len(metrics)))
    b = np.empty_like(a)
    for target_id, target in enumerate(targets):
        for seed in range(3):
            row = next(row for row in rows if row['target'] == target and row['seed'] == seed)
            a[target_id, seed] = [row['static'][metric] for metric in metrics]
            b[target_id, seed] = [row['stnlr'][metric] for metric in metrics]
    return {
        'hierarchy': 'condition -> training seed; method pairing preserved',
        'metrics': summarize(a, b, metrics, repetitions, 20260826),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        'bootstrap_repetitions': 10000,
        'cahn_hilliard': cahn_hilliard(10000),
        'pfhub': pfhub(10000),
    }
    (OUT / 'aggregate.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
