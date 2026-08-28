# Dataset card

## Systems and intended use

The release contains deterministic numerical trajectories for surrogate adaptation and evaluation in three phase-field systems:

| System | Conditions | Grid supplied to surrogate | Output times | Split per condition |
|---|---:|---:|---:|---:|
| Allen--Cahn | 4 | 64 x 64 | 11, 0 to 0.60 | 100 train, 20 validation, 40 test |
| Cahn--Hilliard | 4 | 64 x 64 | 11, 0 to 0.60 | 100 train, 20 validation, 40 test |
| PFHub-3-type | 2 | 96 x 96, two fields | 9, 0 to 300 | 20 train, 4 validation, 8 test |

Allen--Cahn conditions are `(epsilon, lambda)` in `{(0.022,1.4), (0.022,1.6), (0.028,1.4), (0.028,1.6)}`. Cahn--Hilliard conditions are `{(0.020,1.0), (0.020,1.4), (0.026,1.0), (0.026,1.4)}`. PFHub files contain the standard and shifted conditions described in the manuscript.

## Generation and provenance

The Allen--Cahn and Cahn--Hilliard datasets use periodic semi-implicit Fourier schemes. PFHub-3-type data use a finite-volume solver with zero-flux boundaries. Dataset-generation programs and random seeds are included. Spatial/time refinement results and an independent finite-difference transfer audit are in `results/`.

## Limitations

These are numerical phase-field trajectories, not experimental microscopy or atomistic data. They support claims about transfer across parameter conditions, governing dynamics, discretization, and controlled initial-state noise within the documented two-dimensional setting.

## Integrity

File sizes, shapes, dtypes, field names, and SHA-256 hashes are recorded in `data/MANIFEST.json` and `MANIFEST.sha256`.
