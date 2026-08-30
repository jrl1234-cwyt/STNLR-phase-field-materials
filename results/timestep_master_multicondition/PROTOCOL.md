# Frozen protocol: four-condition TimeStep Master control

## Purpose

Extend the strongest published timestep-aware baseline from the representative Allen--Cahn target to all four unseen target conditions, and compute paired hierarchical-bootstrap intervals against ST-NLR.

## Fixed design

- Targets: `eps0022_lam14`, `eps0022_lam16`, `eps0028_lam14`, `eps0028_lam16`.
- Training seeds: 0, 1, 2.
- Data split per target: 100 adaptation, 20 validation, 40 paired test trajectories.
- Static start: rank 16, 1000 updates, batch 4, AdamW, learning rate 5e-4, zero weight decay, cosine schedule, gradient clipping at 1.0, energy/interface weights 0.04/0.02.
- TimeStep Master: published `n1=8`, `n2=1`, `r=alpha=4` configuration. Eight fine-interval core experts and one global context expert are trained for 600 fostering updates. The experts are then frozen and the feature-and-time router is trained for 600 assembling updates.
- Material calibration: batch 4, AdamW, learning rate 3e-4, zero weight decay, cosine schedule, gradient clipping at 1.0, and the same field, energy, interface, field-distillation, and spectral-distillation objectives used in the representative strict control.
- Test metrics: trajectory relative L2, terminal relative L2, terminal interface-fraction MAE, and terminal relative free-energy error.
- Inference: paired bootstrap preserving target condition, training seed, and the 40 paired test trajectories; 10,000 replicates; seed 20260830.

The representative condition reuses the three completed strict TimeStep Master checkpoints and is re-evaluated only to recover per-trajectory metrics. The remaining nine target--seed cells regenerate their paired static starts because those checkpoints are no longer stored.

Results are eligible for the manuscript only after all 12 cells complete and the aggregate file passes consistency checks.
