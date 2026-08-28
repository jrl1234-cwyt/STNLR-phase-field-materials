# Reproducibility record

## Hardware used for the reported runs

- 2 x NVIDIA Tesla V100 PCIe, 32 GB each
- NVIDIA driver 580.173.02
- CUDA runtime compiled with PyTorch: 11.8

## Training configuration

- Main batch size: 4; PFHub batch size: 2
- Optimizer: AdamW
- Initial learning rate: `3e-4`; PFHub: `2e-4`
- Weight decay: 0
- Schedule: cosine annealing without warm-up
- Gradient norm limit: 1.0
- Static initialization: 1000 steps; PFHub: 1500 steps
- Continued-static and ST-NLR calibration: 600 steps
- Training seeds: 0, 1, 2
- Multi-prefix weights for ranks 16, 8, 4: 1, 0.5, 0.25
- LoRA alpha: 1; scale at rank r: 1/r

The exact material-loss weights, spectral normalization, additive validation constants, and system-specific tolerances are implemented in `code/` and reported in the manuscript.

The reported Allen--Cahn objective ablation uses `(epsilon, lambda)=(0.022,1.4)`, seeds 0, 1, and 2, the paired static rank-16 start, and 600 calibration steps for every variant. The release provides the aggregate record, portable runner and aggregator, and the seed-0 rank-16 start as the representative checkpoint. The two additional paired starts can be regenerated with the same static-rank protocol before a full three-seed rerun.

## External dependency provenance

- Poseidon source: `https://github.com/camlab-ethz/poseidon.git`
- Source commit used by this release: `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`
- Pretrained model: `camlab-ethz/Poseidon-T`
- Expected main pretrained-weight SHA-256: `a363f7317fbc3a900a318fc63cc53197705d95fce0e0ce28dd3c8844a89112e2`

## Statistical protocol

The hierarchical bootstrap uses 10,000 replicates and seed 20260825. It resamples target conditions, training seeds within conditions, and paired test trajectories within condition-seed cells. Rank schedules are selected on validation data and frozen before test evaluation.
