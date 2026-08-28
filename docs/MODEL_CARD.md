# Model card

## Model

ST-NLR adapts a frozen Poseidon-T multi-PDE backbone with one ordered maximum-rank factor bank. Prefixes of ranks 4, 8, and 16 share directions. Validation selects the smallest feasible prefix at each physical output time using field and material-observable tolerances.

## Included checkpoints

The release includes one paired ST-NLR/comparison pair for a representative condition in each system:

- Allen--Cahn `(epsilon, lambda)=(0.022,1.4)`, seed 0: ST-NLR, static rank 7, and the paired static rank-16 objective-ablation start.
- Cahn--Hilliard `(epsilon, lambda)=(0.020,1.0)`, seed 0: ST-NLR and continued-static rank 16.
- PFHub-3-type shifted condition, seed 0: material-calibrated ST-NLR and continued-static.

The complete multi-condition and three-seed numerical results are supplied as aggregate records. The pretrained Poseidon weights are downloaded separately because they retain upstream provenance and licensing.

## Evaluation

Evaluation covers field errors, interface measures, free energy, conservation, structure factor, solid fraction, tip position, PDE residual, active rank, adapter arithmetic, total model arithmetic, and measured latency. Statistical intervals preserve the condition, seed, and paired-trajectory hierarchy.
