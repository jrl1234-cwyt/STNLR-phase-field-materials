# ST-NLR phase-field transfer: FAIR research package

This package supports the manuscript *Physical-time-adaptive transfer of phase-field foundation models to unseen material conditions*. It contains the code path, complete phase-field datasets used by the reported systems, machine-readable aggregate results, one paired representative checkpoint per reported system, exact environment records, and checksum verification.

## Contents

- `code/`: dataset generation, training, evaluation, uncertainty, solver-audit, and aggregation programs.
- `data/full/`: Allen--Cahn, Cahn--Hilliard, and PFHub-3-type trajectory datasets.
- `checkpoints/`: paired ST-NLR and comparison checkpoints for one representative condition of each system, including the Allen--Cahn rank-16 start and seed-0 strict AdaLoRA, DyLoRA, and TimeStep Master baselines.
- `results/`: aggregate and paired per-sample JSON records used by the tables, figures, and statistical statements.
- `figures/`: scripts used to regenerate manuscript figures from the aggregate records.
- `environment/`: pinned Python/CUDA environment and container recipe.
- `docs/`: dataset, model, reproducibility, and FAIR records.
- `docs/MANUSCRIPT_EVIDENCE_MAP.md`: traceability from every manuscript figure and table to its aggregate record, code path, or representative artifact.
- `MANIFEST.sha256`: integrity record for every release file.

No real-material or experimental-microscopy claim is made by this release. All included trajectories come from the stated phase-field solvers.

## Quick start

```bash
conda env create -f environment/environment.yml
conda activate stnlr-cms
bash scripts/download_poseidon.sh external
python scripts/verify_release.py
PYTHONPATH=code python scripts/smoke_test.py
python scripts/summarize_results.py
python figures/make_summary_figures.py
python figures/make_objective_ablation_figure.py
```

The Poseidon source and pretrained weights are external dependencies and are not redistributed. The download script fixes the source commit and records the expected pretrained-weight checksum.

The fixed-prefix audit behind manuscript Figure 5 can be recomputed from the released representative checkpoint without retraining:

```bash
PYTHONPATH=code/materials_phasefield_stnlr_application_20260813/experiments:code \
python code/materials_phasefield_stnlr_application_20260813/experiments/evaluate_fixed_prefix_pareto.py
```

## Reproducing a dataset

For example, the fourth Allen--Cahn condition can be regenerated with

```bash
PYTHONPATH=code python code/materials_phasefield_stnlr_application_20260813/experiments/generate_allen_cahn_trajectories.py \
  --out-dir regenerated/eps0022_lam16 --only-target --seed 20260816 \
  --target-epsilon 0.022 --target-reaction 1.6
```

The target generator uses `seed + 1`, giving the documented target-data seed 20260817.

## Publishing

Track `*.pt` and `*.npz` with Git LFS. Publish the source repository as a versioned GitHub release, then archive the same release and complete datasets on Zenodo. Add the Zenodo DOI to this README, `CITATION.cff`, and the manuscript data-availability statement. GitHub alone provides access and version control; the Zenodo archive supplies the persistent identifier required for a complete FAIR deposit.

Repository: <https://github.com/jrl1234-cwyt/STNLR-phase-field-materials>. After Zenodo mints the archival DOI, add it as `doi` in `CITATION.cff` and replace the provisional wording in the manuscript's data-and-code availability statement. The concrete release sequence is recorded in `docs/GITHUB_RELEASE.md`.

## License and provenance

The package uses the repository's CC BY-NC 4.0 license; see `LICENSE`. Poseidon and other external dependencies retain their own licenses. Checkpoint use is also subject to the upstream Poseidon license. The exact code provenance and hardware/software environment are recorded in `docs/REPRODUCIBILITY.md`.
