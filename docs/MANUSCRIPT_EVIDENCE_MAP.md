# Manuscript evidence map

This map identifies the release artifact behind every figure and table in the current manuscript. Aggregate JSON files are the machine-readable records for numerical statements. Field-state figures additionally require the listed full trajectory data and representative checkpoints.

| Manuscript item | Evidence in this release |
|---|---|
| Figure 1, ST-NLR framework | `figures/artwork/FIG1_framework.svg`, vector PDF, and 1000 dpi PNG/TIFF; validator in `figures/prepare_figure1_svg.py` |
| Table 1, matched-budget Allen--Cahn | `results/static_rank_controls.json`, `results/allen_cahn_bootstrap.json` |
| Figure 2, time-resolved quality and rank | `results/figure_metadata/allen_cahn_time_resolved_quality.json`, `figures/make_time_resolved_quality_figure.py`, and final artwork |
| Figure 3, Allen--Cahn states | `data/full/allen_cahn/eps0022_lam14.npz`, representative Allen--Cahn checkpoints, `figures/material_state_inference.py`, `figures/make_state_figures.py`, and final artwork |
| Figure 4, free-energy and interface trajectories | Same representative Allen--Cahn data/checkpoints, `figures/make_material_observable_figure.py`, and final artwork |
| Table 2, arithmetic and latency | `results/efficiency_latency.json` |
| Figure 5, fixed-prefix Pareto relation | `results/static_rank_controls.json`, `figures/make_summary_figures.py` |
| Table 3, dynamic-capacity controls | `results/dynamic_baselines.json` |
| Table 4, Cahn--Hilliard transfer | `results/cahn_hilliard.json`, `results/paired_uncertainty.json` |
| Table 5, PFHub-3-type transfer | `results/pfhub3.json`, `results/paired_uncertainty.json` |
| Table B.6, solver refinement | `results/reference_convergence_ac_ch.json`, `results/reference_convergence_pfhub_standard.json`, `results/reference_convergence_pfhub_shifted.json` |
| Table C.7, full fine-tuning | `results/full_finetuning.json` |
| Figure D.6 and Table D.8, objective ablation | `results/objective_ablation.json`, portable runner and aggregator under `code/`, and `figures/make_objective_ablation_figure.py` |
| Figure E.7, data budget | `results/data_budget.json`, `figures/make_summary_figures.py` |
| Table F.9, independent solver and noisy inputs | `results/external_solver_noise.json` |
| Figure G.8, two-field morphology | `data/full/pfhub3/`, representative PFHub checkpoints, `figures/material_state_inference.py`, `figures/make_state_figures.py`, and final artwork |

All final artwork is stored in `figures/artwork/`. `MANIFEST.sha256` fixes the release state used for review.
