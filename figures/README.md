# Figure reproduction

`artwork/` contains the exact figure files used by the current manuscript. Plot PDFs are vector artwork. The two representative-state PDFs embed their raster field maps above 1000 pixels per inch at manuscript size. Figure 1 is supplied as an editable SVG and vector PDF, together with 7491 x 3959 PNG/TIFF exports rendered at 1000 dpi.

The lightweight aggregate plots can be regenerated on CPU with

```bash
python figures/make_summary_figures.py
python figures/make_objective_ablation_figure.py
python figures/make_time_resolved_quality_figure.py
```

These commands regenerate the data-budget, fixed-prefix audit, objective-ablation, and time-resolved plots from `results/`. Figure 5 reads `results/fixed_prefix_pareto.json`; the corresponding evaluator records validation feasibility and independent test metrics before plotting. `prepare_figure1_svg.py` validates the editable Figure 1 source; the final source is `artwork/FIG1_framework.svg`.

The material-observable and field-state figures require the bundled checkpoints and the POSEIDON dependency described in `docs/REPRODUCIBILITY.md`. They can be regenerated with

```bash
python figures/make_material_observable_figure.py
python figures/make_state_figures.py
```

Their numerical trajectories and representative checkpoints are in `data/full/` and `checkpoints/`. `material_state_inference.py` links the plotting workflow to the model definitions under `code/`. Machine-readable sample-selection and per-time plotting records are stored in `results/figure_metadata/`.
