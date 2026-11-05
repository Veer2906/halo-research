# Final halo-mass-inference release

Self-contained drop of everything needed to reproduce the five MEGA models
that we trained on CAMELS-LH data for the central-galaxy / halo-mass
regression problem, plus the per-simulation baselines and the
cross-model comparison plots.

```bash
jupyter lab <folder>/model.ipynb   # or:  jupyter nbconvert --to notebook --execute <folder>/model.ipynb
```

The notebooks are intentionally inline (zero runtime imports from
`shared/`) so a reader can audit one model end-to-end in a single
file. `shared/metrics.py` exists only so `compare_models/` doesn't
duplicate the RMSE/χ²/PCC/ε/R² code.

## Identifier convention

Every halo and every galaxy gets a stable string id so predictions from
different models can be joined without ambiguity:

| id type      | format                        | example                    |
|--------------|-------------------------------|----------------------------|
| halo id      | `<sim>|LH<lh>|halo<h>`        | `Astrid|LH3|halo57`        |
| galaxy id    | `<sim>|LH<lh>|sub<s>`         | `SIMBA|LH14|sub902`        |

`lh` is the CAMELS-LH realisation index recovered from the filename
(`groups_090 (LH).hdf5`), `h` is the group index inside that realisation,
`s` is the subhalo index inside that realisation.

## Train/test split

The split is deterministic for every script:

* `shared.splits.index_split(N)` shuffles `range(N)` with seed `42`,
  keeps the last 80% as train, the first 20% as test.
* All MEGA NF models that share the same feature set (all four suites +
  same galaxy-quality cuts) therefore see exactly the same test-set
  halos, which is what lets `compare_models/` overlay them cleanly.
* Per-simulation NF runs apply the split to the filtered subset of halos
  for that one suite. Their test halos are a subset of the MEGA test set
  iff the selection criteria match (they do for all per-sim folders in
  this release).
* GNN runs use 80% train / 10% val (carved from train) / 10% test with
  the same seed. The test halos are the same as the NF MEGA models *only*
  up to the `min_galaxies=1` filter; in practice this is the
  overwhelming majority.

## Outputs produced by every leaf folder

| file                                | description                                          |
|-------------------------------------|------------------------------------------------------|
| `model.ipynb`                       | runnable notebook: data loading + training + inference + plots |
| `best_model.pt`                     | torch state dict + normalisation + arch + bands      |
| `hyperparameters.txt`               | one-file human-readable summary                      |
| `optuna_study.pkl` (NF only)        | joblib pickle of the Optuna study                    |
| `test_predictions.npz`              | per-test-halo IDs, mean, std, 1000 posterior samples |
| `metrics.json`                      | overall + per-simulation metrics                     |
| `plots/training_curves.png`         | loss vs epoch                                        |
| `plots/residuals_combined.png`      | residual scatter, full test set                      |
| `plots/residuals_per_simulation.png`| residuals split by suite (2×2 grid; MEGA only)       |
| `plots/tarp_coverage.png`           | TARP curve + KS p-value                              |
| `plots/metrics_vs_mass.png`         | RMSE and chi² in halo-mass bins                      |

## Cross-model comparison

```bash
python release/compare_models/make_comparison.py
```

Reads `test_predictions.npz` from each MEGA model folder and produces:

* `plots/residuals_overlay.png` – residuals for all five models on one axes
* `plots/metrics_vs_mass.png` – RMSE and chi² vs halo-mass per model
* `plots/metrics_table.{csv,md}` – overall + per-simulation metrics for all models
* `plots/posterior_examples.png` – 1D posterior overlays for a handful of shared halos

## End-to-end reproduction from scratch

```bash
# 0. install (see requirements.txt)
pip install -r release/requirements.txt

# 1. run every per-model notebook (training + inference + plots in one pass each)
for f in release/nf_sm_only_mega release/nf_sm_pos_vel_mega release/nf_sm_mag_mega \
         release/gnn_sm_vel_mega release/gnn_sm_vel_mag_mega \
         release/per_sim/*/; do
    jupyter nbconvert --to notebook --execute --inplace "${f%/}/model.ipynb"
done

# 2. cross-model overlay + tables (uses each test_predictions.npz)
python release/compare_models/make_comparison.py
```

Individual cells inside `model.ipynb` are also independent: open one notebook,
run the data-loading cells once, and you can re-execute the plotting / TARP
cells without retraining. `make_comparison.py` only depends on the
`test_predictions.npz` files.

## Notes

* **CUDA**: every notebook picks the best available device automatically.
* **Photometry files**: the two magnitude-aware models skip groups files
  that have no matching `<prefix>_LH_<i>_photometry.hdf5`, so the test
  set for those models is strictly smaller than for the plain-SM models.
* **HaloGraphNet dependency**: the GNN notebooks add `HaloGraphNet/` to
  `sys.path` so they can import the original `ModelGNN`. Don't move or
  rename that folder.
