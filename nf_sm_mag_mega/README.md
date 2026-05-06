# NF (MEGA) — stellar mass + magnitudes

Conditional normalizing-flow trained on all four CAMELS-LH suites
(Astrid, IllustrisTNG, SIMBA, SWIFT-EAGLE) using stellar mass plus
SDSS *g, r, i, z* and UKIDSS *Y* magnitudes computed from BC03 attenuated
luminosities (``-2.5·log10(L) - 30``).

## Inputs

| Feature              | Source                                              | Units              |
|----------------------|-----------------------------------------------------|--------------------|
| `log_stellar_mass`   | `Subhalo/SubhaloMassType[:,4] * 1e10`               | log10 Msun/h       |
| `magnitude[SDSS.g]`  | `BC03/photometry/luminosity/attenuated/SLOAN/SDSS.g`| AB-like, mag       |
| `magnitude[SDSS.r]`  | "                                                   | "                  |
| `magnitude[SDSS.i]`  | "                                                   | "                  |
| `magnitude[SDSS.z]`  | "                                                   | "                  |
| `magnitude[UKIDSS.Y]`| `BC03/photometry/luminosity/attenuated/UKIRT/UKIDSS.Y`| "                |

Galaxies missing any band are dropped, as are simulation realisations that
have no photometry HDF5.

## Target

| Quantity             | Source                                              | Units              |
|----------------------|-----------------------------------------------------|--------------------|
| `log_halo_mass`      | `Group/Group_M_Crit200`                             | log10 Msun/h       |

## Reproducing the model

Open `model.ipynb` and run all cells, or execute it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace nf_sm_mag_mega/model.ipynb
```

The notebook is fully self-contained — it loads the CAMELS-LH groups
files inline, builds the feature matrix, defines a Pyro conditional
spline flow, trains it, draws 1000 posterior samples per test halo,
and writes every artifact below.

## Files

Same structure as [`nf_sm_only_mega/`](../nf_sm_only_mega/README.md);
the context vector has dimension 6 here (1 stellar mass + 5 magnitudes).

## Files in this folder

| File                                  | Origin                                                |
|---------------------------------------|-------------------------------------------------------|
| `model.ipynb`                         | runnable notebook: loads data, trains, infers, plots  |
| `best_model.pt`                       | weights + normalisation + arch + bands                |
| `optuna_study.pkl`                    | full Optuna study (joblib pickle)                     |
| `hyperparameters.txt`                 | human-readable summary of the chosen hyperparameters  |
| `test_predictions.npz`                | per-test-halo IDs, true mass, **1000 posterior samples** |
| `metrics.json`                        | overall + per-simulation metrics (RMSE, chi², PCC, ε, R²) |
| `plots/training_curves.png`           | NLL train vs test                                     |
| `plots/residuals_per_simulation.png`  | residual grid (one panel per sim) — MEGA only         |
| `plots/residuals_combined.png`        | combined residual scatter                             |
| `plots/tarp_coverage.png`             | TARP credibility curve + KS p-value                   |
| `plots/metrics_vs_mass.png`           | RMSE and chi² vs true halo mass                       |
