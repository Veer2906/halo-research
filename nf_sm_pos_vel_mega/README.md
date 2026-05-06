# NF (MEGA) — stellar mass + |pos| + |vel|

Conditional normalizing-flow trained on all four CAMELS-LH suites with
**three** central-galaxy features.

## Inputs

| Feature              | Source                                       | Units               |
|----------------------|----------------------------------------------|---------------------|
| `log_stellar_mass`   | `Subhalo/SubhaloMassType[:,4] * 1e10`        | log10 Msun/h        |
| `log_pos_modulus`    | log10 ‖`Subhalo/SubhaloPos`‖                 | log10 kpc/h         |
| `log_vel_modulus`    | log10 ‖`Subhalo/SubhaloVel`‖                 | log10 km/s          |

## Target

| Quantity             | Source                                       | Units               |
|----------------------|----------------------------------------------|---------------------|
| `log_halo_mass`      | `Group/Group_M_Crit200`                      | log10 Msun/h        |

## Reproducing the model

Open `model.ipynb` and run all cells, or execute it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace nf_sm_pos_vel_mega/model.ipynb
```

The notebook is fully self-contained — it loads the CAMELS-LH groups
files inline, builds the feature matrix, defines a Pyro conditional
spline flow, trains it, draws 1000 posterior samples per test halo,
and writes every artifact below.

## Files

Same structure as [`nf_sm_only_mega/`](../nf_sm_only_mega/README.md); only
the feature columns and resulting context dimensionality differ.

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
