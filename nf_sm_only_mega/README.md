# NF (MEGA) — stellar mass only

Conditional normalizing-flow trained on **all four** CAMELS-LH suites
(Astrid, IllustrisTNG, SIMBA, SWIFT-EAGLE) using only the central galaxy's
log10 stellar mass as context.  This is the simplest of the five MEGA
comparison models; it serves as a baseline.

## Inputs

| Feature              | Source                               | Units               |
|----------------------|--------------------------------------|---------------------|
| `log_stellar_mass`   | `Subhalo/SubhaloMassType[:,4] * 1e10`| log10 Msun/h        |

## Target

| Quantity             | Source                               | Units               |
|----------------------|--------------------------------------|---------------------|
| `log_halo_mass`      | `Group/Group_M_Crit200`              | log10 Msun/h        |

## Reproducing the model

Open `model.ipynb` and run all cells, or execute it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace nf_sm_only_mega/model.ipynb
```

The notebook is fully self-contained — it loads the CAMELS-LH groups
files inline, builds the feature matrix, defines a Pyro conditional
spline flow, trains it, draws 1000 posterior samples per test halo,
and writes every artifact below.

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

## Layout of `test_predictions.npz`

| key                          | shape              | description                                     |
|------------------------------|--------------------|-------------------------------------------------|
| `simulation`                 | `(N_test,)`        | "Astrid"/"IllustrisTNG"/"SIMBA"/"SWIFT-EAGLE"   |
| `halo_id`                    | `(N_test,)`        | unique `<sim>|LH<i>|halo<j>`                    |
| `galaxy_id`                  | `(N_test,)`        | unique `<sim>|LH<i>|sub<k>`                     |
| `true_log_halo_mass`         | `(N_test,)`        | ground-truth log10 M_halo                       |
| `posterior_log_halo_mass`    | `(N_test, 1000)`   | posterior samples in log10 Msun/h               |
| `posterior_mean`             | `(N_test,)`        | mean of posterior                                |
| `posterior_std`              | `(N_test,)`        | std of posterior                                 |
| `feature_names`              | `(F,)`             | column labels of the context vector             |
| `feature_values`             | `(N_test, F)`      | the actual context values fed to the flow       |
| `n_posterior`                | scalar             | always 1000                                     |
| `model_name`                 | scalar             | `"NF (MEGA) – stellar mass only"`               |
