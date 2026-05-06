# NF (IllustrisTNG) - stellar mass + |pos| + |vel|

Per-simulation version trained **only** on the IllustrisTNG CAMELS-LH suite,
using the same three features as
[`nf_sm_pos_vel_mega/`](../../nf_sm_pos_vel_mega/README.md).

## Reproducing the model

Open `model.ipynb` and run all cells, or execute it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace per_sim/illustristng_nf_sm_pos_vel/model.ipynb
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
