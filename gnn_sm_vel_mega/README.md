# GNN+MNN (MEGA) — stellar mass + |vel|



## Per-node features

| index | feature                | source                                  |
|-------|------------------------|-----------------------------------------|
| 0..2  | normalised position    | `Subhalo/SubhaloPos / boxsize` (then halo-centred) |
| 3     | `log_stellar_mass`     | `Subhalo/SubhaloMassType[:,4]`          |
| 4     | log10 \|velocity\|     | `Subhalo/SubhaloVel` minus halo CM, normalised by 100 km/s |

## Global features ``u``

| index | feature                                                |
|-------|--------------------------------------------------------|
| 0     | number of galaxies in halo                             |
| 1     | log10 of total stellar mass within halo                |

## Architecture

| Field                | Value      |
|----------------------|------------|
| `use_model`          | `EdgeNet`  |
| `n_layers`           | `2`        |
| `k_nn` (radius_graph)| `5.0`      |
| `hidden_channels`    | `300`      |
| `latent_channels`    | `100`      |
| Final head           | MLP -> (μ, σ) |
| Loss                 | Moment loss `log(MSE) + log(LFI)` |

## Reproducing the model

Open `model.ipynb` and run all cells, or execute it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace gnn_sm_vel_mega/model.ipynb
```

The notebook is fully self-contained — it loads halo graphs inline,
builds PyG `Data` objects, instantiates `ModelGNN` from the original
`HaloGraphNet/Source/networks.py`, trains with the 2-moment MNN loss,
and writes every artifact below.

## Layout of `test_predictions.npz`

The MNN head emits ``(μ, σ)`` per halo, so we additionally draw 1000
Gaussian samples from ``N(μ, σ²)`` for the TARP test.

| key                          | shape              | description                                     |
|------------------------------|--------------------|-------------------------------------------------|
| `simulation`                 | `(N_test,)`        | suite tag                                       |
| `halo_id`                    | `(N_test,)`        | unique `<sim>|LH<i>|halo<j>`                    |
| `galaxy_ids`                 | `(N_test,)` object | array of `<sim>|LH<i>|sub<k>` per halo          |
| `true_log_halo_mass`         | `(N_test,)`        | ground-truth log10 M_halo                       |
| `posterior_mean`             | `(N_test,)`        | μ from the MNN head                             |
| `posterior_std`              | `(N_test,)`        | σ from the MNN head                             |
| `posterior_log_halo_mass`    | `(N_test, 1000)`   | Gaussian draws used for TARP                    |
| `n_posterior`                | scalar             | always 1000                                     |
| `model_name`                 | scalar             | `"GNN+MNN (MEGA) – stellar mass + |vel|"`       |

## Files in this folder

| File                                  | Origin                                                |
|---------------------------------------|-------------------------------------------------------|
| `model.ipynb`                         | runnable notebook: loads data, trains, infers, plots  |
| `best_model.pt`                       | weights + arch + feature flags + normalisers          |
| `hyperparameters.txt`                 | human-readable summary                                |
| `test_predictions.npz`                | (μ, σ) per test halo + 1000 Gaussian posterior draws  |
| `metrics.json`                        | overall + per-simulation metrics                      |
| `plots/training_curves.png`           | MNN-loss train vs val                                 |
| `plots/residuals_per_simulation.png`  | residual grid per sim                                 |
| `plots/residuals_combined.png`        | combined residual scatter                             |
| `plots/tarp_coverage.png`             | TARP credibility curve + KS p-value                   |
| `plots/metrics_vs_mass.png`           | RMSE and chi² vs true halo mass                       |
