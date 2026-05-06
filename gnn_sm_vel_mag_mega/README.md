# GNN+MNN (MEGA) — stellar mass + |vel| + magnitudes

HaloGraphNet's GNN with the Moment Neural Network head, trained on all
four CAMELS-LH suites with the *photometric* node-feature vector: each
galaxy node carries position, stellar mass, |velocity|, the five SDSS *g r
i z* + UKIDSS *Y* magnitudes, and the *(g-r)*, *(r-i)* colours.

## Per-node features

| index | feature                                  | source                                       |
|-------|------------------------------------------|----------------------------------------------|
| 0..2  | normalised position                      | `Subhalo/SubhaloPos / boxsize`               |
| 3     | `log_stellar_mass`                       | `Subhalo/SubhaloMassType[:,4]`               |
| 4     | log10 \|velocity\|                       | `Subhalo/SubhaloVel`                         |
| 5..9  | 5 magnitudes (g, r, i, z, Y)             | BC03 attenuated SLOAN + UKIRT photometry     |
| 10    | colour `g-r`                             | derived                                      |
| 11    | colour `r-i`                             | derived                                      |

## Global features ``u``

Same as in [`gnn_sm_vel_mega/`](../gnn_sm_vel_mega/README.md#global-features-u).

## Architecture

Same as [`gnn_sm_vel_mega/`](../gnn_sm_vel_mega/README.md#architecture);
only the per-node feature dimensionality differs (12 vs. 5).

## Reproducing the model

Open `model.ipynb` and run all cells, or execute it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace gnn_sm_vel_mag_mega/model.ipynb
```

The notebook is fully self-contained — it loads halo graphs inline,
builds PyG `Data` objects, instantiates `ModelGNN` from the original
`HaloGraphNet/Source/networks.py`, trains with the 2-moment MNN loss,
and writes every artifact below.

## Layout of `test_predictions.npz`

Same as [`gnn_sm_vel_mega/`](../gnn_sm_vel_mega/README.md#layout-of-test_predictionsnpz).

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
