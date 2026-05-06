# Comparison of the five MEGA models

This folder builds the cross-model comparison plots and metric tables that
are surfaced in the top-level [README](../README.md).

| short name              | family       | conditioning features                              |
|-------------------------|--------------|----------------------------------------------------|
| `nf_sm_only_mega`       | NF (central) | log10 stellar mass                                 |
| `nf_sm_pos_vel_mega`    | NF (central) | log10 stellar mass, \|pos\|, \|vel\|               |
| `nf_sm_mag_mega`        | NF (central) | log10 stellar mass, *g r i z* + *Y* magnitudes     |
| `gnn_sm_vel_mega`       | GNN+MNN      | per-galaxy: log10 stellar mass, \|vel\|            |
| `gnn_sm_vel_mag_mega`   | GNN+MNN      | per-galaxy: SM, \|vel\|, *g r i z Y* + colours     |

Every model uses the **same train/test split** (seed = 42, see
`shared.splits`), so test predictions can be overlaid directly.

## How to run

After every model's ``infer.py`` has produced ``test_predictions.npz``:

```bash
python release/compare_models/make_comparison.py
```

## Outputs

| File                                  | Description                                     |
|---------------------------------------|-------------------------------------------------|
| `plots/residuals_overlay.png`         | Pred − True vs True for all five models         |
| `plots/metrics_vs_mass.png`           | RMSE and chi² versus halo-mass bin per model    |
| `plots/posterior_examples.png`        | 1D posterior overlays for a few shared halos    |
| `plots/metrics_table.csv`             | Wide table: overall + per simulation + per model|
| `plots/metrics_table.md`              | Same table formatted as Markdown                |
