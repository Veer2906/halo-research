# Per-simulation NF runs

Every MEGA NF in `release/` is trained on **all four** CAMELS-LH suites.
The folders in here are single-suite re-runs of the
[`nf_sm_pos_vel_mega`](../nf_sm_pos_vel_mega/README.md) model, one per
suite.

| Folder                            | Suite        |
|-----------------------------------|--------------|
| `astrid_nf_sm_pos_vel/`           | Astrid       |
| `illustristng_nf_sm_pos_vel/`     | IllustrisTNG |
| `simba_nf_sm_pos_vel/`            | SIMBA        |
| `swift_eagle_nf_sm_pos_vel/`      | SWIFT-EAGLE  |

Each folder has the same three-script interface as the MEGA models:

```bash
python release/per_sim/<folder>/train.py
python release/per_sim/<folder>/infer.py
python release/per_sim/<folder>/analysis.py
```

They produce the exact same artefacts (`best_model.pt`,
`test_predictions.npz`, `plots/*.png`, `metrics.json`) so their outputs can
be picked up by `compare_models/` or any ad-hoc notebook unchanged.
