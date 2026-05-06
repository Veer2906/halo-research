"""Cross-model comparison plots and metric tables.

Reads ``test_predictions.npz`` from each MEGA model folder and produces:

  * ``plots/residuals_overlay.png``   --- residuals vs true mass for all models
  * ``plots/metrics_table.csv``       --- overall + per-simulation metrics
  * ``plots/metrics_table.md``        --- same table, markdown for README inclusion
  * ``plots/metrics_vs_mass.png``     --- RMSE and chi^2 vs halo-mass bin
  * ``plots/posterior_examples.png``  --- 1D posterior overlays for a few halos

Run *after* every model's ``infer.py`` has produced its ``test_predictions.npz``.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from shared.metrics import compute_metrics, metrics_in_bins  # noqa: E402


MODELS = [
    ("nf_sm_only_mega",        "NF — SM",                       "C0"),
    ("nf_sm_pos_vel_mega",     "NF — SM + |pos| + |vel|",       "C1"),
    ("nf_sm_mag_mega",         "NF — SM + magnitudes",          "C2"),
    ("gnn_sm_vel_mega",        "GNN+MNN — SM + |vel|",          "C3"),
    ("gnn_sm_vel_mag_mega",    "GNN+MNN — SM + |vel| + mags",   "C4"),
]


def load_predictions(model_dir: str) -> Dict[str, np.ndarray]:
    p = np.load(os.path.join(model_dir, "test_predictions.npz"), allow_pickle=True)
    return {
        "simulation": np.asarray(p["simulation"], dtype=str),
        "halo_id":    np.asarray(p["halo_id"], dtype=str),
        "true":       np.asarray(p["true_log_halo_mass"], dtype=np.float64),
        "mu":         np.asarray(p["posterior_mean"], dtype=np.float64),
        "sd":         np.asarray(p["posterior_std"], dtype=np.float64),
        "samples":    np.asarray(p["posterior_log_halo_mass"], dtype=np.float64),
    }


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    preds: Dict[str, dict] = {}
    for sub, label, _ in MODELS:
        path = os.path.join(ROOT, sub, "test_predictions.npz")
        if not os.path.exists(path):
            print(f"[compare] WARN: {path} missing -- skip {label}")
            continue
        preds[sub] = load_predictions(os.path.join(ROOT, sub))
        print(f"[compare] loaded {label}: N={len(preds[sub]['true'])}")

    if not preds:
        print("[compare] nothing to compare")
        return

    # ---------- Residuals overlay
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for sub, label, color in MODELS:
        if sub not in preds:
            continue
        d = preds[sub]
        ax.scatter(d["true"], d["mu"] - d["true"], s=8, alpha=0.4,
                   color=color, edgecolors="none", label=label)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel(r"True $\log_{10}\,M_{\rm halo}$")
    ax.set_ylabel(r"Pred $-$ True $\log_{10}\,M_{\rm halo}$")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Residuals on the held-out test set")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "residuals_overlay.png"), dpi=150)
    plt.close(fig)

    # ---------- Metrics table
    rows: List[dict] = []
    for sub, label, _ in MODELS:
        if sub not in preds:
            continue
        d = preds[sub]
        m = compute_metrics(d["true"], d["mu"], d["sd"])
        rows.append({"model": label, "scope": "Overall", "N": m.n,
                     "RMSE": m.rmse, "chi2": m.chi2, "PCC": m.pcc,
                     "eps": m.eps, "R2": m.r2})
        for sim in sorted(set(d["simulation"].tolist())):
            sel = d["simulation"] == sim
            mm = compute_metrics(d["true"][sel], d["mu"][sel], d["sd"][sel])
            rows.append({"model": label, "scope": sim, "N": mm.n,
                         "RMSE": mm.rmse, "chi2": mm.chi2, "PCC": mm.pcc,
                         "eps": mm.eps, "R2": mm.r2})

    csv_path = os.path.join(plots_dir, "metrics_table.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["model", "scope", "N",
                                            "RMSE", "chi2", "PCC", "eps", "R2"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    md_path = os.path.join(plots_dir, "metrics_table.md")
    with open(md_path, "w") as fh:
        fh.write("| Model | Scope | N | RMSE | chi2 | PCC | eps (%) | R2 |\n")
        fh.write("|-------|-------|---|------|------|-----|---------|----|\n")
        for r in rows:
            fh.write(f"| {r['model']} | {r['scope']} | {r['N']} | "
                     f"{r['RMSE']:.3f} | {r['chi2']:.3f} | {r['PCC']:.3f} | "
                     f"{r['eps']:.2f} | {r['R2']:.3f} |\n")

    print(f"[compare] wrote {csv_path}")
    print(f"[compare] wrote {md_path}")

    # ---------- Metrics vs halo-mass bins
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for sub, label, color in MODELS:
        if sub not in preds:
            continue
        d = preds[sub]
        c, rmse_b = metrics_in_bins(d["true"], d["mu"], d["sd"], metric="rmse")
        _, chi_b  = metrics_in_bins(d["true"], d["mu"], d["sd"], metric="chi2")
        axes[0].plot(c, rmse_b, "o-", label=label, color=color)
        axes[1].plot(c, chi_b,  "o-", label=label, color=color)
    axes[0].set_xlabel(r"True $\log_{10}\,M_{\rm halo}$"); axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3); axes[0].set_title("RMSE in halo-mass bins")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"True $\log_{10}\,M_{\rm halo}$"); axes[1].set_ylabel(r"$\chi^2$")
    axes[1].axhline(1.0, color="k", lw=1, ls="--")
    axes[1].grid(alpha=0.3); axes[1].set_title(r"$\chi^2$ in halo-mass bins")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "metrics_vs_mass.png"), dpi=150)
    plt.close(fig)

    # ---------- Posterior overlays for a few common halos
    common: set = set.intersection(*[set(d["halo_id"].tolist()) for d in preds.values()])
    if common:
        rng = np.random.default_rng(0)
        chosen = sorted(common)
        chosen = list(rng.choice(chosen, size=min(6, len(chosen)), replace=False))
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes = axes.ravel()
        for i, hid in enumerate(chosen):
            ax = axes[i]
            for sub, label, color in MODELS:
                if sub not in preds:
                    continue
                d = preds[sub]
                idx = np.where(d["halo_id"] == hid)[0]
                if not len(idx):
                    continue
                k = idx[0]
                ax.hist(d["samples"][k], bins=40, density=True,
                        histtype="step", color=color, label=label)
            true = preds[next(iter(preds))]["true"][
                np.where(preds[next(iter(preds))]["halo_id"] == hid)[0][0]]
            ax.axvline(true, color="k", ls="--", lw=1.2, label="truth")
            ax.set_title(hid, fontsize=9)
            ax.set_xlabel(r"$\log_{10} M_{\rm halo}$"); ax.set_ylabel("density")
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.suptitle("Per-halo posterior overlays (random shared sample)",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "posterior_examples.png"), dpi=150)
        plt.close(fig)
        print(f"[compare] wrote posterior_examples.png with {len(chosen)} halos")
    else:
        print("[compare] no shared halo_id between models -- skip posterior_examples.png")


if __name__ == "__main__":
    main()
