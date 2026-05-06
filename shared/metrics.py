"""Single-point metrics used for model comparison.

All metrics operate on log10 halo mass, in line with the convention used
throughout the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy import stats


@dataclass
class Metrics:
    rmse:       float
    chi2:       float
    pcc:        float
    eps:        float       # mean relative error (%) on linear halo mass
    r2:         float
    n:          int

    def as_dict(self) -> dict:
        return {
            "RMSE":  self.rmse,
            "chi2":  self.chi2,
            "PCC":   self.pcc,
            "eps":   self.eps,
            "R2":    self.r2,
            "N":     self.n,
        }


def compute_metrics(true_log_hm: np.ndarray,
                    pred_log_hm: np.ndarray,
                    pred_log_std: Optional[np.ndarray] = None) -> Metrics:
    """Return single-point metrics over ``true_log_hm`` and ``pred_log_hm``.

    Parameters
    ----------
    true_log_hm:
        True log10 halo mass.
    pred_log_hm:
        Predicted log10 halo mass (typically the posterior mean).
    pred_log_std:
        Posterior standard deviation in log10 halo mass. Required for
        ``chi2``; if absent, ``chi2`` is set to ``nan``.
    """
    true_log_hm = np.asarray(true_log_hm).reshape(-1)
    pred_log_hm = np.asarray(pred_log_hm).reshape(-1)
    n = len(true_log_hm)
    if pred_log_hm.shape != true_log_hm.shape:
        raise ValueError("true_log_hm and pred_log_hm must have the same shape")

    residuals = pred_log_hm - true_log_hm
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    pcc  = float(stats.pearsonr(true_log_hm, pred_log_hm)[0])

    eps = float(np.mean(np.abs(10 ** pred_log_hm - 10 ** true_log_hm) / 10 ** true_log_hm) * 100.0)

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((true_log_hm - true_log_hm.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if pred_log_std is not None:
        std = np.asarray(pred_log_std).reshape(-1)
        std = np.where(std <= 0, 1e-6, std)
        chi2 = float(np.mean((residuals / std) ** 2))
    else:
        chi2 = float("nan")

    return Metrics(rmse=rmse, chi2=chi2, pcc=pcc, eps=eps, r2=r2, n=n)


def metrics_in_bins(true_log_hm: np.ndarray,
                    pred_log_hm: np.ndarray,
                    pred_log_std: Optional[np.ndarray] = None,
                    bins: Optional[np.ndarray] = None,
                    metric: str = "rmse",
                    min_count: int = 10) -> tuple:
    """Return ``(bin_centers, values)`` of a metric in halo-mass bins.

    Bins span the range of ``true_log_hm`` if not supplied.  Bins with fewer
    than ``min_count`` halos are reported as ``nan``.
    """
    if bins is None:
        lo = np.floor(true_log_hm.min() * 2) / 2
        hi = np.ceil(true_log_hm.max() * 2) / 2
        bins = np.arange(lo, hi + 0.25, 0.25)

    centers = 0.5 * (bins[:-1] + bins[1:])
    out = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        sel = (true_log_hm >= bins[i]) & (true_log_hm < bins[i + 1])
        if sel.sum() < min_count:
            continue
        m = compute_metrics(true_log_hm[sel], pred_log_hm[sel],
                            pred_log_std[sel] if pred_log_std is not None else None)
        out[i] = getattr(m, metric)
    return centers, out
