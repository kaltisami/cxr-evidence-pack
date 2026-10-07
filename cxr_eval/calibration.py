"""Calibration of a binary output: ECE, reliability curve, Brier score, temperature scaling.

Inputs are raw logits. TorchXRayVision's default op_norm output is not a probability, so calibration is only
meaningful on logits from a model with `op_threshs = None` (see scripts/01_inference.py).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit


def reliability(y, p, n_bins=15, strategy="uniform"):
    """Per-bin mean predicted probability, observed frequency and count.

    strategy: "uniform" (equal-width bins on [0, 1]) or "quantile" (equal-count bins).
    Empty bins are dropped.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(strategy)
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    count = np.bincount(b, minlength=n_bins)
    keep = count > 0
    mean_p = np.bincount(b, p, n_bins)[keep] / count[keep]
    freq = np.bincount(b, y, n_bins)[keep] / count[keep]
    return {"mean_predicted": mean_p, "observed": freq, "count": count[keep]}


def ece(y, p, n_bins=15, strategy="uniform"):
    """Expected calibration error: count-weighted mean |observed - predicted| over bins."""
    r = reliability(y, p, n_bins, strategy)
    return float(np.sum(r["count"] * np.abs(r["observed"] - r["mean_predicted"])) / r["count"].sum())


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def nll(y, logits):
    """Mean binary cross-entropy computed from logits (numerically stable)."""
    y, z = np.asarray(y, float), np.asarray(logits, float)
    return float(-np.mean(y * log_expit(z) + (1 - y) * log_expit(-z)))


def fit_temperature(y, logits, bounds=(0.05, 20.0)):
    """Temperature T > 0 minimising the NLL of sigmoid(logits / T). Fit on validation data only.

    Temperature scaling is monotone, so it changes calibration but leaves AUROC/AUPRC unchanged.
    """
    res = minimize_scalar(lambda log_t: nll(y, np.asarray(logits) / np.exp(log_t)),
                          bounds=np.log(bounds), method="bounded")
    return float(np.exp(res.x))


def calibration_summary(y, logits, temperature=1.0, n_bins=15):
    """ECE (uniform and quantile bins), Brier and NLL of sigmoid(logits / temperature)."""
    z = np.asarray(logits, float) / temperature
    p = expit(z)
    return {"temperature": temperature,
            "ece": ece(y, p, n_bins, "uniform"),
            "ece_quantile": ece(y, p, n_bins, "quantile"),
            "brier": brier(y, p),
            "nll": nll(y, z),
            "mean_predicted": float(p.mean()),
            "prevalence": float(np.mean(y))}
