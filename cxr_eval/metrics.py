"""Discrimination metrics, bootstrap confidence intervals and operating points.

All functions take a binary reference `y` (0/1) and a continuous score `s` (logit or probability; higher means
more likely positive). Rows with a missing reference (NaN, e.g. an excluded HEDGE label) must be dropped by the
caller.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score


def auroc(y, s):
    """Mann-Whitney AUROC with ties counted as 1/2. NaN if a class is absent."""
    y = np.asarray(y, bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(s)
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(y, s):
    """Average precision (step-wise area under the precision-recall curve). NaN if no positives."""
    y = np.asarray(y, int)
    if y.sum() == 0:
        return np.nan
    return average_precision_score(y, s)


def _cluster_members(clusters):
    """Row indices of each cluster, as a list of arrays."""
    _, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    return np.split(order, np.cumsum(np.bincount(inverse))[:-1])


def bootstrap_ci(metric, y, s, clusters=None, n_boot=2000, alpha=0.05, seed=0):
    """Point estimate and percentile (1 - alpha) CI of metric(y, s).

    clusters: optional group id per row (NIH `Patient ID`). Several images of one patient are not independent,
    so resampling patients rather than images gives honest, usually wider, intervals.
    Resamples where the metric is undefined (a class absent) are dropped; their count is returned.
    """
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    est = metric(y, s)
    members = None if clusters is None else _cluster_members(np.asarray(clusters))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        if members is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        boots[b] = metric(y[idx], s[idx])
    ok = boots[~np.isnan(boots)]
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if len(ok) else (np.nan, np.nan)
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "n_boot_valid": len(ok)}


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def threshold_for_sensitivity(y, s, target=0.90):
    """Highest threshold whose sensitivity (score >= threshold) is at least `target`.

    Fit this on the validation set and apply it unchanged to the test set; choosing it on the test set
    would make the test sensitivity optimistic by construction.
    """
    y = np.asarray(y, bool)
    pos = np.sort(np.asarray(s)[y])[::-1]
    k = int(np.ceil(target * len(pos)))
    return pos[max(k, 1) - 1]


def confusion_at(y, s, threshold, alpha=0.05):
    """Confusion table, sensitivity, specificity, PPV, NPV (Wilson CIs) at score >= threshold."""
    y = np.asarray(y, bool)
    pred = np.asarray(s) >= threshold
    tp, fn = int((pred & y).sum()), int((~pred & y).sum())
    fp, tn = int((pred & ~y).sum()), int((~pred & ~y).sum())
    out = {"threshold": threshold, "tp": tp, "fn": fn, "fp": fp, "tn": tn}
    for name, k, n in [("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp),
                       ("ppv", tp, tp + fp), ("npv", tn, tn + fn)]:
        out[name] = k / n if n else np.nan
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = wilson_ci(k, n, alpha)
    return out
