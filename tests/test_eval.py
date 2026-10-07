"""Known-answer tests for cxr_eval, plus an end-to-end run of 02_evaluate.py on synthetic data."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from cxr_eval import calibration as cal  # noqa: E402
from cxr_eval import data, metrics as m  # noqa: E402


def binormal(n, d, seed=0, prevalence=0.3):
    """Scores N(0,1) for negatives, N(d,1) for positives: true AUROC = Phi(d / sqrt 2)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    return y, rng.normal(d * y, 1.0)


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.integers(0, 5, 500).astype(float)  # heavy ties
    assert m.auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_auroc_edge_cases():
    assert m.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert m.auroc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert np.isnan(m.auroc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(m.auprc([0, 0], [0.1, 0.2]))


def test_bootstrap_ci_covers_true_auroc():
    true = stats.norm.cdf(1.0 / np.sqrt(2))
    covered = 0
    for seed in range(40):
        y, s = binormal(400, 1.0, seed)
        r = m.bootstrap_ci(m.auroc, y, s, n_boot=300, seed=seed)
        covered += r["ci_low"] <= true <= r["ci_high"]
    assert covered >= 34  # nominal 95% of 40 = 38; allow Monte Carlo slack


def test_cluster_bootstrap_is_wider_when_images_repeat_patients():
    y, s = binormal(150, 1.0, 3)
    y, s = np.repeat(y, 4), np.repeat(s, 4) + np.random.default_rng(3).normal(0, 0.1, 600)
    patients = np.repeat(np.arange(150), 4)
    iid = m.bootstrap_ci(m.auroc, y, s, n_boot=500)
    clu = m.bootstrap_ci(m.auroc, y, s, clusters=patients, n_boot=500)
    assert (clu["ci_high"] - clu["ci_low"]) > 1.5 * (iid["ci_high"] - iid["ci_low"])


def test_wilson_ci_known_value():
    lo, hi = m.wilson_ci(8, 10)  # textbook: 0.490 - 0.943
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)


def test_threshold_for_sensitivity_and_confusion():
    y, s = binormal(1000, 1.5, 4)
    t = m.threshold_for_sensitivity(y, s, 0.9)
    c = m.confusion_at(y, s, t)
    assert c["sensitivity"] >= 0.9
    assert m.confusion_at(y, s, np.nextafter(t, np.inf))["sensitivity"] < 0.9  # highest such threshold
    assert c["tp"] + c["fn"] + c["fp"] + c["tn"] == 1000


def test_ece_near_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(5)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p).astype(int)
    assert cal.ece(y, p) < 0.015
    assert cal.ece(y, p, strategy="quantile") < 0.015
    assert cal.ece(y, np.clip(p * 1.5, 0, 1)) > 0.1


def test_temperature_scaling_recovers_true_temperature():
    rng = np.random.default_rng(6)
    z = rng.normal(-1, 3, 20_000)
    y = (rng.random(20_000) < expit(z / 2.0)).astype(int)  # model logits over-confident by T = 2
    t = cal.fit_temperature(y, z)
    assert t == pytest.approx(2.0, rel=0.08)
    before, after = cal.calibration_summary(y, z), cal.calibration_summary(y, z, t)
    assert after["ece"] < before["ece"] and after["nll"] < before["nll"]
    assert m.auroc(y, z / t) == pytest.approx(m.auroc(y, z))  # monotone: discrimination unchanged


def test_label_encoding_and_hedge_policy():
    col = pd.Series(["YES", "no", "HEDGE", " Yes "], name="Pneumothorax")
    assert data.encode_labels(col, "exclude").tolist()[:2] == [1.0, 0.0]
    assert np.isnan(data.encode_labels(col, "exclude")[2])
    assert data.encode_labels(col, "positive").tolist() == [1.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        data.encode_labels(pd.Series(["YES", "MAYBE"], name="x"))


def test_subgroups_parse_nih_metadata():
    df = pd.DataFrame({"Image Index": ["00000013_005.png"] * 4, "Patient Age": ["058Y", 39, 85, "n/a"],
                       "Patient Gender": ["M", "F", "m", "?"], "View Position": ["PA", "AP", "pa", "AP"],
                       "Follow-up #": [0, 3, 5, 12]})
    g = data.add_subgroups(df)
    assert g["sg_age"].tolist()[:3] == ["40-59", "<40", "80+"] and pd.isna(g["sg_age"][3])
    assert g["sg_sex"].tolist()[:3] == ["M", "F", "M"] and pd.isna(g["sg_sex"][3])
    assert g["sg_view"].tolist() == ["PA", "AP", "PA", "AP"]
    assert g["sg_followup"].tolist() == ["0", "1-4", "5+", "5+"]
    assert data.patient_id(df)[0] == 13


def test_end_to_end_on_synthetic(tmp_path):
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True, text=True)  # noqa: E731
    run(os.path.join(ROOT, "scripts", "make_synthetic.py"), "--out", str(tmp_path))
    out = tmp_path / "results"
    run(os.path.join(ROOT, "scripts", "02_evaluate.py"), "--preds", str(tmp_path / "predictions.csv"),
        "--labels", str(tmp_path / "labels"), "--out", str(out), "--n-boot", "100")

    disc = pd.read_csv(out / "discrimination.csv")
    assert len(disc) == 2 * (4 + 7 + 1)  # AUROC + AUPRC for 4 + 7 findings + abnormal
    assert (disc["ci_low"] <= disc["estimate"]).all() and (disc["estimate"] <= disc["ci_high"]).all()

    calib = pd.read_csv(out / "calibration.csv")
    scaled = calib[calib.version == "temperature_scaled"]
    assert len(scaled) == 4 and (scaled["temperature"] > 1.5).all()
    raw = calib[(calib.version == "raw") & (calib.task == "four_findings")].set_index("finding")
    raw = raw.loc[scaled["finding"], "ece"].to_numpy()
    assert (scaled["ece"].to_numpy() < raw).all()

    view = pd.read_csv(out / "subgroups.csv").query("task == 'four_findings' and subgroup == 'view'")
    by = view.pivot(index="finding", columns="level", values="estimate")
    assert (by["PA"] > by["AP"]).all()  # built-in AP degradation is detected


    info = json.load(open(out / "run.json"))
    assert len(info["tasks"]["four_findings"]["excluded_no_prediction"]) == 3
    assert info["tasks"]["four_findings"]["hedge_rows_excluded"]["Pneumothorax"] > 0
    assert info["tasks"]["four_findings"]["patients_in_both_validation_and_test"] == 0
    assert len(info["sha256"]) == 4
