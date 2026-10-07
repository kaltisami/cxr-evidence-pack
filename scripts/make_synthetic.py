"""Synthetic stand-ins for the Google label CSVs and the M1 predictions, to exercise 02_evaluate.py end to end.

    python scripts/make_synthetic.py --out synthetic
    python scripts/02_evaluate.py --preds synthetic/predictions.csv --labels synthetic/labels --out synthetic/results

Known truths built in, so the output can be checked:
- labels are drawn from sigmoid(logit / 2): the "model" is over-confident, so the fitted temperature is > 1
  (about 2 on PA images; the AP noise below pushes the pooled estimate higher);
- AP images get extra logit noise: AUROC should be lower for view=AP than for PA;
- HEDGE on ~3% of nodule/mass and pneumothorax labels; 3 labelled images have no prediction (exclusions).
Column names follow configs/label_map.yaml and NIH metadata; they are placeholders until the real headers are seen.
Nothing here is real patient data.
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

HERE = os.path.dirname(__file__)
TRUE_T = 2.0
OUTPUTS = ["Atelectasis", "Consolidation", "Pneumothorax", "Edema", "Effusion", "Pneumonia", "Cardiomegaly",
           "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"]
NIH_14 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia",
          "Infiltration", "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]


def images(n, first_patient, rng):
    """NIH-style metadata: patients with 1-4 images each, file names <patient>_<follow-up>.png."""
    rows, pid = [], first_patient
    while len(rows) < n:
        age, sex = int(rng.integers(18, 95)), rng.choice(["M", "F"])
        for f in range(int(rng.integers(1, 5))):
            rows.append({"Image Index": f"{pid:08d}_{f:03d}.png", "Patient ID": pid, "Follow-up #": f,
                         "Patient Age": age, "Patient Gender": sex,
                         "View Position": rng.choice(["PA", "AP"], p=[0.6, 0.4])})
        pid += 1
    return pd.DataFrame(rows[:n])


def logits_for(meta, rng):
    """Per-output logits: base rate + spread; AP images get extra noise unrelated to the label."""
    out = {}
    ap = (meta["View Position"] == "AP").to_numpy()
    for o in OUTPUTS:
        out[o] = rng.normal(-5.0, 3.0, len(meta))
    return out, ap


def labels_from(z, ap, rng):
    """Draw YES/NO from the over-confident model's logit; AP labels are partly decoupled from it."""
    signal = np.where(ap, z + rng.normal(0, 4.0, len(z)), z)
    return np.where(rng.random(len(z)) < expit(signal / TRUE_T), "YES", "NO")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default="synthetic")
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "label_map.yaml")))
    lab = os.path.join(a.out, "labels")

    test, val = images(1962, 1, rng), images(2412, 20_000, rng)
    pred_rows = []
    for meta in (test, val):
        z, ap = logits_for(meta, rng)
        for col, out in cfg["four_findings"].items():
            meta[col] = labels_from(z[out], ap, rng)
            if col in ("Nodule or mass", "Pneumothorax"):
                meta.loc[rng.random(len(meta)) < 0.03, col] = "HEDGE"
        pred_rows.append(pd.DataFrame({"Image Index": meta["Image Index"],
                                       **{f"logit_{o}": z[o] for o in OUTPUTS},
                                       **{f"prob_{o}": expit(z[o]) for o in OUTPUTS}}))
        meta["_z"], meta["_ap"] = [dict(zip(OUTPUTS, v)) for v in zip(*z.values())], ap

    # all-findings set: 810 PA images from the four-findings test set, 14 findings + Other + Abnormal
    pa = test[test["View Position"] == "PA"].sample(810, random_state=a.seed).reset_index(drop=True)
    allf = pa[["Image Index", "Patient ID", "Follow-up #", "Patient Age", "Patient Gender", "View Position"]].copy()
    for f in NIH_14:
        out = cfg["all_findings"].get(f)
        z = np.array([d[out] for d in pa["_z"]]) if out else rng.normal(-4, 2, len(pa))
        allf[f] = labels_from(z, np.zeros(len(pa), bool), rng)
    allf["Other"] = np.where(rng.random(len(pa)) < 0.05, "YES", "NO")
    any_yes = (allf[NIH_14 + ["Other"]] == "YES").any(axis=1)
    allf[cfg["abnormal_column"]] = np.where(any_yes, "YES", "NO")

    for key, df in [("four_findings_test", test), ("four_findings_validation", val), ("all_findings_test", allf)]:
        path = os.path.join(lab, cfg["files"][key])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.drop(columns=["_z", "_ap"], errors="ignore").to_csv(path, index=False)
    preds = pd.concat(pred_rows, ignore_index=True)
    preds = preds[~preds["Image Index"].isin(test["Image Index"].iloc[[5, 6, 7]])]  # simulate unreadable files
    preds.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    print(f"wrote {a.out}/predictions.csv and {lab}/ (true temperature {TRUE_T})")


if __name__ == "__main__":
    main()
