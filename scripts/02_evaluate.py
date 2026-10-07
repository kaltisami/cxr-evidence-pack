"""M2: discrimination, operating points, subgroups and calibration from the M1 predictions.

    python scripts/02_evaluate.py --preds predictions.csv --out results/
    # --labels defaults to label_file in configs/label_map.yaml: Google's four-findings labels as bundled in
    #   TorchXRayVision (google2019_nih-chest-xray-labels.csv.gz), test and validation rows told apart by `Set Id`
    # optional: --all-findings <test_labels.csv>  Google's 810-image all-findings set, if obtained
    # optional: --nih-meta Data_Entry_2017.csv     adds sex/age/view/follow-up if the label file lacks them

Writes to --out:
    discrimination.csv    AUROC, AUPRC with patient-level bootstrap 95% CIs, per finding
    operating_points.csv  confusion tables + sens/spec/PPV/NPV (Wilson CIs) at the as-shipped threshold and at a
                          validation-fitted 90%-sensitivity threshold
    subgroups.csv         AUROC (CI), n, positives per subgroup level
    calibration.csv       ECE, Brier, NLL before and after temperature scaling (T fitted on validation)
    reliability.csv       reliability-diagram bins, raw and scaled
    run.json              settings, counts, excluded images
Try it without the labels: python scripts/make_synthetic.py --out synthetic/ (see its docstring).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cxr_eval import calibration as cal  # noqa: E402
from cxr_eval import data, metrics as m  # noqa: E402

SUBGROUPS = ["sg_sex", "sg_age", "sg_view", "sg_followup"]
MIN_PER_CLASS = 5  # subgroup AUROC is reported only with at least this many positives and negatives


def discrimination(task, finding, y, s, clusters, a):
    rows = []
    for name, fn in [("auroc", m.auroc), ("auprc", m.auprc)]:
        r = m.bootstrap_ci(fn, y, s, clusters, a.n_boot, seed=a.seed)
        rows.append({"task": task, "finding": finding, "metric": name, "n": len(y), "n_pos": int(y.sum()),
                     "prevalence": y.mean(), **r})
    return rows


def subgroups(task, finding, df, y, s, clusters, a):
    rows = []
    for g in [c for c in SUBGROUPS if c in df]:
        levels = df[g].to_numpy()
        for level in sorted(pd.unique(levels[pd.notna(levels)]), key=str):
            k = levels == level
            n_pos, n_neg = int(y[k].sum()), int((1 - y[k]).sum())
            r = {"task": task, "finding": finding, "subgroup": g[3:], "level": level, "n": int(k.sum()),
                 "n_pos": n_pos}
            if min(n_pos, n_neg) >= MIN_PER_CLASS:
                r.update(m.bootstrap_ci(m.auroc, y[k], s[k], clusters[k], a.n_boot, seed=a.seed))
            else:
                r["note"] = f"fewer than {MIN_PER_CLASS} positives or negatives"
            rows.append(r)
    return rows


def labelled(df, finding, output):
    """Rows with a reference label for `finding` (HEDGE excluded upstream); returns y, logit, clusters, frame."""
    d = df[df[finding].notna()]
    return d[finding].to_numpy(int), d[f"logit_{output}"].to_numpy(float), data.patient_id(d), d


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", help="four-findings label file (default: label_file in the config)")
    ap.add_argument("--all-findings", help="all-findings test_labels.csv (optional second task)")
    ap.add_argument("--nih-meta")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "label_map.yaml"))
    ap.add_argument("--thresholds",
                    default=os.path.join(os.path.dirname(__file__), "..", "configs", "shipped_thresholds.yaml"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--hedge", choices=["exclude", "positive"], help="overrides hedge_policy in the config")
    ap.add_argument("--target-sensitivity", type=float, default=0.90)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    shipped = yaml.safe_load(open(a.thresholds))
    hedge = a.hedge or cfg["hedge_policy"]
    os.makedirs(a.out, exist_ok=True)
    preds = data.load_predictions(a.preds)
    meta = pd.read_csv(a.nih_meta) if a.nih_meta else None
    a.labels = a.labels or os.path.join(os.path.dirname(__file__), "..", cfg["label_file"])

    disc, ops, subs, calib, rel = [], [], [], [], []
    files = {"four_findings_labels": a.labels, "all_findings_labels": a.all_findings, "predictions": a.preds}
    run = {"settings": vars(a), "hedge": hedge, "tasks": {},
           "sha256": {k: sha256(p) for k, p in files.items() if p}}

    def evaluate(task, test, val, findings):
        for finding, output in findings.items():
            y, s, cl, d = labelled(test, finding, output)
            disc.extend(discrimination(task, finding, y, s, cl, a))
            subs.extend(subgroups(task, finding, d, y, s, cl, a))

            thresholds = {"as_shipped": logit(shipped[output])}
            temperature = 1.0
            if val is not None:
                yv, sv, _, _ = labelled(val, finding, output)
                thresholds[f"val_sens{a.target_sensitivity:.2f}"] = m.threshold_for_sensitivity(
                    yv, sv, a.target_sensitivity)
                temperature = cal.fit_temperature(yv, sv)
            for name, t in thresholds.items():
                ops.append({"task": task, "finding": finding, "operating_point": name,
                            "threshold_prob": expit(t), **m.confusion_at(y, s, t)})
            for name, temp in [("raw", 1.0)] + ([("temperature_scaled", temperature)] if val is not None else []):
                calib.append({"task": task, "finding": finding, "version": name,
                              **cal.calibration_summary(y, s, temp)})
                r = cal.reliability(y, expit(s / temp))
                rel.extend({"task": task, "finding": finding, "version": name, "mean_predicted": p,
                            "observed": o, "count": c}
                           for p, o, c in zip(r["mean_predicted"], r["observed"], r["count"]))

    # Task 1: four findings, 1.962 test images; validation rows used for thresholds and temperature.
    labels = data.load_labels(a.labels, cfg["four_findings"], hedge)
    split = labels[cfg["split_column"]]
    unknown = sorted(set(split) - set(cfg["splits"].values()))
    if unknown:
        raise ValueError(f"{a.labels}: unexpected {cfg['split_column']} values {unknown}")
    test, no_pred_test = data.merge(labels[split == cfg["splits"]["test"]], preds, meta)
    val, no_pred_val = data.merge(labels[split == cfg["splits"]["validation"]], preds, meta)
    shared = set(data.patient_id(test)) & set(data.patient_id(val))
    val = val[~np.isin(data.patient_id(val), list(shared))]  # tuning data must not share patients with test data
    test, val = data.add_subgroups(test), data.add_subgroups(val)
    evaluate("four_findings", test, val, cfg["four_findings"])
    run["tasks"]["four_findings"] = {
        "test_images": len(test), "validation_images": len(val),
        "excluded_no_prediction": no_pred_test + no_pred_val,
        "patients_in_both_validation_and_test": len(shared),  # dropped from validation
        "hedge_rows_excluded": {f: int(test[f].isna().sum()) for f in cfg["four_findings"]}}

    # Task 2 (optional): all findings, 810 PA test images; no validation labels, so no fitted threshold or
    # temperature.
    if a.all_findings:
        findings = list(cfg["all_findings"]) + [cfg["abnormal_column"]]
        allf, no_pred_all = data.merge(data.load_labels(a.all_findings, findings, "exclude"), preds, meta)
        allf = data.add_subgroups(allf)
        evaluate("all_findings", allf, None, cfg["all_findings"])
        logit_cols = [c for c in preds.columns if c.startswith("logit_")]
        allf["logit_abnormal"] = allf[logit_cols].max(axis=1)  # see abnormal_score in the config
        y, s, cl, d = labelled(allf, cfg["abnormal_column"], "abnormal")
        disc.extend(discrimination("all_findings", "abnormal", y, s, cl, a))
        subs.extend(subgroups("all_findings", "abnormal", d, y, s, cl, a))
        run["tasks"]["all_findings"] = {"test_images": len(allf), "excluded_no_prediction": no_pred_all}

    for name, rows in [("discrimination", disc), ("operating_points", ops), ("subgroups", subs),
                       ("calibration", calib), ("reliability", rel)]:
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
    json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=2, default=str)

    show = pd.DataFrame(disc).pivot_table(index=["task", "finding"], columns="metric", values="estimate")
    print(show.round(3).to_string())
    print(f"\nwrote {a.out}/")


if __name__ == "__main__":
    main()
