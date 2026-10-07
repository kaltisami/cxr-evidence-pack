"""Loading Google expert labels and model predictions; subgroup columns from NIH metadata."""
import re

import numpy as np
import pandas as pd

LABEL_VALUES = {"YES": 1.0, "NO": 0.0}


def encode_labels(col, hedge="exclude"):
    """YES -> 1, NO -> 0, HEDGE -> NaN (hedge="exclude") or 1 (hedge="positive"). Anything else is an error."""
    v = col.astype(str).str.strip().str.upper()
    hedge_value = {"exclude": np.nan, "positive": 1.0}[hedge]
    unknown = sorted(set(v) - set(LABEL_VALUES) - {"HEDGE"})
    if unknown:
        raise ValueError(f"{col.name}: unexpected label values {unknown}")
    return v.map({**LABEL_VALUES, "HEDGE": hedge_value})


def load_labels(path, findings, hedge="exclude"):
    """Read a Google label CSV; encode each finding column in `findings`; keep every other column as metadata."""
    df = pd.read_csv(path)
    missing = [f for f in findings if f not in df.columns]
    if missing:
        raise KeyError(f"{path}: columns {missing} not found; headers are {list(df.columns)}")
    for f in findings:
        df[f] = encode_labels(df[f], hedge)
    return df


def load_predictions(path):
    """Predictions CSV from 01_inference.py: `Image Index` + logit_<output> + prob_<output>."""
    df = pd.read_csv(path)
    if df["Image Index"].duplicated().any():
        raise ValueError(f"{path}: duplicate Image Index rows")
    return df


def merge(labels, preds, meta=None):
    """Inner-join labels and predictions on Image Index; add NIH metadata columns the label file lacks.

    Returns the merged frame and the list of labelled images with no prediction (to report as exclusions).
    """
    if meta is not None:
        extra = [c for c in meta.columns if c not in labels.columns]
        labels = labels.merge(meta[["Image Index"] + extra], on="Image Index", how="left")
    no_pred = sorted(set(labels["Image Index"]) - set(preds["Image Index"]))
    return labels.merge(preds, on="Image Index", how="inner"), no_pred


MAX_AGE = 110  # NIH metadata has a few impossible ages (e.g. 155): treated as missing


def _age(v):
    m = re.search(r"\d+", str(v))  # NIH ages appear as 58 or, in older releases, "058Y"
    age = float(m.group()) if m else np.nan
    return age if age <= MAX_AGE else np.nan


AGE_BINS = [0, 40, 60, 80, 200]
AGE_NAMES = ["<40", "40-59", "60-79", "80+"]


def add_subgroups(df):
    """Subgroup columns from NIH metadata, when present: sex, age band, view position, follow-up band."""
    df = df.copy()
    if "Patient Gender" in df:
        df["sg_sex"] = df["Patient Gender"].astype(str).str.upper().str[0].map({"M": "M", "F": "F"})
    if "Patient Age" in df:
        df["sg_age"] = pd.cut(df["Patient Age"].map(_age), AGE_BINS, right=False, labels=AGE_NAMES).astype(object)
    if "View Position" in df:
        df["sg_view"] = df["View Position"].astype(str).str.upper()
    if "Follow-up #" in df:
        df["sg_followup"] = pd.cut(df["Follow-up #"], [-1, 0, 4, 10_000], labels=["0", "1-4", "5+"]).astype(object)
    return df


def patient_id(df):
    """Bootstrap cluster id: `Patient ID` if present, else the NIH file-name prefix (00000013_005.png -> 13)."""
    if "Patient ID" in df:
        return df["Patient ID"].to_numpy()
    return df["Image Index"].str.split("_").str[0].astype(int).to_numpy()
