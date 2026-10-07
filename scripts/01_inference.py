"""M1: run densenet121-res224-chex on every image that has a Google expert label; save logits.

Kaggle: attach the dataset `nih-chest-xrays/data` and upload the Google label CSVs as a private dataset.
    pip install torchxrayvision
    python 01_inference.py --nih /kaggle/input/data --labels /kaggle/input/<labels> --out predictions.csv

Saves raw logits AND sigmoid probabilities. The library's default output is rescaled around its built-in operating
thresholds (op_norm), which is not a probability, so calibration analysis must use the raw logits.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv


def image_index(nih_root):
    paths = glob.glob(os.path.join(nih_root, "images_*", "images", "*.png"))
    return {os.path.basename(p): p for p in paths}


def load(path, transform):
    img = skimage.io.imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(2)
    img = xrv.datasets.normalize(img, 255)[None, ...]
    return torch.from_numpy(transform(img)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih", required=True)
    ap.add_argument("--labels", required=True, help="folder with the Google expert-label CSVs")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    ids = set()
    for csv in glob.glob(os.path.join(a.labels, "**", "*.csv"), recursive=True):
        df = pd.read_csv(csv)
        print(csv, list(df.columns))  # confirm column names for configs/label_map.yaml
        if "Image Index" in df.columns:
            ids |= set(df["Image Index"])
    index = image_index(a.nih)
    missing = sorted(i for i in ids if i not in index)
    print(f"{len(ids)} labelled images, {len(missing)} not found in NIH folders")
    ids = sorted(i for i in ids if i in index)

    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.op_threshs = None  # disable op_norm: keep raw logits
    model.eval()
    names = model.pathologies
    keep = [k for k, n in enumerate(names) if n]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    rows = []
    with torch.no_grad():
        for s in range(0, len(ids), a.batch):
            chunk = ids[s:s + a.batch]
            x = torch.stack([load(index[i], transform) for i in chunk])
            logits = model(x).numpy()
            for i, l in zip(chunk, logits):
                r = {"Image Index": i}
                for k in keep:
                    r[f"logit_{names[k]}"] = float(l[k])
                    r[f"prob_{names[k]}"] = float(1 / (1 + np.exp(-l[k])))
                rows.append(r)
            print(f"{s + len(chunk)}/{len(ids)}", end="\r")

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
