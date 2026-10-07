"""M1 prep: download only the NIH images that have an expert label (4.376 of 112.120, ~2 GB) from Kaggle.

    pip install kagglehub pandas pyyaml
    python scripts/00_download_images.py                # all labelled images -> data/nih/
    python scripts/00_download_images.py --limit 20     # smoke test
    python scripts/01_inference.py --nih data/nih --labels data/google2019_nih-chest-xray-labels.csv.gz

Anonymous single-file downloads work for a few hundred files, then Kaggle refuses further requests (7 Oct 2026:
~400 files). kagglehub uses Kaggle credentials when present (`kaggle auth login`). Files land in
the Kaggle layout, data/nih/images_XXX/images/<Image Index>, which is what 01_inference.py reads. Already
downloaded files are skipped, so the script can be re-run after an interruption. A manifest (name, folder, bytes,
SHA-256) is written for the data dossier.
"""
import argparse
import bisect
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("TQDM_DISABLE", "1")  # one progress line from this script, not one bar per file
import kagglehub  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(__file__)
DATASET = "nih-chest-xrays/data/versions/3"  # pinned: the version Kaggle served on 7 Oct 2026
# First file of each images_XXX folder in the Kaggle dataset (names sorted; verified against Kaggle on 7 Oct 2026).
FOLDER_FIRST = ["00000001_000.png", "00001336_000.png", "00003923_014.png", "00006585_007.png",
                "00009232_004.png", "00011558_008.png", "00013774_026.png", "00016051_010.png",
                "00018387_035.png", "00020945_050.png", "00024718_000.png", "00028173_003.png"]


MAX_FAILURE_STREAK = 25  # the folder map is verified, so a run of misses means Kaggle is refusing requests


def folder_of(name):
    return bisect.bisect_right(FOLDER_FIRST, name)  # 1-based folder number


def rel_path(name, folder):
    return f"images_{folder:03d}/images/{name}"


def fetch(name, out, retries=3):
    """Download one image; on a miss, try the neighbouring folders. Returns (name, folder) or raises."""
    k = folder_of(name)
    candidates = [k] + [f for f in (k - 1, k + 1) if 1 <= f <= len(FOLDER_FIRST)]
    for folder in candidates:
        dest = os.path.join(out, rel_path(name, folder))
        if os.path.exists(dest):
            return name, folder
        for attempt in range(retries):
            try:
                kagglehub.dataset_download(DATASET, path=rel_path(name, folder), output_dir=out)
                return name, folder
            except Exception as e:  # kagglehub raises different types for 404 and network errors
                if "404" in str(e) or "not found" in str(e).lower():
                    break
                if attempt == retries - 1:
                    raise
    raise FileNotFoundError(f"{name} not found in folders {candidates}")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="label file (default: label_file in configs/label_map.yaml)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "nih"))
    ap.add_argument("--limit", type=int, help="download only the first N images (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    logging.getLogger("kagglehub").setLevel(logging.WARNING)

    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "label_map.yaml")))
    labels = a.labels or os.path.join(HERE, "..", cfg["label_file"])
    names = sorted(pd.read_csv(labels)["Image Index"].unique())[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    done, failed, streak = [], [], 0
    with ThreadPoolExecutor(a.workers) as pool:
        jobs = {pool.submit(fetch, n, a.out): n for n in names}
        for i, job in enumerate(as_completed(jobs), 1):
            try:
                done.append(job.result())
                streak = 0
            except Exception as e:
                failed.append({"Image Index": jobs[job], "error": str(e)[:200]})
                streak += 1
            print(f"{i}/{len(names)}  failed: {len(failed)}", end="\r", flush=True)
            if streak >= MAX_FAILURE_STREAK:
                pool.shutdown(wait=False, cancel_futures=True)
                print(f"\n{streak} failures in a row: Kaggle is refusing requests (anonymous downloads are throttled "
                      "after a few hundred files). Log in (`kaggle auth login`) or wait, then re-run.")
                break

    rows = []
    for name, folder in sorted(done):
        path = os.path.join(a.out, rel_path(name, folder))
        rows.append({"Image Index": name, "folder": f"images_{folder:03d}", "bytes": os.path.getsize(path),
                     "sha256": sha256(path)})
    manifest = pd.DataFrame(rows, columns=["Image Index", "folder", "bytes", "sha256"])
    manifest.to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    if failed:
        pd.DataFrame(failed).to_csv(os.path.join(a.out, "failed.csv"), index=False)
    print(f"\n{len(done)} images ({manifest['bytes'].sum() / 1e9:.2f} GB) in {a.out}; {len(failed)} failed"
          + (" -> failed.csv (re-run to retry)" if failed else ""))


if __name__ == "__main__":
    main()
