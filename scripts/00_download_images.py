"""M1 prep: download only the NIH images that have an expert label (4.376 of 112.120, ~2 GB) from Kaggle.

    pip install kagglehub pandas pyyaml
    python scripts/00_download_images.py                # all labelled images -> data/nih/
    python scripts/00_download_images.py --limit 20     # smoke test
    caffeinate -i python scripts/00_download_images.py --keep-trying   # overnight: waits out Kaggle's throttling
    python scripts/01_inference.py --nih data/nih --labels data/google2019_nih-chest-xray-labels.csv.gz

Kaggle refuses further requests after a few hundred files, with or without an account (7 Oct 2026: ~400 files,
then a few more minutes later). --keep-trying waits and resumes. kagglehub uses Kaggle credentials when present
(`kaggle auth login`). Files land in
the Kaggle layout, data/nih/images_XXX/images/<Image Index>, which is what 01_inference.py reads. Already
downloaded files are skipped, so the script can be re-run after an interruption. A manifest (name, folder, bytes,
SHA-256) is written for the data dossier.
"""
import argparse
import bisect
import hashlib
import logging
import os
import time
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


def download_round(names, out, workers):
    """Download `names` in parallel. Returns ({name: folder}, {name: error}, throttled).

    Stops early (throttled=True) after MAX_FAILURE_STREAK failures in a row: the folder map is verified, so such a
    run means Kaggle is refusing requests, not that the files are missing.
    """
    got = {n: folder_of(n) for n in names if os.path.exists(os.path.join(out, rel_path(n, folder_of(n))))}
    failed, streak = {}, 0  # only real requests count towards the streak, not files already on disk
    with ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(fetch, n, out): n for n in names if n not in got}
        for job in as_completed(jobs):
            try:
                name, folder = job.result()
                got[name] = folder
                streak = 0
            except Exception as e:
                failed[jobs[job]] = str(e)[:200]
                streak += 1
            print(f"{len(got)}/{len(names)}  failed: {len(failed)}", end="\r", flush=True)
            if streak >= MAX_FAILURE_STREAK:
                pool.shutdown(wait=False, cancel_futures=True)
                return got, failed, True
    return got, failed, bool(failed) and streak == len(failed)  # short tail: every failure came in one unbroken run


def stamp():
    return time.strftime("%H:%M")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="label file (default: label_file in configs/label_map.yaml)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "nih"))
    ap.add_argument("--limit", type=int, help="download only the first N images (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-trying", action="store_true",
                    help="when Kaggle refuses requests, wait and resume instead of stopping (for an overnight run)")
    ap.add_argument("--wait-minutes", type=int, default=10, help="first wait; doubles up to 60 while refused")
    ap.add_argument("--max-hours", type=float, default=14)
    a = ap.parse_args()
    logging.getLogger("kagglehub").setLevel(logging.WARNING)

    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "label_map.yaml")))
    labels = a.labels or os.path.join(HERE, "..", cfg["label_file"])
    names = sorted(pd.read_csv(labels)["Image Index"].unique())[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    done, errors, wait = {}, {}, a.wait_minutes
    deadline = time.time() + a.max_hours * 3600
    pending = names
    while pending:
        before = len(done)
        got, failed, throttled = download_round(pending, a.out, a.workers)
        done.update(got)

        errors = failed
        pending = [n for n in pending if n not in done]
        print(f"\n{stamp()} {len(done)}/{len(names)} on disk, {len(pending)} left", flush=True)
        if not throttled:
            break  # what is left failed for other reasons: see failed.csv
        if not a.keep_trying or time.time() + wait * 60 > deadline:
            print("Kaggle is refusing requests (it throttles after a few hundred files). Re-run later"
                  + (" (time limit reached)." if a.keep_trying else ", or use --keep-trying to wait and resume."))
            break
        print(f"{stamp()} Kaggle is refusing requests: waiting {wait} min", flush=True)
        time.sleep(wait * 60)
        wait = a.wait_minutes if len(done) > before else min(wait * 2, 60)  # back off only after a round with nothing new

    rows = []
    for name, folder in sorted(done.items()):
        path = os.path.join(a.out, rel_path(name, folder))
        rows.append({"Image Index": name, "folder": f"images_{folder:03d}", "bytes": os.path.getsize(path),
                     "sha256": sha256(path)})
    manifest = pd.DataFrame(rows, columns=["Image Index", "folder", "bytes", "sha256"])
    manifest.to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    failed_csv = os.path.join(a.out, "failed.csv")
    left = [{"Image Index": n, "error": errors.get(n, "not attempted (stopped)")} for n in pending]
    if left:
        pd.DataFrame(left).to_csv(failed_csv, index=False)
    elif os.path.exists(failed_csv):
        os.remove(failed_csv)
    print(f"{stamp()} {len(done)} images ({manifest['bytes'].sum() / 1e9:.2f} GB) in {a.out}; {len(left)} missing"
          + (" -> failed.csv (re-run to retry)" if left else ""))


if __name__ == "__main__":
    main()
