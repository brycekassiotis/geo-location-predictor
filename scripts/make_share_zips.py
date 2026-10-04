"""Package data/processed for teammates.

    python scripts/make_share_zips.py [--mini-only]

-> share/osv_processed_mini.zip  (CSVs subset + ~2.6k images, for CPU debugging)
-> share/osv_processed_full.zip  (all CSVs + all images)
Unzip either at the repo root; it restores data/processed/.
"""
import argparse
import os
import zipfile

import pandas as pd
from tqdm import tqdm

SRC, DST = "data/processed", "share"
MINI = {"train": 2000, "val": 300, "test": 300}


def add_split(zf, name, df):
    zf.writestr(f"{SRC}/{name}.csv", df.to_csv(index=False))
    for i in tqdm(df["id"], desc=name, leave=False):
        p = os.path.join(SRC, "images", f"{i}.jpg")
        zf.write(p, p.replace(os.sep, "/"))  # jpgs are already compressed -> stored


def build(path, frames):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, df in frames.items():
            add_split(zf, name, df)
    print(f"{path}: {os.path.getsize(path) / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(DST, exist_ok=True)
    frames = {s: pd.read_csv(os.path.join(SRC, f"{s}.csv"), dtype={"id": str, "sequence": str})
              for s in MINI}
    build(os.path.join(DST, "osv_processed_mini.zip"),
          {s: df.sample(min(MINI[s], len(df)), random_state=0) for s, df in frames.items()})
    if not args.mini_only:
        build(os.path.join(DST, "osv_processed_full.zip"), frames)


if __name__ == "__main__":
    main()
