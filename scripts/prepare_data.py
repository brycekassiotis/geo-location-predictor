"""Preprocessing (owner: Owen).

1. Keep top-50 countries (by full train.csv count)
2. Stream images straight out of data/raw/images/{train,test}/*.zip (no extraction - saves ~16 GB)
   and resize once (short side 224) -> data/processed/images/<id>.jpg
3. 90/10 train/val split grouped by `sequence` (no near-duplicate frames across splits)
4. Write data/processed/{train,val,test}.csv

Re-runnable: already-resized images are skipped.

    python scripts/prepare_data.py
"""
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

RAW, OUT = "data/raw", "data/processed"
IMG_OUT = os.path.join(OUT, "images")
TOP_K, VAL_FRAC, SIZE, QUALITY = 50, 0.1, 224, 90
COLS = ["id", "latitude", "longitude", "country", "sequence", "region", "sub-region", "city",
        "unique_country", "unique_region", "unique_sub-region", "unique_city",
        "quadtree_10_1000", "land_cover", "climate", "drive_side", "dist_sea"]


def load_csv(name):
    return pd.read_csv(os.path.join(RAW, name), usecols=COLS, dtype={"id": str, "sequence": str})


def resize_bytes(data, dst):
    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    s = SIZE / min(w, h)
    im.resize((round(w * s), round(h * s)), Image.BICUBIC).save(dst, quality=QUALITY)


def extract_resized(split, keep_ids):
    """Resize every image in the split's zips whose id is in keep_ids. Returns ids written."""
    os.makedirs(IMG_OUT, exist_ok=True)
    done = set()
    with ThreadPoolExecutor(os.cpu_count() or 8) as ex:
        for z in sorted(glob(os.path.join(RAW, "images", split, "*.zip"))):
            futures = []
            with zipfile.ZipFile(z) as f:
                for name in tqdm(f.namelist(), desc=f"{split}/{os.path.basename(z)}"):
                    if not name.endswith(".jpg"):
                        continue
                    img_id = os.path.splitext(os.path.basename(name))[0]
                    if img_id not in keep_ids:
                        continue
                    dst = os.path.join(IMG_OUT, f"{img_id}.jpg")
                    if not os.path.exists(dst):
                        futures.append(ex.submit(resize_bytes, f.read(name), dst))
                        if len(futures) >= 512:  # bound memory held in raw image bytes
                            for fu in futures:
                                fu.result()
                            futures = []
                    done.add(img_id)
            for fu in futures:
                fu.result()
    return done


def main():
    train = load_csv("train.csv")
    test = load_csv("test.csv")
    top = train["country"].value_counts().head(TOP_K).index
    train, test = train[train["country"].isin(top)], test[test["country"].isin(top)]

    train = train[train["id"].isin(extract_resized("train", set(train["id"])))]
    test = test[test["id"].isin(extract_resized("test", set(test["id"])))]
    # TODO(Owen): per-cell/per-country balancing, subset sizes for learning curves

    train["sequence"] = train["sequence"].fillna(train["id"])
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=42)
    tr_idx, va_idx = next(gss.split(train, groups=train["sequence"]))
    train.iloc[tr_idx].to_csv(os.path.join(OUT, "train.csv"), index=False)
    train.iloc[va_idx].to_csv(os.path.join(OUT, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT, "test.csv"), index=False)
    print(f"train={len(tr_idx)} val={len(va_idx)} test={len(test)} countries={len(top)}")


if __name__ == "__main__":
    main()
