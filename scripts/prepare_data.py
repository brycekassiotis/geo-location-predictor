"""Preprocessing (owner: Owen).

1. Unzip data/raw/images/{train,test}/*.zip
2. Keep top-50 countries (by train count)
3. Resize once (short side 224) -> data/processed/images/<id>.jpg
4. 90/10 train/val split grouped by `sequence` (no near-duplicate frames across splits)
5. Write data/processed/{train,val,test}.csv

    python scripts/prepare_data.py
"""
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

RAW, OUT = "data/raw", "data/processed"
TOP_K, VAL_FRAC, SIZE = 50, 0.1, 224


def unzip_all(split):
    dst = os.path.join(RAW, "images", split)
    for z in sorted(glob(os.path.join(dst, "*.zip"))):
        marker = z + ".extracted"
        if os.path.exists(marker):
            continue
        print("extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(dst)
        open(marker, "w").close()
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob(os.path.join(dst, "**", "*.jpg"), recursive=True)}


def resize(src, dst):
    if os.path.exists(dst):
        return
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = SIZE / min(w, h)
    im.resize((round(w * s), round(h * s)), Image.BICUBIC).save(dst, quality=90)


def process(df, paths):
    df = df[df["id"].astype(str).isin(paths)].copy()
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    jobs = [(paths[str(i)], os.path.join(OUT, "images", f"{i}.jpg")) for i in df["id"]]
    with ThreadPoolExecutor(16) as ex:
        list(tqdm(ex.map(lambda a: resize(*a), jobs), total=len(jobs), desc="resize"))
    return df


def main():
    train = pd.read_csv(os.path.join(RAW, "train.csv"))
    test = pd.read_csv(os.path.join(RAW, "test.csv"))
    top = train["country"].value_counts().head(TOP_K).index
    train, test = train[train["country"].isin(top)], test[test["country"].isin(top)]

    train = process(train, unzip_all("train"))
    test = process(test, unzip_all("test"))
    # TODO(Owen): per-cell/per-country balancing, subset sizes for learning curves

    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=42)
    tr_idx, va_idx = next(gss.split(train, groups=train["sequence"]))
    train.iloc[tr_idx].to_csv(os.path.join(OUT, "train.csv"), index=False)
    train.iloc[va_idx].to_csv(os.path.join(OUT, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT, "test.csv"), index=False)
    print(f"train={len(tr_idx)} val={len(va_idx)} test={len(test)}")


if __name__ == "__main__":
    main()
