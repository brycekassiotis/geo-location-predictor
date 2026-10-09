"""Baseline predictions and metrics (owner: Jack).

Creates the same prediction-file format used by the trained model:
results/<name>/preds_test.csv

Baselines:
  1. random geocell -- uniformly samples one of the model's geocell centroids
  2. most-common-country -- predicts the mean training coordinate of the most
     common training country for every test image

Example:
    python scripts/baselines.py --cell_column cell_q500
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import sys
from pathlib import Path
# Allow `python scripts/<name>.py` from anywhere (puts the repo root on sys.path so `src.geo` imports work).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.geo.geocells import build_cell_index
from src.geo.metrics import haversine_np, summarize


DEFAULT_TRAIN = "data/processed/train.csv"
DEFAULT_TEST = "data/processed/test.csv"
DEFAULT_OUT = "results"


def _load_cells(train, cell_column):
    cell_to_idx, centroids = build_cell_index(train, cell_column)
    cell_to_idx = {str(k): int(v) for k, v in cell_to_idx.items()}
    return cell_to_idx, centroids


def _true_cell_indices(df, cell_to_idx, cell_column):
    raw = df[cell_column].astype(str)
    missing = ~raw.isin(cell_to_idx)
    if missing.any():
        raise ValueError(
            f"{missing.sum()} rows have {cell_column} values not present in the training cells"
        )
    return raw.map(cell_to_idx).to_numpy(dtype=np.int64)


def _prediction_frame(test, pred_coords, pred_cells, confidence=None, true_cells=None):
    true = test[["latitude", "longitude"]].to_numpy(dtype=float)
    pred_coords = np.asarray(pred_coords, dtype=float)
    dist = haversine_np(
        true[:, 0], true[:, 1], pred_coords[:, 0], pred_coords[:, 1]
    )
    if confidence is None:
        confidence = np.ones(len(test), dtype=float)
    if true_cells is None:
        true_cells = np.full(len(test), -1, dtype=np.int64)

    return pd.DataFrame({
        "id": test["id"].astype(str).to_numpy(),
        "true_lat": true[:, 0],
        "true_lon": true[:, 1],
        "pred_lat": pred_coords[:, 0],
        "pred_lon": pred_coords[:, 1],
        "pred_cell": np.asarray(pred_cells, dtype=np.int64),
        "true_cell": np.asarray(true_cells, dtype=np.int64),
        "confidence": np.asarray(confidence, dtype=float),
        "dist_km": dist,
    })


def random_geocell_baseline(train, test, cell_column, seed):
    """Uniformly choose a geocell centroid for every test row."""
    cell_to_idx, centroids = _load_cells(train, cell_column)
    true_cells = _true_cell_indices(test, cell_to_idx, cell_column)
    rng = np.random.default_rng(seed)
    pred_cells = rng.integers(0, len(centroids), size=len(test), dtype=np.int64)
    return _prediction_frame(test, centroids[pred_cells], pred_cells, true_cells=true_cells)


def common_country_baseline(train, test, cell_column):
    """Predict the mean train coordinate of the most common training country."""
    cell_to_idx, centroids = _load_cells(train, cell_column)
    true_cells = _true_cell_indices(test, cell_to_idx, cell_column)

    country_counts = train["country"].fillna("").astype(str).value_counts()
    if country_counts.empty:
        raise ValueError("Training data has no country values")
    country = country_counts.index[0]
    country_rows = train[train["country"].astype(str) == country]
    coord = country_rows[["latitude", "longitude"]].mean().to_numpy(dtype=float)
    pred_coords = np.repeat(coord[None, :], len(test), axis=0)

    # This baseline is a coordinate prediction, not a geocell classifier.  -1
    # makes that distinction explicit while retaining the prediction-file schema.
    pred_cells = np.full(len(test), -1, dtype=np.int64)
    return _prediction_frame(test, pred_coords, pred_cells, true_cells=true_cells), country, int(country_counts.iloc[0])


def save_and_print(name, preds, out_root):
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "preds_test.csv")
    preds.to_csv(path, index=False)
    metrics = summarize(preds["dist_km"].to_numpy())
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n{name}")
    print(f"  predictions: {path}")
    for k, v in metrics.items():
        if k != "n":
            print(f"  {k}: {v:.6f}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--cell_column", default="cell_q500")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default=DEFAULT_OUT)
    args = ap.parse_args()

    train = pd.read_csv(args.train, dtype={"id": str})
    test = pd.read_csv(args.test, dtype={"id": str})

    random_preds = random_geocell_baseline(train, test, args.cell_column, args.seed)
    save_and_print("baseline_random", random_preds, args.out_root)

    common_preds, country, count = common_country_baseline(train, test, args.cell_column)
    path = save_and_print("baseline_common_country", common_preds, args.out_root)
    country_acc = float(np.mean(test["country"].astype(str).to_numpy() == country))
    print(f"  most common country: {country} ({count} training rows)")
    print(f"  country_accuracy: {country_acc:.6f}")
    with open(os.path.join(os.path.dirname(path), "baseline.json"), "w") as f:
        json.dump({"country": country, "train_count": count, "country_accuracy": country_acc}, f, indent=2)


if __name__ == "__main__":
    main()
