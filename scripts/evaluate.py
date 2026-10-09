"""Final evaluation (owner: Jack).

Evaluates a saved prediction CSV, or optionally creates test predictions from a
trained checkpoint. The test split should only be used after final model selection.

Examples:
    # Evaluate an already-created prediction file:
    python scripts/evaluate.py --preds results/q500_main/preds_test.csv

    # Generate test predictions from the final checkpoint and evaluate them:
    python scripts/evaluate.py --checkpoint checkpoints/cell_q500/best.pt \
        --cell_column cell_q500
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.geo.data import OSVDataset
from src.geo.geocells import load_cell_index
from src.geo.metrics import haversine_np, summarize, classification_accuracy
from src.geo.model import GeoModel

DEFAULT_TEST = "data/processed/test.csv"
DEFAULT_IMAGE_DIR = "data/processed/images"
DEFAULT_RESULTS = "results"


def _clean_series(s):
    return s.fillna("").astype(str).str.strip()


def _country_to_continent():
    # ISO-2 country code mapping for the 50-country project subset. Unknown
    # codes are left as "Unknown" rather than guessed.
    return {
        "AR": "South America", "AU": "Oceania", "AT": "Europe", "BE": "Europe",
        "BR": "South America", "CA": "North America", "CH": "Europe", "CL": "South America",
        "CN": "Asia", "CO": "South America", "CZ": "Europe", "DE": "Europe",
        "DK": "Europe", "DO": "North America", "DZ": "Africa", "EC": "South America",
        "EG": "Africa", "ES": "Europe", "FI": "Europe", "FR": "Europe",
        "GB": "Europe", "GR": "Europe", "HR": "Europe", "HU": "Europe",
        "ID": "Asia", "IE": "Europe", "IL": "Asia", "IN": "Asia",
        "IT": "Europe", "JP": "Asia", "KE": "Africa", "KR": "Asia",
        "MA": "Africa", "MX": "North America", "MY": "Asia", "NG": "Africa",
        "NL": "Europe", "NO": "Europe", "NZ": "Oceania", "PE": "South America",
        "PH": "Asia", "PL": "Europe", "PT": "Europe", "RO": "Europe",
        "RU": "Europe", "SA": "Asia", "SE": "Europe", "SG": "Asia",
        "TH": "Asia", "TR": "Asia", "TW": "Asia", "UA": "Europe",
        "US": "North America", "VE": "South America", "VN": "Asia", "ZA": "Africa",
    }


def add_continent(df):
    mapping = _country_to_continent()
    countries = _clean_series(df["country"])
    df = df.copy()
    df["continent"] = countries.str.upper().map(mapping).fillna("Unknown")
    return df


def evaluate_prediction_frame(preds, metadata):
    """Join predictions to OSV metadata and return all evaluation outputs."""
    preds = preds.copy()
    preds["id"] = preds["id"].astype(str)
    metadata = metadata.copy()
    metadata["id"] = metadata["id"].astype(str)

    required_pred = {"id", "true_lat", "true_lon", "pred_lat", "pred_lon", "dist_km"}
    missing = required_pred - set(preds.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")

    df = metadata.merge(preds, on="id", how="inner", suffixes=("_meta", "_pred"))
    if len(df) != len(preds):
        missing_ids = len(preds) - len(df)
        raise ValueError(f"{missing_ids} prediction rows did not match test metadata by id")

    # Recompute distance from coordinates instead of trusting a cached value.
    df["dist_km"] = haversine_np(
        df["true_lat"].to_numpy(float), df["true_lon"].to_numpy(float),
        df["pred_lat"].to_numpy(float), df["pred_lon"].to_numpy(float),
    )

    results = {"distance": summarize(df["dist_km"].to_numpy())}

    # OSV-5M's unique_* columns are the project's intended accuracy labels.
    # A prediction coordinate is reverse-geocoded only when reverse_geocoder is
    # available; distance metrics never depend on this optional step.
    admin = reverse_geocode_predictions(df)
    if admin is not None:
        for level, pred_col, true_col in (
            ("country", "pred_country", "unique_country"),
            ("region", "pred_region", "unique_region"),
            ("sub-region", "pred_sub-region", "unique_sub-region"),
            ("city", "pred_city", "unique_city"),
        ):
            if true_col in df.columns and pred_col in df.columns:
                results[f"{level}_accuracy"] = classification_accuracy(
                    _normalize_admin(df[true_col]), _normalize_admin(df[pred_col])
                )

    df = add_continent(df)
    continent_rows = []
    for continent, group in df.groupby("continent", sort=True):
        m = summarize(group["dist_km"].to_numpy())
        continent_rows.append({"continent": continent, **m})
    results["continents"] = continent_rows

    return results, df


def _normalize_admin(series):
    return _clean_series(series).str.casefold()


def reverse_geocode_predictions(df):
    """Return predicted country/region/city columns when reverse_geocoder exists."""
    try:
        import reverse_geocoder as rg
    except ImportError:
        print("WARNING: reverse_geocoder is not installed; administrative accuracy will be skipped.")
        return None

    coords = list(zip(df["pred_lat"].astype(float), df["pred_lon"].astype(float)))
    if not coords:
        return df
    hits = rg.search(coords, mode="batch")
    df["pred_country"] = [h.get("cc", "") for h in hits]
    df["pred_region"] = [h.get("admin1", "") for h in hits]
    df["pred_city"] = [h.get("name", "") for h in hits]
    # reverse_geocoder does not expose a reliable OSV sub-region field.
    df["pred_sub-region"] = ""
    return df


def print_report(results, title="Evaluation"):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print("Distance metrics")
    for k, v in results["distance"].items():
        if k == "n":
            print(f"  n:              {v}")
        elif k == "geoscore":
            print(f"  GeoScore:       {v:.2f}")
        elif k.startswith("acc@"):
            print(f"  {k:14s}: {100*v:7.3f}%")
        else:
            print(f"  {k:14s}: {v:,.2f}")

    admin_keys = ["country_accuracy", "region_accuracy", "sub-region_accuracy", "city_accuracy"]
    if any(k in results for k in admin_keys):
        print("\nAdministrative accuracy")
        for k in admin_keys:
            if k in results and np.isfinite(results[k]):
                print(f"  {k:22s}: {100*results[k]:7.3f}%")

    print("\nPer-continent breakdown")
    headers = ["continent", "n", "median_km", "mean_km", "geoscore", "acc@25km", "acc@200km", "acc@750km", "acc@2500km"]
    print("  " + " | ".join(f"{h:>14s}" for h in headers))
    for row in results["continents"]:
        print("  " + " | ".join(
            f"{row.get(h, '') if h == 'continent' else row.get(h, 0):>14}"
            if h in ("continent", "n") else
            f"{row.get(h, 0):14.2f}" for h in headers
        ))


def _write_results(results, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def make_test_predictions(checkpoint, test_csv, image_dir, cell_column, batch_size, workers, device):
    """Load a final checkpoint and produce the repo-standard test prediction frame."""
    ckpt = torch.load(checkpoint, map_location=device)
    run_dir = os.path.dirname(checkpoint)
    cells_path = os.path.join(run_dir, "cells.json")
    if not os.path.exists(cells_path):
        raise FileNotFoundError(f"Could not find cells.json beside checkpoint: {cells_path}")
    cell_to_idx, centroids = load_cell_index(cells_path)
    cell_to_idx = {str(k): int(v) for k, v in cell_to_idx.items()}

    ds = OSVDataset(test_csv, image_dir, cell_column, cell_to_idx, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=(device.type == "cuda"))
    cfg = ckpt.get("cfg", {})
    pretrained = cfg.get("model", {}).get("pretrained", False)
    model = GeoModel(len(cell_to_idx), pretrained).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_cells, all_conf = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            probs = torch.softmax(model(x.to(device, non_blocking=True)).float(), dim=1)
            conf, cell = probs.max(1)
            all_cells.append(cell.cpu().numpy())
            all_conf.append(conf.cpu().numpy())

    cells = np.concatenate(all_cells)
    conf = np.concatenate(all_conf)
    true = ds.coords.numpy()
    pred = centroids[cells]
    dist = haversine_np(pred[:, 0], pred[:, 1], true[:, 0], true[:, 1])

    return pd.DataFrame({
        "id": ds.ids,
        "true_lat": true[:, 0], "true_lon": true[:, 1],
        "pred_lat": pred[:, 0], "pred_lon": pred[:, 1],
        "pred_cell": cells, "true_cell": ds.labels,
        "confidence": conf, "dist_km": dist,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None, help="Existing results/.../preds_test.csv")
    ap.add_argument("--checkpoint", default=None, help="Final best.pt; generates preds_test.csv")
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--cell_column", default="cell_q500")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    if bool(args.preds) == bool(args.checkpoint):
        ap.error("Provide exactly one of --preds or --checkpoint")

    test = pd.read_csv(args.test, dtype={"id": str})
    if args.preds:
        pred_path = args.preds
        preds = pd.read_csv(pred_path, dtype={"id": str})
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        preds = make_test_predictions(
            args.checkpoint, args.test, args.image_dir, args.cell_column,
            args.batch_size, args.workers, device,
        )
        pred_path = os.path.join(os.path.dirname(args.checkpoint), "preds_test.csv")
        preds.to_csv(pred_path, index=False)
        print(f"Wrote predictions: {pred_path}")

    results, joined = evaluate_prediction_frame(preds, test)
    print_report(results, title=f"Final evaluation: {pred_path}")

    out_path = args.out or os.path.join(os.path.dirname(pred_path), "test_metrics.json")
    _write_results(results, out_path)
    print(f"\nWrote metrics: {out_path}")

    # Save a convenient continent table for the report/presentation.
    continent_path = os.path.join(os.path.dirname(pred_path), "continent_metrics.csv")
    pd.DataFrame(results["continents"]).to_csv(continent_path, index=False)
    print(f"Wrote continent table: {continent_path}")


if __name__ == "__main__":
    main()
