"""Geographic evaluation metrics (owner: Jack).

Distance metrics are intentionally independent of the model so they can be used for
models, baselines, and saved prediction CSVs alike.
"""
import numpy as np
import torch

EARTH_RADIUS_KM = 6371.0
THRESHOLDS_KM = (1, 25, 200, 750, 2500)


def haversine_torch(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs are tensors containing degrees."""
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (lat1, lon1, lat2, lon2))
    a = (
        torch.sin((lat2 - lat1) / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))


def haversine_np(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs are array-like degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geoscore(dist_km):
    """OSV-5M-style GeoScore for one or more distance errors."""
    return 5000 * np.exp(-np.asarray(dist_km, dtype=float) / 1492.7)


def summarize(dist_km):
    """Return the project's primary distance metrics as a JSON-friendly dict."""
    d = np.asarray(dist_km, dtype=float).reshape(-1)
    if d.size == 0:
        raise ValueError("Cannot summarize an empty distance array")
    out = {
        "n": int(d.size),
        "median_km": float(np.median(d)),
        "mean_km": float(d.mean()),
        "geoscore": float(np.mean(geoscore(d))),
    }
    for t in THRESHOLDS_KM:
        out[f"acc@{t}km"] = float(np.mean(d <= t))
    return out


def accuracy_at_thresholds(dist_km, thresholds=THRESHOLDS_KM):
    """Return {threshold_km: accuracy} for arbitrary distance thresholds."""
    d = np.asarray(dist_km, dtype=float).reshape(-1)
    if d.size == 0:
        raise ValueError("Cannot calculate accuracy on an empty distance array")
    return {int(t): float(np.mean(d <= t)) for t in thresholds}


def classification_accuracy(true_values, pred_values):
    """Simple exact-match accuracy, ignoring rows with missing predictions."""
    true_values = np.asarray(true_values, dtype=object)
    pred_values = np.asarray(pred_values, dtype=object)
    valid = np.array([x is not None and not _missing(x) and not _missing(y)
                      for x, y in zip(true_values, pred_values)])
    if not valid.any():
        return float("nan")
    return float(np.mean(true_values[valid] == pred_values[valid]))


def _missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return str(value).strip() == ""

