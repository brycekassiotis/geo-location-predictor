"""Distance metrics (owner: Jack). Haversine / GeoScore / acc@km."""
import math
import numpy as np
import torch

EARTH_RADIUS_KM = 6371.0
THRESHOLDS_KM = (1, 25, 200, 750, 2500)


def haversine_torch(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs in degrees (tensors)."""
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (lat1, lon1, lat2, lon2))
    a = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))


def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geoscore(dist_km):
    return 5000 * np.exp(-np.asarray(dist_km) / 1492.7)


def summarize(dist_km):
    d = np.asarray(dist_km)
    out = {
        "median_km": float(np.median(d)),
        "mean_km": float(d.mean()),
        "geoscore": float(geoscore(d).mean()),
    }
    for t in THRESHOLDS_KM:
        out[f"acc@{t}km"] = float((d <= t).mean())
    return out

# TODO(Jack): country/region/sub-region/city accuracy (reverse geocode predicted coords),
# per-continent breakdown.
