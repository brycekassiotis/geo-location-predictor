"""Geocell utilities.

OSV-5M's CSV only ships quadtree_10_1000, which is built for 4.8M images and far too fine for our
subset. `build_quadtree` makes our own cells from TRAIN coordinates only (scripts/build_geocells.py
writes them as cell_q<max> columns). `build_cell_index` maps raw cell ids -> contiguous class
indices and computes each cell's centroid (mean lat/lon of its training images).
"""
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def _nearest_cell(cent, lat, lon):
    tree = BallTree(np.radians(cent[["lat", "lon"]].to_numpy()), metric="haversine")
    _, nn = tree.query(np.radians(np.c_[lat, lon]), k=1)
    return cent.index.to_numpy()[nn[:, 0]]


def build_quadtree(train_lat, train_lon, query_lat, query_lon, max_per_cell, min_per_cell=10,
                   max_depth=14):
    """Adaptive lat/lon quadtree: split any box holding > max_per_cell train points.

    Returns (train_cells, query_cells) as int arrays with contiguous ids 0..C-1. Leaves with
    < min_per_cell train points are merged into the nearest larger cell; query points that land
    in an empty leaf are assigned the nearest train-cell centroid.
    """
    t_cells = np.full(len(train_lat), -1, dtype=np.int64)
    q_cells = np.full(len(query_lat), -1, dtype=np.int64)
    next_id = [0]

    def rec(ti, qi, lat0, lat1, lon0, lon1, depth):
        if len(ti) > max_per_cell and depth < max_depth:
            lm, om = (lat0 + lat1) / 2, (lon0 + lon1) / 2
            tn, te = train_lat[ti] >= lm, train_lon[ti] >= om
            qn, qe = query_lat[qi] >= lm, query_lon[qi] >= om
            for n, e, box in ((False, False, (lat0, lm, lon0, om)), (False, True, (lat0, lm, om, lon1)),
                              (True, False, (lm, lat1, lon0, om)), (True, True, (lm, lat1, om, lon1))):
                rec(ti[(tn == n) & (te == e)], qi[(qn == n) & (qe == e)], *box, depth + 1)
        elif len(ti) > 0:
            t_cells[ti] = q_cells[qi] = next_id[0]
            next_id[0] += 1

    rec(np.arange(len(train_lat)), np.arange(len(query_lat)), -90.0, 90.01, -180.0, 180.01, 0)

    g = pd.DataFrame({"c": t_cells, "lat": train_lat, "lon": train_lon}).groupby("c")
    cent, counts = g.mean(), g.size()
    big = cent[counts >= min_per_cell]

    small_t = ~np.isin(t_cells, big.index)
    if small_t.any():
        t_cells[small_t] = _nearest_cell(big, train_lat[small_t], train_lon[small_t])
    orphan_q = ~np.isin(q_cells, big.index)
    if orphan_q.any():
        q_cells[orphan_q] = _nearest_cell(big, query_lat[orphan_q], query_lon[orphan_q])

    remap = {c: i for i, c in enumerate(sorted(big.index))}
    return np.vectorize(remap.get)(t_cells), np.vectorize(remap.get)(q_cells)


def build_cell_index(train_df: pd.DataFrame, cell_column: str):
    cells = sorted(train_df[cell_column].unique().tolist())
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    cent = train_df.groupby(cell_column)[["latitude", "longitude"]].mean().loc[cells]
    centroids = cent.to_numpy(dtype=np.float32)  # [num_cells, 2] (lat, lon)
    return cell_to_idx, centroids


def save_cell_index(path, cell_to_idx, centroids):
    with open(path, "w") as f:
        json.dump({"cells": {str(k): v for k, v in cell_to_idx.items()},
                   "centroids": centroids.tolist()}, f)


def load_cell_index(path):
    with open(path) as f:
        d = json.load(f)
    return d["cells"], np.asarray(d["centroids"], dtype=np.float32)
