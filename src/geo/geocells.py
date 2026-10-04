"""Geocell utilities (owner: Owen).

OSV-5M ships precomputed quadtree cell columns (quadtree_10_500, _2500, _25000).
We map raw cell ids -> contiguous class indices and compute each cell's centroid
(mean lat/lon of its *training* images) for distance-based prediction/loss.
"""
import json
import numpy as np
import pandas as pd


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
