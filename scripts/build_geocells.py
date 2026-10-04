"""Add our own quadtree geocell columns to data/processed/{train,val,test}.csv.

Cells are built from TRAIN coordinates only; val/test are assigned to the containing cell
(or the nearest train cell if that leaf is empty). Column `cell_q<max>` = quadtree where each
cell holds at most <max> train images (cells with < 10 are merged into a neighbour).

    python scripts/build_geocells.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.geo.geocells import build_quadtree  # noqa: E402

OUT = "data/processed"
MAX_PER_CELL = (2500, 500, 100)  # coarse / medium / fine
MIN_PER_CELL = 10


def main():
    dfs = {s: pd.read_csv(os.path.join(OUT, f"{s}.csv"), dtype={"id": str, "sequence": str})
           for s in ("train", "val", "test")}
    tr = dfs["train"]
    query = pd.concat([dfs["val"], dfs["test"]])
    n_val = len(dfs["val"])
    for m in MAX_PER_CELL:
        col = f"cell_q{m}"
        t, q = build_quadtree(tr.latitude.to_numpy(), tr.longitude.to_numpy(),
                              query.latitude.to_numpy(), query.longitude.to_numpy(), m, MIN_PER_CELL)
        dfs["train"][col], dfs["val"][col], dfs["test"][col] = t, q[:n_val], q[n_val:]
        c = np.bincount(t)
        print(f"{col}: {len(c)} cells, train imgs/cell median={int(np.median(c))} min={c.min()} max={c.max()}")
    for s, df in dfs.items():
        df.to_csv(os.path.join(OUT, f"{s}.csv"), index=False)


if __name__ == "__main__":
    main()
