"""Overnight experiment queue: hyperparameter sweep -> full runs with the best settings.

    python scripts/overnight.py            # real run (~7 h on an RTX 4060 laptop)
    python scripts/overnight.py --smoke    # tiny sizes, checks the whole queue in a few minutes

Phase 1 (sweep, 20k-image train subset, 3k val subset): finetune_lr x lambda_haversine.
Phase 2 (full data, best sweep settings): main model, label smoothing, granularity, learning curve.

Resumable: a run whose results/<name>/metrics.json exists is skipped, so re-running after a
crash/sleep continues where it stopped. One failed run does not stop the queue.
Outputs: checkpoints/<name>/{best,last}.pt (not in git), results/<name>/{history.json,
metrics.json, preds_val.csv, train.log}, results/overnight_summary.csv.
"""
import argparse
import ctypes
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

SWEEP_LRS = (1e-5, 3e-5, 1e-4)
SWEEP_LAMBDAS = (0.0, 0.1, 0.5)
SWEEP_ARGS = ["--max_train", "20000", "--max_val", "3000", "--head_epochs", "2",
              "--finetune_epochs", "3", "--no_preds"]
FULL_RUNS = [  # (name, extra args) - run in this order, most important first
    ("q500_main", ["--cell_column", "cell_q500"]),
    ("q500_smooth", ["--cell_column", "cell_q500", "--label_smoothing", "0.1"]),
    ("q2500", ["--cell_column", "cell_q2500"]),
    ("q100", ["--cell_column", "cell_q100"]),
    ("lc_10k", ["--cell_column", "cell_q500", "--max_train", "10000"]),
    ("lc_40k", ["--cell_column", "cell_q500", "--max_train", "40000"]),
]
SMOKE_ARGS = ["--max_train", "300", "--max_val", "100", "--head_epochs", "1", "--finetune_epochs", "1"]


def keep_awake():
    """Block idle sleep on Windows while the queue runs (does not override closing the lid)."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run(name, args, root):
    res_dir = os.path.join(root, "results", name)
    metrics_path = os.path.join(res_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {name} (done)", flush=True)
        return json.load(open(metrics_path))
    os.makedirs(res_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "src.geo.train", "--run_dir", os.path.join(root, "checkpoints", name),
           "--results_dir", res_dir, *args]
    print(f"[start] {name}  {time.strftime('%H:%M')}  {' '.join(args)}", flush=True)
    t0 = time.time()
    with open(os.path.join(res_dir, "train.log"), "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
    mins = (time.time() - t0) / 60
    if rc != 0 or not os.path.exists(metrics_path):
        print(f"[FAIL] {name} rc={rc} after {mins:.0f} min - see {res_dir}/train.log", flush=True)
        return None
    m = json.load(open(metrics_path))
    print(f"[done] {name} {mins:.0f} min  val geoscore={m['best_val']['geoscore']:.0f} "
          f"median={m['best_val']['median_km']:.0f} km", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    keep_awake()
    root = os.path.join("smoke_overnight") if a.smoke else "."
    rows = []

    # Phase 1: sweep
    grid = list(itertools.product(SWEEP_LRS, SWEEP_LAMBDAS))
    for lr, lam in (grid[:2] if a.smoke else grid):
        name = f"sweep/lr{lr:g}_lam{lam:g}"
        m = run(name, ["--finetune_lr", str(lr), "--lambda_haversine", str(lam),
                       *(SMOKE_ARGS + ["--no_preds"] if a.smoke else SWEEP_ARGS)], root)
        shutil.rmtree(os.path.join(root, "checkpoints", name), ignore_errors=True)  # save disk
        if m:
            rows.append({"run": name, "finetune_lr": lr, "lambda": lam, **m["best_val"]})
    sweep = pd.DataFrame(rows)
    if sweep.empty:
        print("all sweep runs failed - using config defaults", flush=True)
        best = {"finetune_lr": None, "lambda": None}
    else:
        best = sweep.sort_values("geoscore", ascending=False).iloc[0].to_dict()
        print(f"[sweep best] finetune_lr={best['finetune_lr']:g} lambda={best['lambda']:g} "
              f"geoscore={best['geoscore']:.0f}", flush=True)
    best_args = [] if best["finetune_lr"] is None else \
        ["--finetune_lr", str(best["finetune_lr"]), "--lambda_haversine", str(best["lambda"])]

    # Phase 2: full runs with the best settings
    for name, extra in ([FULL_RUNS[1], FULL_RUNS[3]] if a.smoke else FULL_RUNS):
        if a.smoke:  # keep cell/smoothing flags, shrink sizes
            extra = [x for x in extra if x != "--max_train"] + SMOKE_ARGS
        m = run(name, extra + best_args, root)
        if m:
            rows.append({"run": name, **m["best_val"], "num_cells": m["num_cells"]})

    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(root, "results", "overnight_summary.csv"), index=False)
    print("ALL DONE", time.strftime("%H:%M"), flush=True)


if __name__ == "__main__":
    main()
