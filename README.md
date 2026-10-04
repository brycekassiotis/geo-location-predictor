# Geographic Location Predictor (CS 4343)

ResNet-50 (ImageNet) fine-tuned to classify OSV-5M street-view images into quadtree geocells.
Loss: `L = CE + λ·haversine`. Eval: median/mean km, acc@1/25/200/750/2500 km, GeoScore.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Data (OSV-5M, CC-BY-SA 4.0)
**Most people should NOT download raw data.** Get the cleaned data (shared link from Bryce) and unzip it
so you have `data/processed/{train,val,test}.csv` and `data/processed/images/`.

Regenerating from scratch (only if needed; ~17.5 GB download):
```bash
bash scripts/download.sh                # resumable; full test set + train shards 00-01
python scripts/prepare_data.py          # top-50 countries, resize from zips, grouped 90/10 split
<<<<<<< HEAD
=======
python scripts/build_geocells.py        # add cell_q2500 / cell_q500 / cell_q100 columns
>>>>>>> 22b6c38a9ee01defb5cb32f8bb62241742a7f15b
```
The split is deterministic (`random_state=42`). **Do not regenerate or edit the shared split** —
everyone's numbers must come from the same train/val/test.

## Train
```bash
<<<<<<< HEAD
python -m src.geo.train --config configs/default.yaml [--cell_column quadtree_10_1000]
=======
python -m src.geo.train --config configs/default.yaml [--cell_column cell_q500]
# smoke test (~1 min):
python -m src.geo.train --max_train 2500 --max_val 500 --head_epochs 1 --finetune_epochs 1 --out_dir checkpoints/smoke
>>>>>>> 22b6c38a9ee01defb5cb32f8bb62241742a7f15b
```

## Workflow
- Branch per task (`git checkout -b <name>/<task>`), push, open a PR into `main`. Pull `main` often.
- Don't edit someone else's in-progress file without saying so. `configs/default.yaml` and
  `src/geo/train.py` are shared — coordinate changes to them.
- Never touch the test set until final model selection. Tune on `val.csv` only.

---

## Interfaces (the contracts — code against these)

### 1. Processed CSVs — `data/processed/{train,val,test}.csv`
One row per image; image file is `data/processed/images/<id>.jpg` (short side 224 px, RGB).

| Column | Type | Notes |
|---|---|---|
| `id` | str | image id (read with `dtype={"id": str}`) |
| `latitude`, `longitude` | float | ground truth, degrees |
| `country` | str | ISO-2 code (top-50 countries only) |
| `region`, `sub-region`, `city` | str | admin names (may be empty) |
| `unique_country`, `unique_region`, `unique_sub-region`, `unique_city` | str | **use these for country/region/sub-region/city accuracy** |
| `sequence` | str | Mapillary capture sequence; train/val are split by this (no leakage) |
<<<<<<< HEAD
| `quadtree_10_1000` | int | OSV-5M geocell id (the only quadtree shipped in the CSV) |
| `land_cover`, `climate`, `drive_side`, `dist_sea` | num | metadata for error analysis |

New geocell granularities: add a new column (e.g. `cell_500`) via a script that writes it to all three
CSVs **using train-only statistics**, then train with `--cell_column cell_500`.
=======
| `cell_q2500`, `cell_q500`, `cell_q100` | int | **our geocells** (coarse 87 / medium 407 / fine 1841 cells). Default: `cell_q500` |
| `quadtree_10_1000` | int | OSV-5M's own cell id — too fine for our subset (9.3k cells, ~9 imgs/cell); don't train on it |
| `land_cover`, `climate`, `drive_side`, `dist_sea` | num | metadata for error analysis |

Our geocells come from `scripts/build_geocells.py`: an adaptive quadtree built on **train coordinates
only** (`cell_q<N>` = at most N train images per cell; cells with < 10 merged into a neighbour). Every
val/test row has a cell (empty-leaf rows get the nearest train cell). New granularity = add a value to
`MAX_PER_CELL` and re-run, then train with `--cell_column cell_q<N>`.
>>>>>>> 22b6c38a9ee01defb5cb32f8bb62241742a7f15b

### 2. Geocells — `src/geo/geocells.py`
```python
cell_to_idx, centroids = build_cell_index(train_df, cell_column)
# cell_to_idx: {cell_id(str) -> class index 0..C-1}
# centroids:   np.float32 [C, 2] = (lat, lon) mean of TRAIN images in each cell
```
Saved per run as `checkpoints/<cell_column>/cells.json` (`{"cells": {...}, "centroids": [[lat, lon], ...]}`).

### 3. Dataset — `src/geo/data.py`
`OSVDataset(csv, image_dir, cell_column, cell_to_idx, train)` yields
`(image [3,224,224] float, label int, coords [2] float (lat, lon))`. ImageNet-normalized.
<<<<<<< HEAD
=======
All rows are kept; a cell unseen in train gets `label = -1` (only possible with `quadtree_10_1000`).
>>>>>>> 22b6c38a9ee01defb5cb32f8bb62241742a7f15b

### 4. Model — `src/geo/model.py`
```python
model = GeoModel(num_cells)          # ResNet-50 backbone + head
logits = model(x)                    # x: [B,3,224,224] -> logits: [B, C]
model.freeze_backbone(True/False)
```
Extra heads (country/continent) must keep `forward` returning cell logits first, e.g. return a dict
`{"cell": ..., "country": ...}` **and** update `train.py` in the same PR.

### 5. Loss — `src/geo/losses.py`
```python
loss, parts = loss_fn(logits, labels, coords)
# logits [B,C] float32, labels [B] long, coords [B,2] (lat, lon) degrees
# loss: scalar tensor; parts: dict of floats for logging, e.g. {"cls": .., "hav": ..}
```
Any new loss (label smoothing, etc.) is an `nn.Module` with this exact call signature, constructed
from `centroids` + hyperparameters, so it drops into `train.py` with one line.

### 6. Metrics — `src/geo/metrics.py`
```python
haversine_np(lat1, lon1, lat2, lon2) -> km array   # also haversine_torch
geoscore(dist_km) -> array                         # 5000 * exp(-d / 1492.7)
summarize(dist_km) -> {"median_km", "mean_km", "geoscore", "acc@1km", ..., "acc@2500km"}
```

### 7. Predictions file — `results/<run_name>/preds_<split>.csv`
**The handoff between training and all analysis** (eval, baselines, per-continent, maps, Grad-CAM picks).
Baselines write the same format so every table is built by one script.

| Column | Notes |
|---|---|
| `id` | matches processed CSV |
| `true_lat`, `true_lon` | ground truth |
| `pred_lat`, `pred_lon` | predicted coords (centroid of argmax cell) |
| `pred_cell`, `true_cell` | class indices |
| `confidence` | softmax prob of predicted cell |
| `dist_km` | haversine error |

Join back to the processed CSV on `id` for country/continent/climate breakdowns.
Until a model exists, build analysis code against a fake file (e.g. predict random training rows).

### 8. Checkpoints — `checkpoints/<cell_column>/{best,last}.pt`
`{"model": state_dict, "metrics": {...}, "cfg": {...}, "stage": "head"|"finetune", "epoch": int}`.
`best.pt` = highest val GeoScore.

## Layout
| Path | Purpose |
|---|---|
| `scripts/download.sh`, `scripts/prepare_data.py` | raw download, cleaning, split |
| `src/geo/geocells.py` | cell index + centroids |
| `src/geo/data.py`, `model.py`, `train.py` | training pipeline |
| `src/geo/losses.py` | CE + haversine (+ smoothing / extra heads) |
| `src/geo/metrics.py` | distance metrics, GeoScore |
| `scripts/evaluate.py`, `scripts/baselines.py` | test eval, baselines (write preds files) |
| `notebooks/` | granularity studies, maps, demo |

Data: OSV-5M (Astruc et al., CVPR 2024), CC-BY-SA 4.0 — https://huggingface.co/datasets/osv5m/osv5m
