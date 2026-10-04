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
Raw files go in `data/raw/` (`train.csv`, `test.csv`, `images/{train,test}/*.zip`) from
https://huggingface.co/datasets/osv5m/osv5m. Full test set + train shards 00–01 are downloaded so far.
```bash
python scripts/prepare_data.py          # unzip, top-50 countries, resize, grouped 90/10 split
python -m src.geo.train --cell_column quadtree_10_1000
```

## Layout
| Path | Purpose | Owner |
|---|---|---|
| `scripts/prepare_data.py` | unzip / resize / split | Owen |
| `src/geo/geocells.py` | cell index + centroids | Owen |
| `src/geo/train.py`, `model.py`, `data.py` | training pipeline | Bryce |
| `src/geo/losses.py` | CE + haversine, label smoothing, extra heads | Madison |
| `src/geo/metrics.py`, `scripts/evaluate.py`, `scripts/baselines.py` | metrics, baselines, test eval | Jack |
| `notebooks/` | granularity studies, maps, demo | Raghavan |
