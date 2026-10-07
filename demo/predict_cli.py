"""
Predicts where one photo was taken from the command line
- python demo/predict_cli.py "C:/path/to/photo.jpg"
- ... --truth 48.8584,2.2945 (also prints km error + GeoScore)
- ... --save-crop ../crop.jpg (saves the 224x224 crop the model sees)
"""
import argparse
import time

import torch
from PIL import Image

from predictor import Predictor, model_input_preview 
from src.geo.metrics import geoscore, haversine_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--truth", help="true location as LAT,LON (optional)")
    ap.add_argument("--save-crop", help="write the 224x224 model input to this file")
    args = ap.parse_args()

    t0 = time.time()
    pred = Predictor()
    load_s = time.time() - t0
    dev = pred.device.type
    gpu = f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""
    m = pred.meta["val_metrics"]
    stored = (f"stored val: geoscore={m['geoscore']:.0f}, median={m['median_km']:.0f} km"
              if "geoscore" in m else "no metrics stored")
    print(f"device: {dev}{gpu}")
    print(f"checkpoint: {pred.checkpoint_dir}")
    print(f"  cells={pred.meta['num_cells']} cell_column={pred.meta['cell_column']} "
          f"stage={pred.meta['stage']} epoch={pred.meta['epoch']}  {stored}")

    img = Image.open(args.image)
    print(f"image: {args.image}  size={img.size} mode={img.mode}")

    t0 = time.time()
    pred.predict(img)  # first call includes GPU warm-up
    first_s = time.time() - t0
    t0 = time.time()
    out = pred.predict(img)
    print(f"load model {load_s:.1f}s | first predict {first_s:.2f}s | second predict {time.time() - t0:.3f}s")

    print(f"top {len(out['top'])} guesses:")
    for rank, c in enumerate(out["top"], 1):
        print(f"  {rank}. {100 * c['prob']:5.1f}%   lat {c['lat']:7.2f}   lon {c['lon']:8.2f}   (cell {c['cell']})")

    if args.truth:
        tlat, tlon = (float(v) for v in args.truth.split(","))
        g = out["guess"]
        d = float(haversine_np(g["lat"], g["lon"], tlat, tlon))
        print(f"truth {tlat:.4f}, {tlon:.4f} -> error {d:.0f} km, GeoScore {float(geoscore(d)):.0f} / 5000")

    if args.save_crop:
        model_input_preview(img).save(args.save_crop)
        print(f"saved the model's 224x224 input to {args.save_crop}")


if __name__ == "__main__":
    main()