"""
Loads the team's final (best) checkpoint and predicts where a photo was taken.
Reuses the repo's own code (GeoModel and the eval transforms) so preprocessing and the model definition match training exactly. Nothing in the repo is modified.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torchvision import transforms as T

# Makes repo's code importable
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.geo.data import get_transforms  # noqa: E402  (the team's eval transforms)
from src.geo.model import GeoModel  # noqa: E402

def load_config():
    with open(DEMO_DIR / "config.yaml") as f:
        return yaml.safe_load(f)

# Turns any uploaded image into something the model can take
# Any uploaded image -> plain RGB: honors phone rotation (EXIF) and flattens transparency.
def to_rgb(img):
    # Handles photos taken on phones that are rotated
    img = ImageOps.exif_transpose(img)
    # Handles transparent PNGs by flattening them onto white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    # Converts to RGB
    return img.convert("RGB")

# Previews model input after resize/crop
def model_input_preview(img):
    return T.CenterCrop(224)(T.Resize(224)(to_rgb(img)))

# Predictor class
# Loads model once (slow, ~seconds), then can call .predict(image) as many times (fast)
class Predictor:

    def __init__(self, checkpoint_dir=None, device=None, top_k=None):
        cfg = load_config()
        ckpt_dir = Path(checkpoint_dir or cfg["checkpoint_dir"])
        if not ckpt_dir.is_absolute():
            ckpt_dir = REPO_ROOT / ckpt_dir
        dev = device or cfg.get("device", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.top_k = top_k or cfg.get("top_k", 5)
        self.checkpoint_dir = ckpt_dir
        self.model, self.centroids, self.meta = self._load(ckpt_dir)
        self.transform = get_transforms(train=False)  # Resize(224), CenterCrop(224), ImageNet normalize


    # _load runs automatically when Predictor() is called
    def _load(self, ckpt_dir):
        # Both files must exist: best.pt (the learned weights) and cells.json (cell number -> lat/lon).
        best_path, cells_path = ckpt_dir / "best.pt", ckpt_dir / "cells.json"
        for p in (best_path, cells_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}. Run `git pull` to get the checkpoint Bryce committed "
                                        "(checkpoints/q500_main/), or fix checkpoint_dir in demo/config.yaml.")

        # Read the cell centroids (lat lon) from cells.json
        with open(cells_path) as f:
            centroids = np.asarray(json.load(f)["centroids"], dtype=np.float32)

        # Read the checkpoint (best weights) onto the CPU first
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)

        # Check that checkpoint and cells.json are compatible
        state = ckpt["model"]
        if "head.1.weight" not in state:
            raise KeyError("best.pt has no 'head.1.weight'. Has the team's GeoModel changed "
                           "(for example extra country/continent heads)? The demo needs updating.")
        num_cells = state["head.1.weight"].shape[0]  # size of the final layer = number of cells
        if num_cells != len(centroids):
            raise ValueError(f"best.pt has {num_cells} cells but cells.json has {len(centroids)} centroids. "
                             "They must come from the same training run.")

        # Build the empty ResNet-50 + head, fill in the trained weights.
        model = GeoModel(num_cells, pretrained=False)
        model.load_state_dict(state, strict=True) # loads weights 
        model.to(self.device).eval()

        # Returns model, centroids, and metadata
        meta = {
            "stage": ckpt.get("stage"),
            "epoch": ckpt.get("epoch"),
            "val_metrics": ckpt.get("metrics") or {},
            "cell_column": (ckpt.get("cfg") or {}).get("data", {}).get("cell_column"),
            "num_cells": num_cells,
        }
        return model, centroids, meta

    # Function to predict location of img
    # Returns a dict with:
    # - "guess": the single best cell, with its probability and lat/lon
    # - "top": the k most likely cells, with their probabilities and lat/lon
    # - "probs": the whole distribution over all cells (used later for the globe's heat layer)
    @torch.inference_mode() # Disable gradient computation
    def predict(self, img, k=None):
        k = min(k or self.top_k, len(self.centroids))
        x = self.transform(to_rgb(img)).unsqueeze(0).to(self.device) # cleans, resizes, crops, and normalizes image. Then, wraps in batch of 1 and moves to device
        probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu().numpy() # get probabilities for each cell
        order = np.argsort(-probs)[:k] # get the indices of the top k probabilities
        top = [{"cell": int(i), "prob": float(probs[i]), "lat": float(self.centroids[i, 0]), "lon": float(self.centroids[i, 1])} for i in order] # Finds lat/lon for each of the top k cells
        
        # "guess" is the single best cell (along with its probability and lat/lon)
        # "probs" is the probabilities for all cells
        return {"guess": top[0], "top": top, "probs": probs.tolist()}