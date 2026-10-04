"""Training pipeline (owner: Bryce).

Stage 1: frozen ResNet-50, train head.  Stage 2: unfreeze, fine-tune at low LR.
AdamW, AMP, checkpointing, early stopping on val GeoScore.

    python -m src.geo.train --config configs/default.yaml
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
<<<<<<< HEAD
from torch.utils.data import DataLoader
=======
from torch.utils.data import DataLoader, Subset
>>>>>>> 22b6c38a9ee01defb5cb32f8bb62241742a7f15b
from tqdm import tqdm

from .data import OSVDataset
from .geocells import build_cell_index, save_cell_index
from .losses import GeoLoss
from .metrics import haversine_np, summarize
from .model import GeoModel


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, centroids, device):
    model.eval()
    dists = []
    for x, _, coords in tqdm(loader, desc="eval", leave=False):
        logits = model(x.to(device, non_blocking=True))
        pred = centroids[logits.argmax(1).cpu().numpy()]  # argmax cell centroid
        c = coords.numpy()
        dists.append(haversine_np(pred[:, 0], pred[:, 1], c[:, 0], c[:, 1]))
    return summarize(np.concatenate(dists))


def run_epoch(model, loader, loss_fn, opt, scaler, device, amp):
    model.train()
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()  # frozen stage: keep BN stats fixed
    tot, n = 0.0, 0
    for x, y, coords in tqdm(loader, desc="train", leave=False):
        x, y, coords = x.to(device, non_blocking=True), y.to(device), coords.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        loss, _ = loss_fn(logits.float(), y, coords)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += loss.item() * x.size(0); n += x.size(0)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cell_column", default=None, help="override data.cell_column")
<<<<<<< HEAD
=======
    ap.add_argument("--max_train", type=int, default=None, help="random train subset (smoke tests)")
    ap.add_argument("--max_val", type=int, default=None, help="random val subset (smoke tests)")
    ap.add_argument("--head_epochs", type=int, default=None)
    ap.add_argument("--finetune_epochs", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
>>>>>>> 22b6c38a9ee01defb5cb32f8bb62241742a7f15b
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dc, tc = cfg["data"], cfg["train"]
    if args.cell_column:
        dc["cell_column"] = args.cell_column
<<<<<<< HEAD
=======
    for k in ("head_epochs", "finetune_epochs", "out_dir"):
        if getattr(args, k) is not None:
            tc[k] = getattr(args, k)
>>>>>>> 22b6c38a9ee01defb5cb32f8bb62241742a7f15b
    seed_all(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = tc["amp"] and device.type == "cuda"
    out_dir = os.path.join(tc["out_dir"], dc["cell_column"])
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(dc["train_csv"])
    cell_to_idx, centroids = build_cell_index(train_df, dc["cell_column"])
    cell_to_idx = {str(k): v for k, v in cell_to_idx.items()}
    save_cell_index(os.path.join(out_dir, "cells.json"), cell_to_idx, centroids)
    print(f"{len(cell_to_idx)} cells for {dc['cell_column']}")

    def make_loader(csv, train):
        ds = OSVDataset(csv, dc["image_dir"], dc["cell_column"], cell_to_idx, train=train)
<<<<<<< HEAD
=======
        limit = args.max_train if train else args.max_val
        if limit and limit < len(ds):
            ds = Subset(ds, torch.randperm(len(ds))[:limit].tolist())
>>>>>>> 22b6c38a9ee01defb5cb32f8bb62241742a7f15b
        return DataLoader(ds, batch_size=tc["batch_size"], shuffle=train,
                          num_workers=dc["num_workers"], pin_memory=True,
                          persistent_workers=dc["num_workers"] > 0)

    train_loader = make_loader(dc["train_csv"], True)
    val_loader = make_loader(dc["val_csv"], False)

    model = GeoModel(len(cell_to_idx), cfg["model"]["pretrained"]).to(device)
    loss_fn = GeoLoss(centroids, lam=tc["lambda_haversine"]).to(device)
    scaler = torch.amp.GradScaler(enabled=amp)
    best_path = os.path.join(out_dir, "best.pt")

    best = -1.0
    stages = [("head", True, tc["head_lr"], tc["head_epochs"]),
              ("finetune", False, tc["finetune_lr"], tc["finetune_epochs"])]
    for name, frozen, lr, epochs in stages:
        model.freeze_backbone(frozen)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=tc["weight_decay"])
        bad = 0
        for ep in range(epochs):
            tr_loss = run_epoch(model, train_loader, loss_fn, opt, scaler, device, amp)
            m = evaluate(model, val_loader, centroids, device)
            print(f"[{name} {ep}] loss={tr_loss:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
            ckpt = {"model": model.state_dict(), "metrics": m, "cfg": cfg, "stage": name, "epoch": ep}
            torch.save(ckpt, os.path.join(out_dir, "last.pt"))
            if m["geoscore"] > best:
                best, bad = m["geoscore"], 0
                torch.save(ckpt, best_path)
            else:
                bad += 1
                if bad >= tc["patience"]:
                    print("early stop")
                    break
        # roll back to best checkpoint before next stage / at end
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    print(f"best val geoscore {best:.1f}")


if __name__ == "__main__":
    main()
