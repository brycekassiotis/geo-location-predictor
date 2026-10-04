"""L = L_cls + lambda * L_haversine (owner: Madison for smoothing variants)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import haversine_torch


class GeoLoss(nn.Module):
    def __init__(self, centroids, lam=0.1, dist_scale_km=1000.0):
        super().__init__()
        self.register_buffer("centroids", torch.as_tensor(centroids, dtype=torch.float32))
        self.lam = lam
        self.scale = dist_scale_km  # normalize km so the two terms are comparable

    def predicted_coords(self, logits):
        # Expected centroid under softmax. NOTE: naive lat/lon averaging breaks across the
        # antimeridian; fine for a first pass, swap for 3D unit-vector averaging later.
        return F.softmax(logits, dim=1) @ self.centroids

    def forward(self, logits, labels, coords):
        l_cls = F.cross_entropy(logits, labels)
        pred = self.predicted_coords(logits)
        l_hav = haversine_torch(pred[:, 0], pred[:, 1], coords[:, 0], coords[:, 1]).mean() / self.scale
        return l_cls + self.lam * l_hav, {"cls": l_cls.item(), "hav": l_hav.item()}

# TODO(Madison): distance-aware label smoothing (PIGEON-style haversine smoothing).
