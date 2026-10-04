"""L = L_cls + lambda * L_haversine (owner: Madison for smoothing variants)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import haversine_torch


class GeoLoss(nn.Module):
    def __init__(
        self,
        centroids,
        lam=0.1,
        dist_scale_km=1000.0,
        label_smoothing=0.0,
        smoothing_scale_km=2000.0,
    ):
        # Start with a small value (0.05 to 0.2 for label_smoothing) and adjust with lam
        super().__init__()
        self.register_buffer("centroids", torch.as_tensor(centroids, dtype=torch.float32))
        self.lam = lam
        self.scale = dist_scale_km  # normalize km so the two terms are comparable
        self.label_smoothing = float(label_smoothing)
        self.smoothing_scale_km = float(smoothing_scale_km)

        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.smoothing_scale_km <= 0.0:
            raise ValueError("smoothing_scale_km must be positive.")

    def _centroid_unit_vectors(self, device):
        lat = torch.deg2rad(self.centroids[:, 0].to(device))
        lon = torch.deg2rad(self.centroids[:, 1].to(device))
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.stack((x, y, z), dim=1)

    def predicted_coords(self, logits):
        """Predict a latitude/longitude by averaging the class centroids on the unit sphere."""
        probs = F.softmax(logits, dim=1)
        unit_vecs = self._centroid_unit_vectors(logits.device)
        mean_vec = probs @ unit_vecs
        mean_norm = mean_vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
        mean_vec = mean_vec / mean_norm

        lat = torch.atan2(mean_vec[:, 2], torch.sqrt(mean_vec[:, 0] ** 2 + mean_vec[:, 1] ** 2))
        lon = torch.atan2(mean_vec[:, 1], mean_vec[:, 0])
        return torch.stack((torch.rad2deg(lat), torch.rad2deg(lon)), dim=1)

    def _distance_aware_targets(self, logits, labels, coords):
        if self.label_smoothing <= 0.0:
            return None

        num_classes = logits.size(1)
        labels = labels.to(device=logits.device, dtype=torch.long)
        coords = coords.to(device=logits.device, dtype=torch.float32)

        one_hot = F.one_hot(labels, num_classes=num_classes).to(logits.dtype)
        centroids = self.centroids.to(logits.device)
        cent_lat = centroids[:, 0].unsqueeze(0).expand(labels.numel(), -1)
        cent_lon = centroids[:, 1].unsqueeze(0).expand(labels.numel(), -1)
        true_lat = coords[:, 0].unsqueeze(1).expand(-1, num_classes)
        true_lon = coords[:, 1].unsqueeze(1).expand(-1, num_classes)

        dists_km = haversine_torch(cent_lat, cent_lon, true_lat, true_lon)
        affinity = torch.exp(-dists_km / self.smoothing_scale_km)
        affinity = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1e-12)

        smoothed = (1.0 - self.label_smoothing) * one_hot + self.label_smoothing * affinity
        smoothed = smoothed / smoothed.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return smoothed

    def forward(self, logits, labels, coords):
        labels = labels.to(device=logits.device, dtype=torch.long)
        coords = coords.to(device=logits.device, dtype=torch.float32)

        if self.label_smoothing > 0.0:
            targets = self._distance_aware_targets(logits, labels, coords)
            l_cls = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            l_cls = F.cross_entropy(logits, labels)

        pred = self.predicted_coords(logits)
        l_hav = haversine_torch(pred[:, 0], pred[:, 1], coords[:, 0], coords[:, 1]).mean() / self.scale
        return l_cls + self.lam * l_hav, {
            "cls": l_cls.detach().item(),
            "hav": l_hav.detach().item(),
            "smooth": self.label_smoothing,
        }

# TODO(Madison): explore city/region-wise smoothing from metadata instead of centroid-only distance.
