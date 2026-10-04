import unittest

import torch

from src.geo.losses import GeoLoss


class TestGeoLoss(unittest.TestCase):
    def test_predicted_coords_uses_spherical_average(self):
        centroids = torch.tensor([[0.0, 170.0], [0.0, -170.0]], dtype=torch.float32)
        logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

        loss = GeoLoss(centroids)
        pred = loss.predicted_coords(logits)

        self.assertAlmostEqual(float(pred[0, 0]), 0.0, places=2)
        self.assertGreater(abs(float(pred[0, 1])), 150.0)

    def test_distance_aware_label_smoothing_contract(self):
        centroids = torch.tensor(
            [[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]],
            dtype=torch.float32,
        )
        logits = torch.tensor(
            [[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
            dtype=torch.float32,
        )
        coords = torch.tensor([[0.0, 0.1], [0.0, 5.0]], dtype=torch.float32)

        loss = GeoLoss(
            centroids,
            lam=0.1,
            label_smoothing=0.25,
            smoothing_scale_km=500.0,
        )

        total, parts = loss(logits, torch.tensor([0, 2]), coords)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("cls", parts)
        self.assertIn("hav", parts)
        self.assertIn("smooth", parts)
        self.assertGreater(parts["smooth"], 0.0)


if __name__ == "__main__":
    unittest.main()
