"""ResNet-50 backbone + geocell classification head."""
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class GeoModel(nn.Module):
    def __init__(self, num_cells, pretrained=True):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat_dim, num_cells))
        # TODO(Madison): extra country / continent heads

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self, frozen=True):
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        # keep BN stats fixed while frozen
        self.backbone.train(not frozen)
