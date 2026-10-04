"""Dataset + transforms."""
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),  # NOTE: may hurt (driving side / text); ablate
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(224), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class OSVDataset(Dataset):
    def __init__(self, csv_path, image_dir, cell_column, cell_to_idx, train=False):
        df = pd.read_csv(csv_path, dtype={"id": str})
        # Keep every row: eval must cover the whole split. Cells unseen in train get label -1
        # (only distance metrics use them; never happens for the train split).
        self.ids = df["id"].tolist()
        self.labels = [cell_to_idx.get(str(c), -1) for c in df[cell_column]]
        self.coords = torch.tensor(df[["latitude", "longitude"]].to_numpy(), dtype=torch.float32)
        self.image_dir = image_dir
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.image_dir, f"{self.ids[i]}.jpg")).convert("RGB")
        return self.tf(img), self.labels[i], self.coords[i]
