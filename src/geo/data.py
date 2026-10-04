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
        df = pd.read_csv(csv_path)
        df = df[df[cell_column].astype(str).isin(cell_to_idx)].reset_index(drop=True)
        self.df = df
        self.image_dir = image_dir
        self.cell_column = cell_column
        self.cell_to_idx = cell_to_idx
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.image_dir, f"{r['id']}.jpg")).convert("RGB")
        label = self.cell_to_idx[str(r[self.cell_column])]
        coords = torch.tensor([r["latitude"], r["longitude"]], dtype=torch.float32)
        return self.tf(img), label, coords
