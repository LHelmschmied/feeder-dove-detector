from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
LABEL_TO_INT = {"other": 0, "dove": 1}

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

class FeederFrames(Dataset):
    """One item = one extracted frame. Split by clip via data/folds.csv."""
    def __init__(self, val_fold, train, folds_csv="data/folds.csv",
                 frames_csv="data/frames.csv", frame_dir="data/frames"):
        frames = pd.read_csv(frames_csv, dtype={"clip_id": str})
        folds = pd.read_csv(folds_csv, dtype={"clip_id": str})[["clip_id", "fold"]]
        df = frames.merge(folds, on="clip_id")
        df = df[df["fold"] != val_fold] if train else df[df["fold"] == val_fold]
        self.df = df.reset_index(drop=True)
        self.frame_dir = Path(frame_dir)
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.frame_dir / r["frame_file"]).convert("RGB")
        return self.tf(img), LABEL_TO_INT[r["label"]], r["clip_id"]