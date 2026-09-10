import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Return standard training, validation, and test transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


class CropDiseaseDataset(Dataset):
    """PyTorch Dataset for multi-crop leaf disease classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str | Path = "data",
        transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = row["image_path"]

        # Resolve path
        full_path = self.image_root / rel_path
        if not full_path.exists():
            # Fallback if image_root is already part of rel_path
            full_path = Path(rel_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        label = int(row["label"])
        class_name = row["class_name"]

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
            "image_path": str(full_path),
        }


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    image_root: str | Path = "data",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Dict[str, DataLoader]:
    """Construct PyTorch DataLoaders for train, validation, and test sets."""
    all_transforms = get_transforms(image_size=image_size)

    train_ds = CropDiseaseDataset(train_df, image_root=image_root, transform=all_transforms["train"])
    val_ds = CropDiseaseDataset(val_df, image_root=image_root, transform=all_transforms["val"])

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    if test_df is not None:
        test_ds = CropDiseaseDataset(test_df, image_root=image_root, transform=all_transforms["test"])
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
