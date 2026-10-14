import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transforms(image_size: int = 224, resize_dim: int = 256) -> Dict[str, transforms.Compose]:
    """
    Standard transforms following requirements:
    Resize preserving aspect ratio then center-crop to 224, flip, rotation 15, color jitter.
    No other tricks yet.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }



class RandomShadow:
    """Random shadow augmentation simulating outdoor lighting variations."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        import numpy as np
        if random.random() > self.p:
            return img
        w, h = img.size
        shadow_factor = random.uniform(0.5, 0.8)
        arr = np.array(img).astype(np.float32)
        mask = np.linspace(1.0, shadow_factor, w).reshape(1, w, 1)
        if random.random() > 0.5:
            mask = np.flip(mask, axis=1)
        arr = np.clip(arr * mask, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def get_strong_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Stronger augmentation pipeline for training from scratch:
    RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter,
    GaussianBlur, and RandomShadow.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        RandomShadow(p=0.3),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
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

        # Crop mapping: Papaya=0, Potato=1, Rice=2
        crop_name = row["crop"] if "crop" in row else class_name.split("___")[0]
        crop_map = {"Papaya": 0, "Potato": 1, "Rice": 2}
        crop_label = crop_map.get(crop_name, label // 4)

        group_id = row["group_id"] if "group_id" in row else idx

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "crop_label": torch.tensor(crop_label, dtype=torch.long),
            "class_name": class_name,
            "crop": crop_name,
            "image_path": str(full_path),
            "group_id": group_id,
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
