import os
import re
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import numpy as np


def extract_base_id(crop: str, disease: str, filename: str) -> str:
    """
    Extract the true base image ID.
    For Papaya: strips the '_aug_N' suffix to bundle all augmented variants of
    the same original capture into a single group.
    For Potato & Rice: uses the full image stem as an individual group.
    """
    if crop == "Papaya":
        # Matches patterns like image_100_aug_1.jpg -> image_100
        m = re.match(r"^(.*)_aug_\d+\.[a-zA-Z0-9]+$", filename)
        if m:
            base_name = m.group(1)
            return f"{crop}_{disease}_{base_name}"
    # Default for non-augmented or other naming formats
    stem = Path(filename).stem
    return f"{crop}_{disease}_{stem}"


def prepare_group_stratified_splits(
    csv_path: str | Path,
    output_dir: str | Path = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a leakage-free, group-aware stratified train/val/test split.
    Guarantees:
      1. Zero base_id overlap between train, val, and test splits.
      2. Stratified class distribution preserved across splits.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    df = pd.read_csv(csv_path)
    if "base_id" not in df.columns:
        df["base_id"] = [
            extract_base_id(r["crop"], r["disease"], r["filename"])
            for _, r in df.iterrows()
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Class-wise grouped stratification
    train_base_ids = set()
    val_base_ids = set()
    test_base_ids = set()

    for class_name, group_df in df.groupby("class_name"):
        unique_base_ids = np.array(sorted(group_df["base_id"].unique()))
        rng.shuffle(unique_base_ids)

        n_total = len(unique_base_ids)
        n_train = int(np.floor(train_ratio * n_total))
        n_val = int(np.floor(val_ratio * n_total))

        # Ensure at least 1 base ID per split if class size allows
        if n_train == 0 and n_total >= 1:
            n_train = 1
        if n_val == 0 and (n_total - n_train) >= 2:
            n_val = 1

        train_ids = unique_base_ids[:n_train]
        val_ids = unique_base_ids[n_train : n_train + n_val]
        test_ids = unique_base_ids[n_train + n_val :]

        train_base_ids.update(train_ids)
        val_base_ids.update(val_ids)
        test_base_ids.update(test_ids)

    # Assign split column
    def get_split(base_id: str) -> str:
        if base_id in train_base_ids:
            return "train"
        if base_id in val_base_ids:
            return "val"
        return "test"

    df["split"] = df["base_id"].apply(get_split)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    # Verify zero leakage
    train_groups = set(train_df["base_id"])
    val_groups = set(val_df["base_id"])
    test_groups = set(test_df["base_id"])

    leak_train_val = train_groups & val_groups
    leak_train_test = train_groups & test_groups
    leak_val_test = val_groups & test_groups

    if leak_train_val or leak_train_test or leak_val_test:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Overlap: train-val={len(leak_train_val)}, "
            f"train-test={len(leak_train_test)}, val-test={len(leak_val_test)}"
        )

    # Save to disk
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    df.to_csv(output_dir / "full_dataset_with_splits.csv", index=False)

    return train_df, val_df, test_df
