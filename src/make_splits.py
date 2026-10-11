"""
Dataset Split Generator for Multi-Crop Leaf Disease Diagnosis.

Splits dataset into train/val/test (70/15/15) using GroupShuffleSplit
with base image ID grouping and class stratification, fixed seed 42.
Saves train.csv, val.csv, test.csv to splits/.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_base_id(crop: str, class_name: str, filename: str) -> str:
    """
    Extract the true base image ID.
    
    For Papaya: removes the '_aug_N' suffix so all 5 augmented variants of
    the same original physical capture share the identical group ID.
    For Potato & Rice: each capture is unique, using its stem as group ID.
    
    The returned ID is prefixed with crop and class_name to guarantee
    disjoint group IDs across distinct disease categories.
    """
    stem = Path(filename).stem
    m = re.match(r"^(.*)_aug_\d+$", stem, re.IGNORECASE)
    if m:
        base_name = m.group(1)
    else:
        base_name = stem
    return f"{crop}_{class_name}_{base_name}"


def extract_aug_num(crop: str, filename: str) -> int:
    """
    Extract offline augmentation index (1..5 for Papaya, 0 for original captures).
    """
    stem = Path(filename).stem
    m = re.match(r"^.*_aug_(\d+)$", stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def scan_dataset(data_dir: str | Path) -> pd.DataFrame:
    """
    Scan image dataset folder hierarchically and construct metadata dataframe.
    
    Expects structure:
      <data_dir>/<crop>/<class_name>/<filename>
    or:
      <data_dir>/Final Dataset/<crop>/<class_name>/<filename>
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Check if user pointed to repo root or data root containing 'Final Dataset'
    if (data_path / "Final Dataset").is_dir():
        data_path = data_path / "Final Dataset"

    records: List[Dict[str, Any]] = []
    
    for crop_dir in sorted(data_path.iterdir()):
        if not crop_dir.is_dir() or crop_dir.name.startswith("."):
            continue
        crop = crop_dir.name
        
        for class_dir in sorted(crop_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
            class_name = class_dir.name
            
            # Extract disease name (strip crop prefix if formatted like Crop___Disease)
            prefix = f"{crop}___"
            if class_name.startswith(prefix):
                disease = class_name[len(prefix):]
            else:
                disease = class_name
                
            for img_file in sorted(class_dir.iterdir()):
                if not img_file.is_file() or img_file.name.startswith("."):
                    continue
                if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                
                filename = img_file.name
                base_id = extract_base_id(crop, class_name, filename)
                aug_num = extract_aug_num(crop, filename)
                
                # Relative path from project root if possible, else from data_path
                try:
                    repo_root = Path(__file__).resolve().parent.parent
                    rel_path = img_file.relative_to(repo_root).as_posix()
                except ValueError:
                    rel_path = img_file.as_posix()
                
                records.append({
                    "image_path": rel_path,
                    "filename": filename,
                    "crop": crop,
                    "disease": disease,
                    "class_name": class_name,
                    "base_id": base_id,
                    "aug_num": aug_num,
                })

    if not records:
        raise ValueError(f"No valid image files found in {data_path}")

    df = pd.DataFrame(records)
    
    # Deterministic integer class label mapping (alphabetical order)
    unique_classes = sorted(df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    df["label"] = df["class_name"].map(class_to_idx)
    
    return df


def make_group_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train, val, and test splits using GroupShuffleSplit
    with class-wise stratification.
    
    Parameters:
    - df: DataFrame containing 'class_name' and 'base_id' columns.
    - train_ratio: Target train fraction (default 0.70).
    - val_ratio: Target validation fraction (default 0.15).
    - test_ratio: Target test fraction (default 0.15).
    - seed: Random seed for reproducibility (default 42).
    
    Guarantees:
    - Zero base image ID overlap across splits (leakage-free).
    - GroupShuffleSplit is applied per class, preserving class stratification.
    - Deterministic split reproduction.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    if "base_id" not in df.columns:
        df = df.copy()
        df["base_id"] = [
            extract_base_id(r["crop"], r["class_name"], r["filename"])
            for _, r in df.iterrows()
        ]

    temp_ratio = val_ratio + test_ratio
    relative_test_ratio = test_ratio / temp_ratio
    
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    
    for class_name, group_df in df.groupby("class_name"):
        group_df_indexed = group_df.reset_index()
        
        # Split 1: Train vs. Temp (Val + Test) using GroupShuffleSplit
        gss_train = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
        tr_local_idx, temp_local_idx = next(
            gss_train.split(group_df_indexed, groups=group_df_indexed["base_id"])
        )
        
        # Split 2: Val vs. Test from Temp using GroupShuffleSplit
        temp_df = group_df_indexed.iloc[temp_local_idx].reset_index(drop=True)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=seed)
        val_local_idx, test_local_idx = next(
            gss_val.split(temp_df, groups=temp_df["base_id"])
        )
        
        # Map local indices back to original dataframe row indices
        train_indices.extend(group_df_indexed.iloc[tr_local_idx]["index"].tolist())
        val_indices.extend(temp_df.iloc[val_local_idx]["index"].tolist())
        test_indices.extend(temp_df.iloc[test_local_idx]["index"].tolist())
    
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    # Strict validation of zero base_id leakage
    train_base = set(train_df["base_id"])
    val_base = set(val_df["base_id"])
    test_base = set(test_df["base_id"])
    
    leak_train_val = train_base & val_base
    leak_train_test = train_base & test_base
    leak_val_test = val_base & test_base
    
    if leak_train_val or leak_train_test or leak_val_test:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Overlap counts: "
            f"train-val={len(leak_train_val)}, train-test={len(leak_train_test)}, val-test={len(leak_val_test)}"
        )
        
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = "splits",
) -> None:
    """Save train.csv, val.csv, test.csv and combined full_dataset_with_splits.csv to output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    train_csv = out_path / "train.csv"
    val_csv = out_path / "val.csv"
    test_csv = out_path / "test.csv"
    combined_csv = out_path / "full_dataset_with_splits.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(combined_csv, index=False)
    
    print(f"Saved: {train_csv} ({len(train_df):,} samples)")
    print(f"Saved: {val_csv} ({len(val_df):,} samples)")
    print(f"Saved: {test_csv} ({len(test_df):,} samples)")
    print(f"Saved: {combined_csv} ({len(combined_df):,} samples)")


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print detailed breakdown of images and unique base captures per class per split."""
    total_images = len(train_df) + len(val_df) + len(test_df)
    train_base = set(train_df["base_id"])
    val_base = set(val_df["base_id"])
    test_base = set(test_df["base_id"])
    total_base = len(train_base) + len(val_base) + len(test_base)
    
    print("\n" + "=" * 88)
    print("LEAKAGE-FREE GROUP STRATIFIED SPLITS SUMMARY (70/15/15, Seed 42)")
    print("=" * 88)
    print(f"Train Split:      {len(train_df):>6,} images ({len(train_df)/total_images*100:>5.2f}%) | {len(train_base):>5,} base IDs ({len(train_base)/total_base*100:>5.2f}%)")
    print(f"Validation Split: {len(val_df):>6,} images ({len(val_df)/total_images*100:>5.2f}%) | {len(val_base):>5,} base IDs ({len(val_base)/total_base*100:>5.2f}%)")
    print(f"Test Split:       {len(test_df):>6,} images ({len(test_df)/total_images*100:>5.2f}%) | {len(test_base):>5,} base IDs ({len(test_base)/total_base*100:>5.2f}%)")
    print(f"Total:            {total_images:>6,} images (100.00%) | {total_base:>5,} base IDs (100.00%)")
    print("-" * 88)
    
    # Leakage verification assertion
    assert len(train_base & val_base) == 0, "Train-Val base ID overlap detected!"
    assert len(train_base & test_base) == 0, "Train-Test base ID overlap detected!"
    assert len(val_base & test_base) == 0, "Val-Test base ID overlap detected!"
    print("[LEAKAGE CHECK PASSED]: Exactly 0 overlapping base IDs across all splits.")
    print("-" * 88)
    
    # Per-class counts table
    all_classes = sorted(set(train_df["class_name"]).union(val_df["class_name"]).union(test_df["class_name"]))
    print(f"{'Class Name':<32} | {'Train':<14} | {'Val':<14} | {'Test':<14} | {'Total':<10}")
    print(f"{'':<32} | {'(img / base)':<14} | {'(img / base)':<14} | {'(img / base)':<14} | {'(img)':<10}")
    print("-" * 88)
    
    for c in all_classes:
        tr_c = train_df[train_df["class_name"] == c]
        va_c = val_df[val_df["class_name"] == c]
        te_c = test_df[test_df["class_name"] == c]
        
        tr_img, tr_b = len(tr_c), tr_c["base_id"].nunique()
        va_img, va_b = len(va_c), va_c["base_id"].nunique()
        te_img, te_b = len(te_c), te_c["base_id"].nunique()
        tot_img = tr_img + va_img + te_img
        
        print(
            f"{c:<32} | "
            f"{f'{tr_img:,} ({tr_b:,})':<14} | "
            f"{f'{va_img:,} ({va_b:,})':<14} | "
            f"{f'{te_img:,} ({te_b:,})':<14} | "
            f"{tot_img:<10,}"
        )
    print("=" * 88 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate leakage-free group-stratified splits (70/15/15, seed 42) to splits/."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/Final Dataset",
        help="Path to image dataset directory (default: 'data/Final Dataset')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="splits",
        help="Output directory for split CSVs (default: 'splits')",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    print(f"Scanning dataset from: {args.data_dir}...")
    df = scan_dataset(args.data_dir)
    print(f"Found {len(df):,} images across {df['class_name'].nunique()} classes and {df['base_id'].nunique():,} unique base IDs.")

    print(f"Generating splits using GroupShuffleSplit (seed={args.seed}, ratio={args.train_ratio}/{args.val_ratio}/{args.test_ratio})...")
    train_df, val_df, test_df = make_group_stratified_splits(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_splits(train_df, val_df, test_df, output_dir=args.output_dir)
    print_split_summary(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
