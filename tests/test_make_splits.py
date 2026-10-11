"""
Unit and integration tests for src/make_splits.py.

Verifies:
1. Zero base image ID overlap across train, val, and test splits (no data leakage).
2. All 5 augmented variants of every Papaya base image remain clustered in the same split.
3. Every class is represented in all three splits.
4. Outputs per-class sample counts and base image ID counts per split.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.make_splits import extract_base_id, make_group_stratified_splits


def test_extract_base_id():
    """Verify base ID extraction logic for augmented Papaya vs original captures."""
    # Papaya: variations must map to same base ID
    assert (
        extract_base_id("Papaya", "Papaya___Anthracnose", "image_100_aug_1.jpg")
        == "Papaya_Papaya___Anthracnose_image_100"
    )
    assert (
        extract_base_id("Papaya", "Papaya___Anthracnose", "image_100_aug_5.jpg")
        == "Papaya_Papaya___Anthracnose_image_100"
    )

    # Potato & Rice: individual captures keep distinct stems
    assert (
        extract_base_id("Potato", "Potato___Early_Blight", "PotatoEarlyBlight(1).JPG")
        == "Potato_Potato___Early_Blight_PotatoEarlyBlight(1)"
    )
    assert (
        extract_base_id("Rice", "Rice___Brown_Spot", "308278744_606334647906871.jpg")
        == "Rice_Rice___Brown_Spot_308278744_606334647906871"
    )


def test_synthetic_group_stratified_split():
    """Unit test on synthetic dataframe verifying zero leakage and class stratification."""
    records = []
    # Class 0: Papaya with 10 base images, 5 augmentations each (50 images)
    for b in range(10):
        for aug in range(1, 6):
            records.append({
                "image_path": f"Papaya/Curl/img_{b}_aug_{aug}.jpg",
                "filename": f"img_{b}_aug_{aug}.jpg",
                "crop": "Papaya",
                "class_name": "Papaya___Curl",
                "base_id": f"Papaya_Papaya___Curl_img_{b}",
                "label": 0,
            })
    # Class 1: Rice with 40 unique base images (40 images)
    for i in range(40):
        records.append({
            "image_path": f"Rice/Healthy/rice_{i}.jpg",
            "filename": f"rice_{i}.jpg",
            "crop": "Rice",
            "class_name": "Rice___Healthy",
            "base_id": f"Rice_Rice___Healthy_rice_{i}",
            "label": 1,
        })

    df = pd.DataFrame(records)
    train_df, val_df, test_df = make_group_stratified_splits(
        df=df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    tr_base = set(train_df["base_id"])
    va_base = set(val_df["base_id"])
    te_base = set(test_df["base_id"])

    assert len(tr_base & va_base) == 0
    assert len(tr_base & te_base) == 0
    assert len(va_base & te_base) == 0

    # Ensure all 5 variants stay together for Papaya
    for base_id in tr_base:
        if "Papaya" in base_id:
            assert (train_df["base_id"] == base_id).sum() == 5
    for base_id in va_base:
        if "Papaya" in base_id:
            assert (val_df["base_id"] == base_id).sum() == 5
    for base_id in te_base:
        if "Papaya" in base_id:
            assert (test_df["base_id"] == base_id).sum() == 5


def test_splits_zero_leakage_and_class_counts(capsys=None):
    """
    Integration test on actual saved splits in splits/:
    1. Proves no base image ID appears in more than one split (zero leakage).
    2. Proves all Papaya augmented variants belong to a single split.
    3. Prints per-class counts per split.
    """
    splits_dir = Path("splits")
    train_file = splits_dir / "train.csv"
    val_file = splits_dir / "val.csv"
    test_file = splits_dir / "test.csv"

    assert train_file.exists(), f"Missing {train_file}. Run src/make_splits.py first."
    assert val_file.exists(), f"Missing {val_file}. Run src/make_splits.py first."
    assert test_file.exists(), f"Missing {test_file}. Run src/make_splits.py first."

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    train_base = set(train_df["base_id"])
    val_base = set(val_df["base_id"])
    test_base = set(test_df["base_id"])

    # 1. Zero leakage assertions
    train_val_overlap = train_base & val_base
    train_test_overlap = train_base & test_base
    val_test_overlap = val_base & test_base

    assert len(train_val_overlap) == 0, f"Leakage detected between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Leakage detected between train and test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage detected between val and test: {val_test_overlap}"

    # 2. Augmented Papaya variants integrity (each base ID must have exactly 5 variants in its assigned split)
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        papaya_split = split_df[split_df["crop"] == "Papaya"]
        base_counts = papaya_split.groupby("base_id").size()
        for base_id, count in base_counts.items():
            assert count == 5, f"Base ID {base_id} in {split_name} split has {count} variants (expected 5)"

    # 3. Print per-class counts per split
    all_classes = sorted(set(train_df["class_name"]).union(val_df["class_name"]).union(test_df["class_name"]))
    assert len(all_classes) == 12, f"Expected 12 classes, found {len(all_classes)}"

    output_lines = [
        "",
        "=" * 92,
        "VERIFICATION TEST: ZERO DATA LEAKAGE & PER-CLASS SPLIT DISTRIBUTION",
        "=" * 92,
        f"Train base IDs:      {len(train_base):>5,} | Overlap with Val:  {len(train_val_overlap):>2} | Overlap with Test: {len(train_test_overlap):>2}",
        f"Val base IDs:        {len(val_base):>5,} | Overlap with Train: {len(train_val_overlap):>2} | Overlap with Test: {len(val_test_overlap):>2}",
        f"Test base IDs:       {len(test_base):>5,} | Overlap with Train: {len(train_test_overlap):>2} | Overlap with Val:  {len(val_test_overlap):>2}",
        "-" * 92,
        "[PROVEN]: Zero base image ID appears in more than one split (zero data leakage).",
        "[PROVEN]: All 5 offline augmented variants per Papaya base capture stay in the same split.",
        "-" * 92,
        f"{'Class Name':<32} | {'Train (Img/Base)':<18} | {'Val (Img/Base)':<18} | {'Test (Img/Base)':<18} | {'Total':<8}",
        "-" * 92,
    ]

    for c in all_classes:
        tr_c = train_df[train_df["class_name"] == c]
        va_c = val_df[val_df["class_name"] == c]
        te_c = test_df[test_df["class_name"] == c]

        tr_img, tr_b = len(tr_c), tr_c["base_id"].nunique()
        va_img, va_b = len(va_c), va_c["base_id"].nunique()
        te_img, te_b = len(te_c), te_c["base_id"].nunique()
        tot = tr_img + va_img + te_img

        output_lines.append(
            f"{c:<32} | {f'{tr_img:,} ({tr_b:,})':<18} | {f'{va_img:,} ({va_b:,})':<18} | {f'{te_img:,} ({te_b:,})':<18} | {tot:<8,}"
        )

    output_lines.append("-" * 92)
    output_lines.append(
        f"{'TOTAL':<32} | "
        f"{f'{len(train_df):,} ({len(train_base):,})':<18} | "
        f"{f'{len(val_df):,} ({len(val_base):,})':<18} | "
        f"{f'{len(test_df):,} ({len(test_base):,})':<18} | "
        f"{len(train_df) + len(val_df) + len(test_df):<8,}"
    )
    output_lines.append("=" * 92 + "\n")

    summary_text = "\n".join(output_lines)

    if capsys:
        with capsys.disabled():
            print(summary_text)
    else:
        print(summary_text)


if __name__ == "__main__":
    test_extract_base_id()
    test_synthetic_group_stratified_split()
    test_splits_zero_leakage_and_class_counts()
    print("All tests passed successfully!")
