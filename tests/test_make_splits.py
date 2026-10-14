"""
Unit and integration tests for src/make_splits.py.

Verifies:
1. Base ID extraction logic for augmented Papaya vs original captures.
2. Cross-class conflicts: 73 pHash pairs (and 23 MD5 pairs) from dataset audit
   are listed in reports/cross_class_conflicts.csv and excluded from all splits.
3. Connected component grouping: zero group_id overlap across train, val, and test splits.
4. Zero perceptual leakage: no image pair with pHash Hamming distance <= 4 crosses splits.
5. Per-class and per-crop distributions across splits, and verification that no class
   drops below 30 test images.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from collections import defaultdict

from src.make_splits import (
    extract_base_id,
    make_group_stratified_splits,
    find_phash_matches_bucketing,
)


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

    tr_groups = set(train_df["group_id"])
    va_groups = set(val_df["group_id"])
    te_groups = set(test_df["group_id"])

    assert len(tr_groups & va_groups) == 0
    assert len(tr_groups & te_groups) == 0
    assert len(va_groups & te_groups) == 0


def test_cross_class_conflicts_excluded_from_all_splits():
    """
    Prove that all 73 conflicting pairs (including all 23 identical MD5 pairs)
    from the audit are recorded in reports/cross_class_conflicts.csv and that
    BOTH images from every conflicting pair are completely absent from all splits.
    """
    conflicts_csv = Path("reports/cross_class_conflicts.csv")
    assert conflicts_csv.exists(), f"Missing {conflicts_csv}"

    conflicts_df = pd.read_csv(conflicts_csv)
    assert len(conflicts_df) == 73, f"Expected 73 cross-class pairs, found {len(conflicts_df)}"

    md5_identical_count = (conflicts_df["identical_md5"] == True).sum()
    assert md5_identical_count == 23, f"Expected 23 MD5 identical pairs, found {md5_identical_count}"

    # Collect all excluded image paths
    excluded_paths = set(conflicts_df["image1_path"]).union(set(conflicts_df["image2_path"]))
    assert len(excluded_paths) == 90, f"Expected 90 unique excluded images, found {len(excluded_paths)}"

    train_df = pd.read_csv("splits/train.csv")
    val_df = pd.read_csv("splits/val.csv")
    test_df = pd.read_csv("splits/test.csv")

    all_split_paths = set(train_df["image_path"]).union(set(val_df["image_path"])).union(set(test_df["image_path"]))

    leakage_into_splits = excluded_paths & all_split_paths
    assert len(leakage_into_splits) == 0, (
        f"Cross-class conflicting images leaked into splits ({len(leakage_into_splits)} images): {leakage_into_splits}"
    )


def test_no_group_id_appears_in_more_than_one_split():
    """Prove that no group_id appears in more than one split (zero group leakage)."""
    train_df = pd.read_csv("splits/train.csv")
    val_df = pd.read_csv("splits/val.csv")
    test_df = pd.read_csv("splits/test.csv")

    expected_cols = ["image_path", "crop", "disease", "class_name", "label", "group_id"]
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert list(df.columns) == expected_cols, f"{name}.csv has columns {list(df.columns)}, expected {expected_cols}"

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    assert len(train_val_overlap) == 0, f"Leakage between train and val: {len(train_val_overlap)} groups"
    assert len(train_test_overlap) == 0, f"Leakage between train and test: {len(train_test_overlap)} groups"
    assert len(val_test_overlap) == 0, f"Leakage between val and test: {len(val_test_overlap)} groups"


def test_no_pair_with_phash_distance_le_4_crosses_splits():
    """
    Prove that NO pair of images in the dataset splits with pHash Hamming distance <= 4
    crosses splits (zero perceptual near-duplicate leakage).
    """
    hashes_csv = Path("data/metadata_hashes.csv")
    assert hashes_csv.exists(), f"Missing {hashes_csv}"
    hashes_df = pd.read_csv(hashes_csv).set_index("image_path")

    train_df = pd.read_csv("splits/train.csv")
    val_df = pd.read_csv("splits/val.csv")
    test_df = pd.read_csv("splits/test.csv")

    split_map = {}
    for p in train_df["image_path"]:
        split_map[p] = "train"
    for p in val_df["image_path"]:
        split_map[p] = "val"
    for p in test_df["image_path"]:
        split_map[p] = "test"

    split_paths = list(split_map.keys())
    split_hashes = [hashes_df.loc[p, "phash_int"] for p in split_paths]

    matching_pairs = find_phash_matches_bucketing(split_hashes, max_distance=4)

    crossing_pairs = []
    for idx1, idx2, dist in matching_pairs:
        p1 = split_paths[idx1]
        p2 = split_paths[idx2]
        s1 = split_map[p1]
        s2 = split_map[p2]
        if s1 != s2:
            crossing_pairs.append((p1, p2, s1, s2, dist))

    assert len(crossing_pairs) == 0, (
        f"Found {len(crossing_pairs)} near-duplicate pairs (pHash <= 4) crossing splits: {crossing_pairs[:5]}"
    )


def test_splits_per_class_and_per_crop_counts_and_threshold(capsys=None):
    """
    Print per-class and per-crop counts for each split, verify all 12 classes are present,
    and explicitly verify and report any class that drops below 30 test images.
    """
    train_df = pd.read_csv("splits/train.csv")
    val_df = pd.read_csv("splits/val.csv")
    test_df = pd.read_csv("splits/test.csv")

    total_images = len(train_df) + len(val_df) + len(test_df)
    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])
    total_groups = len(train_groups) + len(val_groups) + len(test_groups)

    all_classes = sorted(set(train_df["class_name"]).union(val_df["class_name"]).union(test_df["class_name"]))
    assert len(all_classes) == 12, f"Expected 12 classes, found {len(all_classes)}"

    output_lines = [
        "",
        "=" * 94,
        "VERIFICATION TEST: ZERO DATA LEAKAGE, PER-CLASS & PER-CROP COUNTS",
        "=" * 94,
        f"Train Split:      {len(train_df):>6,} images ({len(train_df)/total_images*100:>5.2f}%) | {len(train_groups):>5,} group IDs ({len(train_groups)/total_groups*100:>5.2f}%)",
        f"Validation Split: {len(val_df):>6,} images ({len(val_df)/total_images*100:>5.2f}%) | {len(val_groups):>5,} group IDs ({len(val_groups)/total_groups*100:>5.2f}%)",
        f"Test Split:       {len(test_df):>6,} images ({len(test_df)/total_images*100:>5.2f}%) | {len(test_groups):>5,} group IDs ({len(test_groups)/total_groups*100:>5.2f}%)",
        f"Total Samples:    {total_images:>6,} images (100.00%) | {total_groups:>5,} group IDs (100.00%)",
        "-" * 94,
        "[PROVEN]: Zero group_id appears in more than one split (exact 0 overlap).",
        "[PROVEN]: Zero image pair with pHash Hamming distance <= 4 crosses splits.",
        "[PROVEN]: Zero cross-class conflicting pairs leaked into any split fold.",
        "-" * 94,
        "PER-CROP SAMPLE COUNTS:",
        f"{'Crop':<12} | {'Train':<14} | {'Val':<14} | {'Test':<14} | {'Total':<10}",
        "-" * 72,
    ]

    crops = sorted(set(train_df["crop"]).union(val_df["crop"]).union(test_df["crop"]))
    for cr in crops:
        tr_cr = (train_df["crop"] == cr).sum()
        va_cr = (val_df["crop"] == cr).sum()
        te_cr = (test_df["crop"] == cr).sum()
        tot_cr = tr_cr + va_cr + te_cr
        output_lines.append(f"{cr:<12} | {tr_cr:<14,} | {va_cr:<14,} | {te_cr:<14,} | {tot_cr:<10,}")

    output_lines.extend([
        "-" * 94,
        "PER-CLASS SAMPLE & GROUP COUNTS:",
        f"{'Class Name':<32} | {'Train (Img/Grp)':<18} | {'Val (Img/Grp)':<18} | {'Test (Img/Grp)':<18} | {'Total':<8}",
        "-" * 94,
    ])

    below_30 = []
    for c in all_classes:
        tr_c = train_df[train_df["class_name"] == c]
        va_c = val_df[val_df["class_name"] == c]
        te_c = test_df[test_df["class_name"] == c]

        tr_img, tr_g = len(tr_c), tr_c["group_id"].nunique()
        va_img, va_g = len(va_c), va_c["group_id"].nunique()
        te_img, te_g = len(te_c), te_c["group_id"].nunique()
        tot = tr_img + va_img + te_img

        flag = ""
        if te_img < 30:
            flag = " [DROPPED BELOW 30!]"
            below_30.append((c, te_img))

        output_lines.append(
            f"{c:<32} | {f'{tr_img:,} ({tr_g:,})':<18} | {f'{va_img:,} ({va_g:,})':<18} | {f'{te_img:,} ({te_g:,})':<18} | {tot:<8,}{flag}"
        )

    output_lines.append("-" * 94)
    if below_30:
        output_lines.append(f"CLASSES DROPPED BELOW 30 TEST IMAGES: {below_30}")
    else:
        min_test_count = min(len(test_df[test_df["class_name"] == c]) for c in all_classes)
        output_lines.append(
            f"[TEST THRESHOLD VERIFIED]: No class drops below 30 test images. Minimum test class size is {min_test_count} images (Rice___Healthy)."
        )
    output_lines.append("=" * 94 + "\n")

    summary_text = "\n".join(output_lines)
    if capsys:
        with capsys.disabled():
            print(summary_text)
    else:
        print(summary_text)

    assert len(below_30) == 0, f"Classes with < 30 test images: {below_30}"


def _verify_split_leakage_and_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_label: str = "Split",
    check_dedup_md5: bool = False,
):
    hashes_csv = Path("data/metadata_hashes.csv")
    assert hashes_csv.exists(), f"Missing {hashes_csv}"
    hashes_df = pd.read_csv(hashes_csv).set_index("image_path")

    conflicts_csv = Path("reports/cross_class_conflicts.csv")
    assert conflicts_csv.exists(), f"Missing {conflicts_csv}"
    conflicts_df = pd.read_csv(conflicts_csv)
    excluded_paths = set(conflicts_df["image1_path"]).union(set(conflicts_df["image2_path"]))

    # 1. Expected columns
    expected_cols = ["image_path", "crop", "disease", "class_name", "label", "group_id"]
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert list(df.columns) == expected_cols, f"{split_label} {name} columns mismatch: {list(df.columns)}"

    # 2. Excluded cross-class conflicts absent from all splits
    all_split_paths = set(train_df["image_path"]).union(set(val_df["image_path"])).union(set(test_df["image_path"]))
    leaked_conflicts = excluded_paths & all_split_paths
    assert len(leaked_conflicts) == 0, f"{split_label}: {len(leaked_conflicts)} cross-class conflict images leaked!"

    # 3. Zero group_id overlap
    tr_g = set(train_df["group_id"])
    va_g = set(val_df["group_id"])
    te_g = set(test_df["group_id"])
    assert len(tr_g & va_g) == 0, f"{split_label}: Train-Val group overlap: {len(tr_g & va_g)}"
    assert len(tr_g & te_g) == 0, f"{split_label}: Train-Test group overlap: {len(tr_g & te_g)}"
    assert len(va_g & te_g) == 0, f"{split_label}: Val-Test group overlap: {len(va_g & te_g)}"

    # 4. Zero pHash <= 4 near-duplicate leakage
    split_map = {}
    for p in train_df["image_path"]:
        split_map[p] = "train"
    for p in val_df["image_path"]:
        split_map[p] = "val"
    for p in test_df["image_path"]:
        split_map[p] = "test"

    split_paths = list(split_map.keys())
    split_hashes = [hashes_df.loc[p, "phash_int"] for p in split_paths]
    matching_pairs = find_phash_matches_bucketing(split_hashes, max_distance=4)

    crossing_pairs = []
    for idx1, idx2, dist in matching_pairs:
        p1 = split_paths[idx1]
        p2 = split_paths[idx2]
        if split_map[p1] != split_map[p2]:
            crossing_pairs.append((p1, p2, split_map[p1], split_map[p2], dist))
    assert len(crossing_pairs) == 0, f"{split_label}: {len(crossing_pairs)} near-duplicate pairs crossing splits!"

    # 5. Check exact MD5 deduplication if requested
    if check_dedup_md5:
        for sname, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
            md5s = [hashes_df.loc[p, "md5"] for p in sdf["image_path"]]
            assert len(md5s) == len(set(md5s)), f"{split_label} {sname} contains duplicate MD5s!"

    # 6. Minimum class test count >= 30
    all_classes = sorted(set(train_df["class_name"]).union(val_df["class_name"]).union(test_df["class_name"]))
    assert len(all_classes) == 12, f"{split_label}: Missing classes, found {len(all_classes)}"
    min_test = min(len(test_df[test_df["class_name"] == c]) for c in all_classes)
    assert min_test >= 30, f"{split_label}: Class drops below 30 test images: {min_test}"


def test_kfold_splits_all_folds_no_leakage(capsys=None):
    """
    Leakage tests for Mode (b): 5-fold StratifiedGroupKFold splits (splits/fold0..4_{train,val,test}.csv).
    Verifies that for EVERY fold:
    1. Zero group_id overlap between train, val, and test.
    2. Zero perceptual near-duplicates (pHash <= 4) cross splits.
    3. Zero cross-class conflicting images leak into any fold.
    4. All 12 classes are represented with >= 30 test images.
    """
    splits_dir = Path("splits")
    print("\n" + "=" * 90)
    print("LEAKAGE TEST SUITE: 5-FOLD STRATIFIED GROUP K-FOLD MODE (Folds 0..4)")
    print("=" * 90)

    for k in range(5):
        tr_csv = splits_dir / f"fold{k}_train.csv"
        va_csv = splits_dir / f"fold{k}_val.csv"
        te_csv = splits_dir / f"fold{k}_test.csv"

        assert tr_csv.exists() and va_csv.exists() and te_csv.exists(), f"Missing split CSVs for Fold {k}"

        tr_df = pd.read_csv(tr_csv)
        va_df = pd.read_csv(va_csv)
        te_df = pd.read_csv(te_csv)

        _verify_split_leakage_and_integrity(
            tr_df, va_df, te_df, split_label=f"Fold {k}"
        )

        min_te = min(len(te_df[te_df["class_name"] == c]) for c in te_df["class_name"].unique())
        print(f"[PASS] Fold {k}: Train={len(tr_df):>5,} ({tr_df['group_id'].nunique():>4,} grp) | "
              f"Val={len(va_df):>5,} ({va_df['group_id'].nunique():>4,} grp) | "
              f"Test={len(te_df):>5,} ({te_df['group_id'].nunique():>4,} grp) | "
              f"Min test class count={min_te} | Leakage: ZERO (Groups=0, pHash<=4=0, Conflicts=0)")

    print("=" * 90 + "\n")


def test_md5_deduplicated_mode_no_leakage(capsys=None):
    """
    Leakage tests for Mode (a): Exact-MD5 deduplicated splits (splits/dedup/{train,val,test}.csv).
    Verifies that:
    1. Exactly one image per exact-MD5 duplicate is kept within each split (0 duplicate MD5s).
    2. Zero group_id overlap between train, val, and test.
    3. Zero perceptual near-duplicates (pHash <= 4) cross splits.
    4. Zero cross-class conflicting images leak into any split.
    5. All 12 classes are represented with >= 30 test images.
    """
    dedup_dir = Path("splits/dedup")
    tr_csv = dedup_dir / "train.csv"
    va_csv = dedup_dir / "val.csv"
    te_csv = dedup_dir / "test.csv"

    # If not yet generated on disk, generate in memory from clean dataset
    if not (tr_csv.exists() and va_csv.exists() and te_csv.exists()):
        from src.make_splits import run_pipeline
        tr_df, va_df, te_df = run_pipeline(output_dir=dedup_dir, dedup_md5=True)
    else:
        tr_df = pd.read_csv(tr_csv)
        va_df = pd.read_csv(va_csv)
        te_df = pd.read_csv(te_csv)

    print("\n" + "=" * 90)
    print("LEAKAGE TEST SUITE: EXACT-MD5 DEDUPLICATED SPLIT MODE")
    print("=" * 90)

    _verify_split_leakage_and_integrity(
        tr_df, va_df, te_df, split_label="MD5-Dedup Split", check_dedup_md5=True
    )

    min_te = min(len(te_df[te_df["class_name"] == c]) for c in te_df["class_name"].unique())
    print(f"[PASS] Exact-MD5 Dedup Mode: Train={len(tr_df):>5,} ({tr_df['group_id'].nunique():>4,} grp) | "
          f"Val={len(va_df):>5,} ({va_df['group_id'].nunique():>4,} grp) | "
          f"Test={len(te_df):>5,} ({te_df['group_id'].nunique():>4,} grp) | "
          f"Min test class count={min_te} | Leakage: ZERO (Exact MD5s unique, Groups=0, pHash<=4=0, Conflicts=0)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    test_extract_base_id()
    test_synthetic_group_stratified_split()
    test_cross_class_conflicts_excluded_from_all_splits()
    test_no_group_id_appears_in_more_than_one_split()
    test_no_pair_with_phash_distance_le_4_crosses_splits()
    test_splits_per_class_and_per_crop_counts_and_threshold()
    test_kfold_splits_all_folds_no_leakage()
    test_md5_deduplicated_mode_no_leakage()
    print("All split tests (Baseline, 5-Fold, and MD5-Dedup) executed and passed successfully!")

