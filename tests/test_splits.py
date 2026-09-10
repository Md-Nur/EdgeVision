import pytest
import pandas as pd
from pathlib import Path
from src.data.splitter import extract_base_id, prepare_group_stratified_splits


def test_extract_base_id_papaya_aug():
    assert extract_base_id("Papaya", "Anthracnose", "image_100_aug_1.jpg") == "Papaya_Anthracnose_image_100"
    assert extract_base_id("Papaya", "Anthracnose", "image_100_aug_5.jpg") == "Papaya_Anthracnose_image_100"


def test_extract_base_id_rice_potato():
    assert extract_base_id("Rice", "Brown Spot", "rice_01.jpg") == "Rice_Brown Spot_rice_01"
    assert extract_base_id("Potato", "Potato Early blight", "potato_05.jpg") == "Potato_Potato Early blight_potato_05"


def test_prepare_group_stratified_splits(tmp_path):
    # Create synthetic dataset with augmented and non-augmented data
    rows = []
    # Class 0 (simulating Papaya augmented 5x)
    for base in range(20):
        for aug in range(1, 6):
            rows.append({
                "image_path": f"Papaya/Curl/img_{base}_aug_{aug}.jpg",
                "filename": f"img_{base}_aug_{aug}.jpg",
                "crop": "Papaya",
                "disease": "Curl",
                "class_name": "Papaya_Curl",
                "label": 0,
            })
    # Class 1 (simulating Rice, single captures)
    for i in range(30):
        rows.append({
            "image_path": f"Rice/Healthy/img_{i}.jpg",
            "filename": f"img_{i}.jpg",
            "crop": "Rice",
            "disease": "Healthy",
            "class_name": "Rice_Healthy",
            "label": 1,
        })

    df = pd.DataFrame(rows)
    csv_file = tmp_path / "test_dataset.csv"
    df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "splits"
    train_df, val_df, test_df = prepare_group_stratified_splits(
        csv_path=csv_file,
        output_dir=out_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # 1. Zero leakage check
    train_base = set(train_df["base_id"])
    val_base = set(val_df["base_id"])
    test_base = set(test_df["base_id"])

    assert len(train_base & val_base) == 0
    assert len(train_base & test_base) == 0
    assert len(val_base & test_base) == 0

    # 2. All 5 augmented variants stay together
    for base_id in train_base:
        if "Papaya" in base_id:
            assert (train_df["base_id"] == base_id).sum() == 5
    for base_id in val_base:
        if "Papaya" in base_id:
            assert (val_df["base_id"] == base_id).sum() == 5
    for base_id in test_base:
        if "Papaya" in base_id:
            assert (test_df["base_id"] == base_id).sum() == 5

    # 3. Both classes exist in all splits
    for split_df in [train_df, val_df, test_df]:
        assert set(split_df["class_name"]) == {"Papaya_Curl", "Rice_Healthy"}
