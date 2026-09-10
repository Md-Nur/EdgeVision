import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
from src.data.splitter import prepare_group_stratified_splits
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Prepare leakage-free stratified group splits.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    csv_path = data_cfg["csv_path"]
    splits_dir = data_cfg["splits_dir"]
    train_split = data_cfg.get("train_split", 0.70)
    val_split = data_cfg.get("val_split", 0.15)
    test_split = data_cfg.get("test_split", 0.15)
    seed = data_cfg.get("random_state", 42)

    print(f"Loading data from: {csv_path}")
    train_df, val_df, test_df = prepare_group_stratified_splits(
        csv_path=csv_path,
        output_dir=splits_dir,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        seed=seed,
    )

    print("\n=======================================================")
    print("LEAKAGE-FREE GROUP STRATIFIED SPLITS CREATED")
    print("=======================================================")
    print(f"Train samples: {len(train_df)} | Unique base images: {train_df['base_id'].nunique()}")
    print(f"Val samples:   {len(val_df)} | Unique base images: {val_df['base_id'].nunique()}")
    print(f"Test samples:  {len(test_df)} | Unique base images: {test_df['base_id'].nunique()}")
    print(f"Total:         {len(train_df) + len(val_df) + len(test_df)}")

    # Verification of zero leakage
    train_base = set(train_df["base_id"])
    val_base = set(val_df["base_id"])
    test_base = set(test_df["base_id"])

    assert len(train_base & val_base) == 0, "Train and Val share base images!"
    assert len(train_base & test_base) == 0, "Train and Test share base images!"
    assert len(val_base & test_base) == 0, "Val and Test share base images!"
    print("\n[VERIFICATION PASSED]: Zero base_id leakage detected across train, val, and test splits.")

    # Class distribution summary
    print("\nPer-Class Distribution Across Splits (Total images / Base images):")
    classes = sorted(train_df["class_name"].unique())
    summary = []
    for c in classes:
        tr_total = (train_df["class_name"] == c).sum()
        tr_base = train_df[train_df["class_name"] == c]["base_id"].nunique()
        va_total = (val_df["class_name"] == c).sum()
        va_base = val_df[val_df["class_name"] == c]["base_id"].nunique()
        te_total = (test_df["class_name"] == c).sum()
        te_base = test_df[test_df["class_name"] == c]["base_id"].nunique()
        summary.append({
            "Class Name": c,
            "Train Total (Base)": f"{tr_total} ({tr_base})",
            "Val Total (Base)": f"{va_total} ({va_base})",
            "Test Total (Base)": f"{te_total} ({te_base})",
            "Total Base": tr_base + va_base + te_base
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print(f"\nSplits saved to: {Path(splits_dir).resolve()}")


if __name__ == "__main__":
    main()
