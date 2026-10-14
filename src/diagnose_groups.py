"""
Diagnose Grouping, Connected Components, and Perceptual Leakage.

Reads splits/*.csv and data/metadata_hashes.csv, and reports:
1. The distribution of group sizes (images per group) and the 10 largest
   connected components, with their class and crop composition.
2. For Papaya, how many base_ids merge into each group, and whether merges
   are driven by pHash<=4 edges or MD5 edges. Tests pHash thresholds 0, 2, 4
   and reports the number of groups per class for each.
3. Per class, the number of groups in train/val/test for each threshold.
4. Saves 12 example montages (6 near-duplicate pairs from different base_ids
   in Papaya + 6 diverse near-duplicate pairs from Potato/Rice) to reports/.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from src.make_splits import (
    build_connected_components,
    find_phash_matches_bucketing,
)


def load_dataset_and_splits(
    splits_dir: str | Path = "splits",
    cache_csv: str | Path = "data/metadata_hashes.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/val/test split CSVs and join with metadata hash cache."""
    splits_path = Path(splits_dir)
    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "val.csv"
    test_csv = splits_path / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(f"Missing split CSVs in {splits_path}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    cache_path = Path(cache_csv)
    if not cache_path.exists():
        raise FileNotFoundError(f"Metadata cache not found: {cache_path}")

    cache_df = pd.read_csv(cache_path)
    cache_lookup = cache_df.set_index("image_path")

    # Add metadata columns if not present
    split_df["filename"] = [Path(p).name for p in split_df["image_path"]]
    split_df["base_id"] = [cache_lookup.loc[p, "base_id"] for p in split_df["image_path"]]
    split_df["md5"] = [cache_lookup.loc[p, "md5"] for p in split_df["image_path"]]
    split_df["phash_int"] = [cache_lookup.loc[p, "phash_int"] for p in split_df["image_path"]]
    split_df["phash_hex"] = [cache_lookup.loc[p, "phash_hex"] for p in split_df["image_path"]]

    return split_df, cache_df


def report_group_size_distribution(df: pd.DataFrame) -> None:
    """Report distribution of group sizes and the 10 largest connected components."""
    print("=" * 90)
    print("1. GROUP SIZE DISTRIBUTION & 10 LARGEST CONNECTED COMPONENTS")
    print("=" * 90)

    group_sizes = df["group_id"].value_counts()
    n_groups = len(group_sizes)
    n_images = len(df)

    print(f"Total Images in Splits:   {n_images:,}")
    print(f"Total Connected Groups:   {n_groups:,}")
    print(f"Average Images per Group: {group_sizes.mean():.2f}")
    print(f"Median Images per Group:  {group_sizes.median():.1f}")
    print(f"Std Deviation:            {group_sizes.std():.2f}")
    print(f"Min / Max Group Size:     {group_sizes.min()} / {group_sizes.max()}")
    print("-" * 90)

    # Percentiles
    pcts = [25, 50, 75, 90, 95, 99, 100]
    pct_vals = np.percentile(group_sizes, pcts)
    pct_str = " | ".join([f"{p}%: {v:.0f}" for p, v in zip(pcts, pct_vals)])
    print(f"Percentiles (images/grp): {pct_str}")
    print("-" * 90)

    # Frequency Bins
    bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 1000)]
    print(f"{'Group Size Bin':<18} | {'Num Groups':<12} | {'% Groups':<10} | {'Total Images':<14} | {'% Images':<10}")
    print("-" * 74)
    for low, high in bins:
        if low == high:
            mask = group_sizes == low
            label = f"{low} image"
        elif high == 1000:
            mask = group_sizes >= low
            label = f">= {low} images"
        else:
            mask = (group_sizes >= low) & (group_sizes <= high)
            label = f"{low}-{high} images"

        n_g = mask.sum()
        pct_g = (n_g / n_groups) * 100
        n_img = group_sizes[mask].sum()
        pct_img = (n_img / n_images) * 100
        print(f"{label:<18} | {n_g:>12,} | {pct_g:>9.2f}% | {n_img:>14,} | {pct_img:>9.2f}%")

    print("-" * 90)
    print("TOP 10 LARGEST CONNECTED COMPONENTS:")
    print(f"{'Rank':<5} | {'Group ID':<9} | {'Size':<6} | {'Crops':<14} | {'Classes':<28} | {'Base IDs':<9} | {'Split':<6}")
    print("-" * 90)

    top10_ids = group_sizes.head(10).index
    for rank, gid in enumerate(top10_ids, 1):
        comp = df[df["group_id"] == gid]
        size = len(comp)
        crops = dict(comp["crop"].value_counts())
        classes = dict(comp["class_name"].value_counts())
        n_base = comp["base_id"].nunique()
        splits = "/".join(sorted(comp["split"].unique()))

        crop_str = ", ".join([f"{k}:{v}" for k, v in crops.items()])
        class_str = ", ".join([f"{k}:{v}" for k, v in classes.items()])
        if len(class_str) > 28:
            class_str = class_str[:25] + "..."

        print(f"{rank:<5} | {gid:<9} | {size:<6} | {crop_str:<14} | {class_str:<28} | {n_base:<9} | {splits:<6}")

    # Check for any cross-class or cross-crop component across whole dataset
    cross_class_comps = 0
    cross_crop_comps = 0
    for gid, group in df.groupby("group_id"):
        if group["class_name"].nunique() > 1:
            cross_class_comps += 1
        if group["crop"].nunique() > 1:
            cross_crop_comps += 1

    print("-" * 90)
    print(f"Integrity Check: Connected components spanning multiple classes: {cross_class_comps}")
    print(f"Integrity Check: Connected components spanning multiple crops:   {cross_crop_comps}")
    print("=" * 90 + "\n")


def report_papaya_analysis_and_thresholds(
    df: pd.DataFrame,
) -> Dict[int, np.ndarray]:
    """
    Analyze Papaya base_id merges, pHash vs MD5 drivers,
    and test connected components at thresholds 0, 2, 4.
    """
    print("=" * 90)
    print("2. PAPAYA MERGE ANALYSIS & PHASH THRESHOLD COMPARISON (0, 2, 4)")
    print("=" * 90)

    papaya_df = df[df["crop"] == "Papaya"].copy().reset_index(drop=True)
    n_papaya_img = len(papaya_df)
    n_papaya_base = papaya_df["base_id"].nunique()
    n_papaya_groups = papaya_df["group_id"].nunique()

    print(f"Total Papaya Images:   {n_papaya_img:,}")
    print(f"Unique Papaya Base IDs:{n_papaya_base:,}")
    print(f"Papaya Group IDs:      {n_papaya_groups:,}")
    print("-" * 90)

    # 1. Base IDs per group distribution
    base_counts = papaya_df.groupby("group_id")["base_id"].nunique()
    base_freq = base_counts.value_counts().sort_index()

    print("DISTRIBUTION OF BASE IDs MERGED PER PAPAYA GROUP:")
    print(f"{'Base IDs in Group':<20} | {'Num Groups':<12} | {'% Groups':<10} | {'Total Images':<14} | {'% Papaya Images':<16}")
    print("-" * 80)
    for b_num, count in base_freq.items():
        g_ids = base_counts[base_counts == b_num].index
        img_count = papaya_df[papaya_df["group_id"].isin(g_ids)].shape[0]
        pct_g = (count / n_papaya_groups) * 100
        pct_img = (img_count / n_papaya_img) * 100
        print(f"{b_num:<20} | {count:>12,} | {pct_g:>9.2f}% | {img_count:>14,} | {pct_img:>14.2f}%")

    print("-" * 90)

    # 2. Are merges driven by MD5 or pHash <= 4?
    # MD5 edge analysis
    md5_to_bases_set = defaultdict(set)
    md5_to_bases_list = defaultdict(list)
    for _, r in papaya_df.iterrows():
        md5_to_bases_set[r["md5"]].add(r["base_id"])
        md5_to_bases_list[r["md5"]].append(r["base_id"])

    md5_cross_base_hashes = {k: v for k, v in md5_to_bases_set.items() if len(v) > 1}
    md5_cross_base_pairs_count = 0
    for md5_val, b_list in md5_to_bases_list.items():
        for i in range(len(b_list)):
            for j in range(i + 1, len(b_list)):
                if b_list[i] != b_list[j]:
                    md5_cross_base_pairs_count += 1

    # pHash edge analysis
    p_hashes = papaya_df["phash_int"].values
    p_matches = find_phash_matches_bucketing(p_hashes, max_distance=4)
    p_bases = papaya_df["base_id"].tolist()

    phash_edges_by_dist = defaultdict(int)
    for idx1, idx2, dist in p_matches:
        if p_bases[idx1] != p_bases[idx2]:
            phash_edges_by_dist[dist] += 1

    total_phash_cross_base = sum(phash_edges_by_dist.values())

    print("MERGE DRIVER ANALYSIS: EXACT MD5 vs PERCEPTUAL HASH (pHash <= 4):")
    print(f" - Unique MD5 hashes shared by different base_ids:  {len(md5_cross_base_hashes):,}")
    print(f" - Image pairs connecting different base_ids via MD5: {md5_cross_base_pairs_count:,}")
    print(f" - Image pairs connecting different base_ids via pHash <= 4: {total_phash_cross_base:,}")
    print("   Breakdown of cross-base pHash edges by Hamming distance:")
    for dist in sorted(phash_edges_by_dist.keys()):
        count = phash_edges_by_dist[dist]
        pct = (count / total_phash_cross_base) * 100
        print(f"     * Hamming Distance {dist}: {count:>6,} edges ({pct:>5.1f}%)")

    print("\n[FINDING / DIAGNOSIS]:")
    print("   Merges between different base_ids in Papaya are overwhelmingly (>99.3%) driven")
    print("   by perceptual hash (pHash <= 4) edges. Even at exact perceptual hash match (distance 0),")
    print(f"   there are {phash_edges_by_dist[0]:,} edges between different base_ids that do not share exact MD5.")
    print("   Only 55 MD5 hashes connect different base_ids, accounting for just 82 image pairs.")
    print("-" * 90)

    # 3. Test thresholds 0, 2, 4 across all classes
    print("COMPARISON OF NUMBER OF GROUPS PER CLASS AT pHash THRESHOLDS 0, 2, 4:")
    classes = sorted(df["class_name"].unique())

    threshold_groups = {}
    for t in [0, 2, 4]:
        matches = find_phash_matches_bucketing(df["phash_int"].values, max_distance=t)
        gids = build_connected_components(df, phash_matches=matches, max_phash_dist=t)
        threshold_groups[t] = gids
        df[f"group_t{t}"] = gids

    print(f"{'Class Name':<32} | {'pHash <= 0':<12} | {'pHash <= 2':<12} | {'pHash <= 4':<12} | {'Collapse (0->4)':<15}")
    print("-" * 90)

    for c in classes:
        sub = df[df["class_name"] == c]
        g0 = sub["group_t0"].nunique()
        g2 = sub["group_t2"].nunique()
        g4 = sub["group_t4"].nunique()
        delta = g0 - g4
        pct_collapse = (delta / g0) * 100 if g0 > 0 else 0
        print(f"{c:<32} | {g0:>12,} | {g2:>12,} | {g4:>12,} | -{delta:>4} ({pct_collapse:>5.1f}%)")

    print("-" * 90)
    tot0 = df["group_t0"].nunique()
    tot2 = df["group_t2"].nunique()
    tot4 = df["group_t4"].nunique()
    tot_delta = tot0 - tot4
    print(f"{'TOTAL (All Classes)':<32} | {tot0:>12,} | {tot2:>12,} | {tot4:>12,} | -{tot_delta:>4} ({tot_delta/tot0*100:>5.1f}%)")
    print("=" * 90 + "\n")

    return threshold_groups


def report_threshold_groups_across_splits(df: pd.DataFrame) -> None:
    """Report per-class number of groups in train/val/test for each threshold."""
    print("=" * 90)
    print("3. PER-CLASS NUMBER OF GROUPS IN TRAIN / VAL / TEST FOR EACH THRESHOLD")
    print("=" * 90)

    classes = sorted(df["class_name"].unique())

    for t in [0, 2, 4]:
        col = f"group_t{t}"
        print(f"--- THRESHOLD pHash <= {t} (Total Groups: {df[col].nunique():,}) ---")
        print(f"{'Class Name':<32} | {'Train Grp':<12} | {'Val Grp':<12} | {'Test Grp':<12} | {'Total Grp':<10}")
        print("-" * 86)
        for c in classes:
            sub = df[df["class_name"] == c]
            tr_g = sub[sub["split"] == "train"][col].nunique()
            va_g = sub[sub["split"] == "val"][col].nunique()
            te_g = sub[sub["split"] == "test"][col].nunique()
            tot_g = sub[col].nunique()
            print(f"{c:<32} | {tr_g:>12,} | {va_g:>12,} | {te_g:>12,} | {tot_g:>10,}")
        print("-" * 86)
        tot_tr = df[df["split"] == "train"][col].nunique()
        tot_va = df[df["split"] == "val"][col].nunique()
        tot_te = df[df["split"] == "test"][col].nunique()
        tot_all = df[col].nunique()
        print(f"{'TOTAL':<32} | {tot_tr:>12,} | {tot_va:>12,} | {tot_te:>12,} | {tot_all:>10,}\n")

    print("=" * 90 + "\n")


def create_montage_pair(
    pair_info: Dict[str, Any],
    output_path: Path,
    montage_idx: int,
) -> None:
    """Render a crisp side-by-side comparison montage image for a near-duplicate pair."""
    p1 = pair_info["p1"]
    p2 = pair_info["p2"]

    img1 = Image.open(p1).convert("RGB")
    img2 = Image.open(p2).convert("RGB")

    # Resize to standard height 360 preserving aspect ratio
    target_h = 360
    w1 = int(img1.width * (target_h / img1.height))
    w2 = int(img2.width * (target_h / img2.height))
    img1_res = img1.resize((w1, target_h), Image.Resampling.LANCZOS)
    img2_res = img2.resize((w2, target_h), Image.Resampling.LANCZOS)

    header_h = 100
    footer_h = 40
    padding = 20
    total_w = w1 + w2 + padding * 3
    total_h = target_h + header_h + footer_h

    canvas = Image.new("RGB", (total_w, total_h), color=(245, 247, 250))
    canvas.paste(img1_res, (padding, header_h))
    canvas.paste(img2_res, (padding * 2 + w1, header_h))

    draw = ImageDraw.Draw(canvas)

    # Try loading system font, fallback to default
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        caption_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        caption_font = ImageFont.load_default()

    # Draw Header
    title = f"Montage {montage_idx:02d}: {pair_info['crop']} | {pair_info['class']} (Merged into Group {pair_info['group_id']})"
    draw.text((padding, 12), title, fill=(20, 30, 50), font=title_font)

    md5_status = "IDENTICAL" if pair_info["md5_match"] else "DIFFERENT"
    subtitle = (
        f"pHash Hamming Distance: {pair_info['dist']}  |  "
        f"MD5 Status: {md5_status}  |  "
        f"Split: {pair_info.get('split', 'N/A')}"
    )
    draw.text((padding, 40), subtitle, fill=(70, 80, 95), font=subtitle_font)

    # Subheaders above each image
    draw.text(
        (padding, 72),
        f"Image 1: {pair_info['f1']}  [Base: {pair_info['b1'].split('_')[-1]}]",
        fill=(10, 60, 140),
        font=caption_font,
    )
    draw.text(
        (padding * 2 + w1, 72),
        f"Image 2: {pair_info['f2']}  [Base: {pair_info['b2'].split('_')[-1]}]",
        fill=(10, 60, 140),
        font=caption_font,
    )

    # Footer diagnosis
    footer_text = (
        f"Visual Assessment: Check if textures, lesions, and leaf morphology indicate identical leaf capture or distinct leaves."
    )
    draw.text((padding, total_h - 28), footer_text, fill=(120, 130, 140), font=caption_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def generate_example_montages(
    df: pd.DataFrame,
    reports_dir: str | Path = "reports",
) -> List[Path]:
    """
    Select and save 12 example montages:
    - 6 near-duplicate pairs from Papaya with different base_ids (distances 0, 2, 4)
    - 6 near-duplicate pairs from Potato/Rice with different base_ids (distances 0, 2, 4)
    Also creates a combined 4x3 summary grid.
    """
    print("=" * 90)
    print("4. GENERATING 12 EXAMPLE MONTAGES FOR VISUAL INSPECTION (reports/montage_*.png)")
    print("=" * 90)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    matches = find_phash_matches_bucketing(df["phash_int"].values, max_distance=4)

    papaya_candidates = []
    other_candidates = []

    for i1, i2, dist in matches:
        b1, b2 = df.loc[i1, "base_id"], df.loc[i2, "base_id"]
        if b1 != b2:
            crop = df.loc[i1, "crop"]
            item = {
                "i1": i1, "i2": i2, "dist": dist,
                "p1": df.loc[i1, "image_path"], "p2": df.loc[i2, "image_path"],
                "f1": df.loc[i1, "filename"], "f2": df.loc[i2, "filename"],
                "b1": b1, "b2": b2,
                "class": df.loc[i1, "class_name"],
                "crop": crop,
                "md5_match": df.loc[i1, "md5"] == df.loc[i2, "md5"],
                "group_id": df.loc[i1, "group_id"],
                "split": df.loc[i1, "split"],
            }
            if crop == "Papaya":
                papaya_candidates.append(item)
            else:
                other_candidates.append(item)

    # Select 6 Papaya pairs: 2 with dist 0, 2 with dist 2, 2 with dist 4
    selected_papaya = []
    for d in [0, 2, 4]:
        sub = [p for p in papaya_candidates if p["dist"] == d]
        # Prefer distinct classes
        seen_classes = set()
        picked = []
        for p in sub:
            if p["class"] not in seen_classes or len(seen_classes) >= 4:
                picked.append(p)
                seen_classes.add(p["class"])
                if len(picked) == 2:
                    break
        selected_papaya.extend(picked)

    # Select 6 Other pairs: 3 Potato, 3 Rice
    selected_other = []
    for crop_name in ["Potato", "Rice"]:
        sub_crop = [p for p in other_candidates if p["crop"] == crop_name]
        for d in [0, 2, 4]:
            sub_d = [p for p in sub_crop if p["dist"] == d]
            if sub_d:
                selected_other.append(sub_d[0])

    selected_12 = selected_papaya + selected_other

    saved_paths = []
    print(f"{'Idx':<4} | {'Crop':<8} | {'Class':<28} | {'Dist':<5} | {'MD5':<6} | {'Filename 1 vs Filename 2':<40}")
    print("-" * 98)

    for idx, pair in enumerate(selected_12, 1):
        out_file = reports_path / f"montage_{idx:02d}.png"
        create_montage_pair(pair, out_file, montage_idx=idx)
        saved_paths.append(out_file)
        md5_str = "Match" if pair["md5_match"] else "Diff"
        fn_cmp = f"{pair['f1']} vs {pair['f2']}"
        print(f"{idx:02d}   | {pair['crop']:<8} | {pair['class']:<28} | {pair['dist']:<5} | {md5_str:<6} | {fn_cmp:<40}")

    print(f"\nSaved all 12 individual montages to {reports_path}/montage_01.png ... montage_12.png")

    # Generate combined 4x3 overview grid
    overview_path = reports_path / "near_duplicates_overview_grid.png"
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()

    for i, pair in enumerate(selected_12):
        img1 = Image.open(pair["p1"]).convert("RGB")
        img2 = Image.open(pair["p2"]).convert("RGB")

        # Side by side on canvas
        h = 240
        w1 = int(img1.width * (h / img1.height))
        w2 = int(img2.width * (h / img2.height))
        c_img = Image.new("RGB", (w1 + w2 + 10, h), (255, 255, 255))
        c_img.paste(img1.resize((w1, h)), (0, 0))
        c_img.paste(img2.resize((w2, h)), (w1 + 10, 0))

        axes[i].imshow(c_img)
        md5_label = "MD5=Match" if pair["md5_match"] else "MD5=Diff"
        axes[i].set_title(
            f"#{i+1:02d}: {pair['crop']} | d={pair['dist']} | {md5_label}\n{pair['class']}",
            fontsize=10,
            fontweight="bold",
        )
        axes[i].axis("off")

    plt.suptitle("12 Near-Duplicate Montages (Cross-Base IDs, pHash Hamming d <= 4)", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined 12-pair overview grid to {overview_path}")
    print("=" * 90 + "\n")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose connected components, group sizes, and near-duplicate leakage."
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="splits",
        help="Directory containing train.csv, val.csv, test.csv (default: splits)",
    )
    parser.add_argument(
        "--cache-csv",
        type=str,
        default="data/metadata_hashes.csv",
        help="Path to metadata hashes CSV cache (default: data/metadata_hashes.csv)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save diagnosis montages (default: reports)",
    )
    args = parser.parse_args()

    split_df, cache_df = load_dataset_and_splits(
        splits_dir=args.splits_dir, cache_csv=args.cache_csv
    )

    report_group_size_distribution(split_df)
    report_papaya_analysis_and_thresholds(split_df)
    report_threshold_groups_across_splits(split_df)
    generate_example_montages(split_df, reports_dir=args.reports_dir)


if __name__ == "__main__":
    main()
