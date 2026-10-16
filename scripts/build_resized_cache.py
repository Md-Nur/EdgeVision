"""
Build resized image cache for multi-crop leaf disease classification.

Reads splits/*.csv, resizes each image so the short side is 256
(preserves aspect ratio, Lanczos interpolation, JPEG quality 95),
saves it under data_cache/<same relative path>, and writes splits_cache/*.csv
pointing to data_cache.

Uses multiprocessing across all CPU cores and skips files already done.
Prints total image count and total cache size in MB.
"""

import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Set

import pandas as pd
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_relative_subpath(raw_path: str) -> str:
    """
    Extract relative path under data directory.
    E.g. 'data/Final Dataset/Papaya/...' -> 'Final Dataset/Papaya/...'
    """
    p = str(raw_path).replace("\\", "/")
    if p.startswith("data/"):
        return p[len("data/"):]
    return p


def resize_and_save_worker(args: Tuple[str, str, int, int]) -> int:
    """
    Worker function to resize a single image.
    args: (src_path_str, dst_path_str, target_short_side, quality)
    Returns: 1 if newly resized, 0 if skipped, -1 if failed.
    """
    src_path_str, dst_path_str, target_short_side, quality = args
    dst_path = Path(dst_path_str)

    # Skip if already done and valid
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return 0

    src_path = Path(src_path_str)
    if not src_path.exists():
        return -1

    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            if w < h:
                new_w = target_short_side
                new_h = int(round(h * target_short_side / w))
            else:
                new_h = target_short_side
                new_w = int(round(w * target_short_side / h))

            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            resized.save(dst_path, format="JPEG", quality=quality)
            return 1
    except Exception as e:
        print(f"Error processing {src_path}: {e}", file=sys.stderr)
        return -1


def build_cache(
    splits_dir: str = "splits",
    splits_cache_dir: str = "splits_cache",
    data_cache_dir: str = "data_cache",
    target_short_side: int = 256,
    quality: int = 95,
) -> None:
    splits_path = Path(splits_dir)
    splits_cache_path = Path(splits_cache_dir)
    data_cache_path = Path(data_cache_dir)

    data_cache_path.mkdir(parents=True, exist_ok=True)
    splits_cache_path.mkdir(parents=True, exist_ok=True)

    # Find all CSV files in splits_dir (including subdirectories like dedup)
    csv_files = sorted(splits_path.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV split files found in {splits_path}")

    print(f"Found {len(csv_files)} split CSV files in {splits_path}")

    # Collect unique image paths
    tasks = []
    seen_destinations: Set[str] = set()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "image_path" not in df.columns:
            continue

        for raw_path in df["image_path"]:
            rel_subpath = get_relative_subpath(raw_path)
            src_path = Path(raw_path)
            if not src_path.exists():
                src_path = Path("data") / rel_subpath

            dst_path = data_cache_path / rel_subpath
            dst_str = str(dst_path)

            if dst_str not in seen_destinations:
                seen_destinations.add(dst_str)
                tasks.append((str(src_path), dst_str, target_short_side, quality))

    num_total = len(tasks)
    num_cpus = cpu_count()
    print(f"Total unique images to cache: {num_total}")
    print(f"Processing using {num_cpus} CPU cores (Lanczos, short side {target_short_side}, JPEG quality {quality})...")

    # Run multiprocessing
    newly_processed = 0
    skipped = 0
    failed = 0

    with Pool(processes=num_cpus) as pool:
        for idx, res in enumerate(pool.imap_unordered(resize_and_save_worker, tasks, chunksize=64)):
            if res == 1:
                newly_processed += 1
            elif res == 0:
                skipped += 1
            else:
                failed += 1

            if (idx + 1) % 2500 == 0 or (idx + 1) == num_total:
                print(f"Progress: {idx + 1}/{num_total} ({(idx + 1) / num_total * 100:.1f}%) | "
                      f"New: {newly_processed} | Skipped: {skipped} | Failed: {failed}")

    # Also create symlink data_cache/data -> data_cache if needed for legacy path compatibility
    symlink_target = data_cache_path / "data"
    if not symlink_target.exists():
        try:
            os.symlink(".", str(symlink_target), target_is_directory=True)
        except Exception:
            pass

    # Write splits_cache/*.csv
    print("\nWriting cached split CSV files to splits_cache/...")
    for csv_file in csv_files:
        rel_to_splits = csv_file.relative_to(splits_path)
        out_csv_path = splits_cache_path / rel_to_splits
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_file)
        if "image_path" in df.columns:
            # Point image_path to data_cache/<rel_subpath>
            df["image_path"] = df["image_path"].apply(
                lambda p: f"data_cache/{get_relative_subpath(p)}"
            )
        df.to_csv(out_csv_path, index=False)
        print(f"  - Wrote {out_csv_path} ({len(df)} rows)")

    # Calculate final cache statistics
    cached_images = [p for p in data_cache_path.rglob("*") if p.is_file() and not p.is_symlink() and not p.name.startswith(".")]
    total_bytes = sum(p.stat().st_size for p in cached_images)
    total_mb = total_bytes / (1024 * 1024)

    print("\n" + "=" * 80)
    print("RESIZED CACHE BUILD COMPLETE")
    print(f"Total Cached Images: {len(cached_images):,}")
    print(f"Total Cache Size:   {total_mb:.2f} MB")
    print(f"Newly Processed:    {newly_processed:,}")
    print(f"Skipped (pre-done): {skipped:,}")
    print(f"Failed:             {failed:,}")
    print(f"Cache Location:     {data_cache_path.resolve()}")
    print(f"Splits Cache:       {splits_cache_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    build_cache()
