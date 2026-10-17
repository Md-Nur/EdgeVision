"""
Path resolution and validation utilities for Mukti models and analysis scripts.

Enforces deterministic resolution according to loss versions:
- v1 resolves to legacy unsuffixed names: <model>_seed<seed>[_best.pth | .json]
- v2 resolves to version-suffixed names: <model>_seed<seed>_lossv2[_best.pth | .json]

Strictly prevents silent fallbacks across versions.
"""

from pathlib import Path
from typing import Dict, Optional, Union


def get_loss_suffix(loss_version: str) -> str:
    """
    Get filename suffix for loss version.
    
    Args:
        loss_version: 'v1' or 'v2'
        
    Returns:
        '' for 'v1' (legacy unsuffixed), '_loss<version>' for others.
    """
    if loss_version not in ("v1", "v2"):
        raise ValueError(f"Invalid loss_version '{loss_version}'. Must be 'v1' or 'v2'.")
    if loss_version == "v1":
        return ""
    return f"_loss{loss_version}"


def get_run_id(model_name: str, seed: int = 42, loss_version: str = "v2") -> str:
    """
    Construct canonical run ID for checkpoint and result naming.
    
    Format:
        v1: <model>_seed<seed>
        v2: <model>_seed<seed>_loss<version>
    """
    clean_name = model_name.replace("-", "_")
    suffix = get_loss_suffix(loss_version)
    return f"{clean_name}_seed{seed}{suffix}"


def resolve_checkpoint_path(
    model_name: str,
    seed: int = 42,
    loss_version: str = "v2",
    checkpoints_dir: Union[str, Path] = "checkpoints",
) -> Path:
    """
    Resolve checkpoint path for a given model, seed, and loss_version.
    
    Format:
        v1: <checkpoints_dir>/<model>_seed<seed>_best.pth
        v2: <checkpoints_dir>/<model>_seed<seed>_loss<version>_best.pth
    """
    checkpoints_dir = Path(checkpoints_dir)
    run_id = get_run_id(model_name, seed=seed, loss_version=loss_version)
    return checkpoints_dir / f"{run_id}_best.pth"


def resolve_result_path(
    model_name: str,
    seed: int = 42,
    loss_version: str = "v2",
    results_dir: Union[str, Path] = "results",
) -> Path:
    """
    Resolve evaluation result JSON path for a given model, seed, and loss_version.
    
    Format:
        v1: <results_dir>/<model>_seed<seed>.json
        v2: <results_dir>/<model>_seed<seed>_loss<version>.json
    """
    results_dir = Path(results_dir)
    run_id = get_run_id(model_name, seed=seed, loss_version=loss_version)
    return results_dir / f"{run_id}.json"


def verify_path_exists(
    path: Union[str, Path],
    description: str = "File",
    loss_version: Optional[str] = None,
) -> Path:
    """
    Verify that path exists.
    
    Raises:
        FileNotFoundError: If the path does not exist (no silent fallback).
    """
    p = Path(path)
    if not p.exists():
        version_str = f" for loss_version '{loss_version}'" if loss_version else ""
        raise FileNotFoundError(
            f"{description}{version_str} does not exist: {p} (silent fallback to another version is forbidden)."
        )
    return p


def print_and_verify_paths(
    checkpoints: Optional[Dict[str, Path]] = None,
    results: Optional[Dict[str, Path]] = None,
    loss_version: str = "v2",
    header: str = "RESOLVED PATHS",
) -> None:
    """
    Print all paths loaded at the top of an analysis run and verify their existence.
    
    Raises:
        FileNotFoundError: If any path does not exist (no silent fallback).
    """
    print("\n" + "=" * 80)
    print(f"{header} (loss_version='{loss_version}')")
    print("=" * 80)

    if checkpoints:
        print("[Checkpoints to Load]:")
        for key, p in checkpoints.items():
            status = "[EXISTS]" if p.exists() else "[MISSING]"
            print(f"  - {key:<20}: {p} {status}")

    if results:
        print("[Results to Load]:")
        for key, p in results.items():
            status = "[EXISTS]" if p.exists() else "[MISSING]"
            print(f"  - {key:<20}: {p} {status}")
    print("=" * 80 + "\n")

    # Strict verification with no silent fallback
    if checkpoints:
        for key, p in checkpoints.items():
            verify_path_exists(p, description=f"Checkpoint '{key}'", loss_version=loss_version)

    if results:
        for key, p in results.items():
            verify_path_exists(p, description=f"Result JSON '{key}'", loss_version=loss_version)
