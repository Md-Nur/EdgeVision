from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
