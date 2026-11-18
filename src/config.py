"""Load YAML config and resolve paths."""

import argparse
import os
from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(mode: str = "pilot") -> dict:
    """Load the YAML config for the given mode and resolve paths."""
    config_path = _PROJECT_ROOT / "configs" / f"{mode}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve all paths relative to project root
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str(_PROJECT_ROOT / rel)
        os.makedirs(cfg["paths"][key], exist_ok=True)

    # Resolve splits file path
    cfg["splits"]["file"] = str(_PROJECT_ROOT / cfg["splits"]["file"])

    return cfg


def get_mode_from_args() -> str:
    """Parse --mode from CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    args, _ = parser.parse_known_args()
    return args.mode
