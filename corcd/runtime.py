"""Reproducibility and result bookkeeping for the main experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random

from .dataset_registry import DATASET_CONFIG

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def source_hash() -> str:
    """Identify the shipped implementation even before a Git repository exists."""
    digest = hashlib.sha256()
    paths = sorted((ROOT / "corcd").glob("*.py")) + [ROOT / "run.py"]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def write_info(directory: Path, info: dict) -> None:
    target = directory / "run_info.json"
    temporary = directory / ".run_info.tmp"
    temporary.write_text(json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def input_signature(data_root: Path, dataset: str) -> str:
    """Identify input files by relative name, byte size, and modification time."""
    spec = DATASET_CONFIG[dataset]
    directory = data_root / spec["path"]
    names = [*spec["modalities"].values(), "ground_truth.csv"]
    files = []
    for name in names:
        metadata = (directory / name).stat()
        files.append((name, metadata.st_size, metadata.st_mtime_ns))
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()[:16]


def completed(directory: Path, *, dataset: str, seed: int, config_id: str,
              device: str, data_root: Path) -> bool:
    """Resume only complete results from this configuration and implementation."""
    required = ("predictions.csv", "joint_embedding.npy", "config.yaml", "run_info.json")
    if not all((directory / name).is_file() for name in required):
        return False
    try:
        info = json.loads((directory / "run_info.json").read_text(encoding="utf-8"))
        return (
            info.get("status") == "success"
            and info.get("method") == "CORCD"
            and info.get("dataset") == dataset
            and info.get("seed") == seed
            and info.get("config_id") == config_id
            and info.get("source_id") == source_hash()
            and info.get("device") == device
            and info.get("data_root") == str(data_root)
            and info.get("input_id") == input_signature(data_root, dataset)
        )
    except (OSError, ValueError):
        return False
