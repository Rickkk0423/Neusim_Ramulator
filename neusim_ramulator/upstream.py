"""Load pinned, unmodified official sources without NeuSim's Ray runtime."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def add_upstream(name: str, relative: str) -> Path:
    path = Path(os.environ.get(name, str(ROOT / relative))).resolve()
    if not path.is_dir():
        raise RuntimeError(f"Missing upstream at {path}; run git submodule update --init --recursive")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
