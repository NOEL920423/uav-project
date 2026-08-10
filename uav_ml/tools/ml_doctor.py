"""Report the local ML runtime without installing or modifying anything."""

import importlib.util
import os
import platform
import sys
from pathlib import Path


def _version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "MISSING"
    module = __import__(name)
    return str(getattr(module, "__version__", "installed/version unknown"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"NumPy: {_version('numpy')}")
    print(f"PyTorch: {_version('torch')}")
    print(f"Dataset location: {root / 'datasets' / 'bc_v0'}")
    print(f"Checkpoint location: {root / 'checkpoints'}")
    print(f"Training run location: {root / 'training_runs'}")
    if importlib.util.find_spec("torch") is None:
        print("CUDA available: unavailable because PyTorch is missing")
        raise SystemExit(1)
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda or 'none'}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    required = ("numpy", "torch")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    print(f"Required packages: {', '.join(required)}")
    print(f"Required package status: {'OK' if not missing else 'MISSING ' + ','.join(missing)}")
    print(f"UAV_ML_DATASET: {os.environ.get('UAV_ML_DATASET', '<unset>')}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

