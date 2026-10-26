"""Shared reproducibility and device selection for AE and BC training."""

import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and all visible CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda without silently accepting unavailable CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device
