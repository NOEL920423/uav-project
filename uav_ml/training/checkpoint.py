"""Versioned BC checkpoint save/load contract."""

from pathlib import Path

import torch

from uav_ml.contracts import (
    ACTION_CONTRACT_VERSION,
    OBSERVATION_CONTRACT_VERSION,
)
from uav_ml.models import BcPolicyConfig, BcPolicyV0
from uav_ml.training.normalization import NormalizationStats


CHECKPOINT_FORMAT_VERSION = "uav_bc_checkpoint_v0.1"


def save_checkpoint(
    path: str | Path,
    model: BcPolicyV0,
    normalization: NormalizationStats,
    training_metadata: dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save weights together with every contract needed for inference."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "normalization": normalization.to_dict(),
        "training_metadata": training_metadata,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, output)
    return output


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[BcPolicyV0, NormalizationStats, dict]:
    """Strictly reconstruct a policy and its preprocessing metadata."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported BC checkpoint format")
    if payload.get("observation_contract_version") != OBSERVATION_CONTRACT_VERSION:
        raise ValueError("checkpoint observation contract is incompatible")
    if payload.get("action_contract_version") != ACTION_CONTRACT_VERSION:
        raise ValueError("checkpoint action contract is incompatible")
    model = BcPolicyV0(BcPolicyConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    stats = NormalizationStats.from_dict(payload["normalization"])
    return model, stats, payload

