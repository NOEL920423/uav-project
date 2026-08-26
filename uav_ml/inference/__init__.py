"""Checkpoint-backed BC policy inference with lazy optional dependencies."""

__all__ = [
    "BcPolicyInference",
    "LatentCityPolicyInput",
    "RgbEncoderInference",
    "load_bc_actor",
]


def __getattr__(name: str):
    """Load PyTorch-backed helpers only when a caller requests them."""
    if name == "BcPolicyInference":
        from uav_ml.inference.policy import BcPolicyInference
        return BcPolicyInference
    if name == "RgbEncoderInference":
        from uav_ml.inference.rgb_encoder import RgbEncoderInference
        return RgbEncoderInference
    if name in {"LatentCityPolicyInput", "load_bc_actor"}:
        from uav_ml.inference.latent_city_policy import (
            LatentCityPolicyInput,
            load_bc_actor,
        )
        return {
            "LatentCityPolicyInput": LatentCityPolicyInput,
            "load_bc_actor": load_bc_actor,
        }[name]
    raise AttributeError(name)
