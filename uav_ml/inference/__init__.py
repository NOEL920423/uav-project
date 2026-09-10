"""Checkpoint-backed BC policy inference with lazy optional dependencies."""

__all__ = [
    "BcPolicyInference",
    "RgbEncoderInference",
]


def __getattr__(name: str):
    """Load PyTorch-backed helpers only when a caller requests them."""
    if name == "BcPolicyInference":
        from uav_ml.inference.policy import BcPolicyInference
        return BcPolicyInference
    if name == "RgbEncoderInference":
        from uav_ml.inference.rgb_encoder import RgbEncoderInference
        return RgbEncoderInference
    raise AttributeError(name)
