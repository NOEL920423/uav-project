"""Deterministic episode-level train/validation splitting."""

import hashlib


def split_episode_ids(
    episode_ids: list[str], validation_fraction: float = 0.2, seed: int = 17
) -> tuple[list[str], list[str]]:
    """Split whole episodes using a stable hash independent of Python hash."""
    if len(episode_ids) < 2:
        raise ValueError("at least two episodes are required for a split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs must be unique")

    def key(episode_id: str) -> bytes:
        return hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()

    ordered = sorted(episode_ids, key=key)
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation_count = min(validation_count, len(ordered) - 1)
    validation = sorted(ordered[:validation_count])
    train = sorted(ordered[validation_count:])
    return train, validation

