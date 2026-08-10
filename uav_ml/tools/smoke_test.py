"""Mandatory deterministic tiny-overfit and checkpoint inference smoke test."""

import argparse
import json
import tempfile
from pathlib import Path

from uav_ml.datasets.dataset import BcEpisodeDataset
from uav_ml.datasets.synthetic_expert import generate_synthetic_dataset
from uav_ml.inference import BcPolicyInference
from uav_ml.train_bc import train


def run_smoke_test(device: str = "cpu", epochs: int = 30) -> dict:
    """Overfit a tiny labeled fixture and prove deterministic reload inference."""
    with tempfile.TemporaryDirectory(prefix="uav-bc-smoke-") as directory:
        root = Path(directory)
        dataset_path = root / "dataset"
        checkpoint_path = root / "bc_smoke.pt"
        history_path = root / "history.csv"
        generate_synthetic_dataset(
            dataset_path,
            episodes=4,
            maximum_steps=10,
            seed=101,
        )
        result = train(
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=30,
            learning_rate=5e-3,
            device_name=device,
            seed=103,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
        )
        ratio = result["final_train_loss"] / result["initial_train_loss"]
        if not ratio < 0.55:
            raise AssertionError(
                f"tiny overfit did not decrease substantially: ratio={ratio:.4f}"
            )
        dataset = BcEpisodeDataset(dataset_path, "train")
        sample = dataset[0]
        first = BcPolicyInference(str(checkpoint_path), device).predict(
            sample["depth"].numpy(),
            sample["velocity"].numpy(),
            sample["goal_direction"].numpy(),
        )
        second = BcPolicyInference(str(checkpoint_path), device).predict(
            sample["depth"].numpy(),
            sample["velocity"].numpy(),
            sample["goal_direction"].numpy(),
        )
        if not (first == second).all():
            raise AssertionError("reloaded eval inference is not deterministic")
        result.update(
            {
                "classification": "SYNTHETIC_SOFTWARE_FIXTURE_ONLY",
                "loss_ratio": ratio,
                "same_sample_inference": first.tolist(),
                "same_sample_inference_deterministic": True,
                "temporary_artifacts_removed": True,
            }
        )
        print("BC SMOKE TEST SUCCESS")
        print(json.dumps(result, indent=2, sort_keys=True))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    run_smoke_test(args.device, args.epochs)


if __name__ == "__main__":
    main()

