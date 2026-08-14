"""Plot the metrics that were actually recorded by the city BC/PPO baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", default="training_runs/latent_bc_city_v0")
    parser.add_argument("--ppo-run", default="training_runs/ppo_city_v0")
    parser.add_argument("--output", default="training_runs/city_bc_ppo_metrics.png")
    args = parser.parse_args()

    bc_root = Path(args.bc_run)
    ppo_root = Path(args.ppo_run)
    with (bc_root / "history.csv").open(encoding="utf-8") as stream:
        bc_history = list(csv.DictReader(stream))
    ppo_summary = json.loads((ppo_root / "summary.json").read_text(encoding="utf-8"))
    ppo_history = ppo_summary["history"]
    bc_evaluation = json.loads(
        (bc_root / "closed_loop_seed50000.json").read_text(encoding="utf-8")
    )
    ppo_evaluation = ppo_summary["evaluation"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    epochs = [int(row["epoch"]) for row in bc_history]
    axes[0].plot(epochs, [float(row["train_mse"]) for row in bc_history], label="train")
    axes[0].plot(
        epochs,
        [float(row["validation_mse"]) for row in bc_history],
        label="validation",
    )
    axes[0].set(title="BC open-loop action error", xlabel="Epoch", ylabel="MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    timesteps = [row["timesteps"] for row in ppo_history]
    axes[1].plot(
        timesteps,
        [100.0 * row["training_episode_success_rate"] for row in ppo_history],
        marker="o",
        markersize=3,
    )
    axes[1].set(
        title="PPO cumulative training success",
        xlabel="Environment timesteps",
        ylabel="Success rate (%)",
        ylim=(0, 100),
    )
    axes[1].grid(alpha=0.25)
    axes[1].text(
        0.02,
        0.03,
        "Training seeds; not held-out evaluation",
        transform=axes[1].transAxes,
        fontsize=8,
    )

    names = ["BC before PPO", "PPO final"]
    success = [100 * bc_evaluation["success_rate"], 100 * ppo_evaluation["success_rate"]]
    collision = [
        100 * bc_evaluation["collision_rate"],
        100 * ppo_evaluation["collision_rate"],
    ]
    positions = range(len(names))
    width = 0.34
    axes[2].bar([x - width / 2 for x in positions], success, width, label="success")
    axes[2].bar([x + width / 2 for x in positions], collision, width, label="collision")
    axes[2].set_xticks(list(positions), names)
    axes[2].set(
        title="Held-out seeds 50000–50019",
        ylabel="Episode rate (%)",
        ylim=(0, 100),
    )
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output.resolve())


if __name__ == "__main__":
    main()
