"""Evaluate the formal BC baseline closed-loop on unseen Isaac city seeds."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(
    prog="./uav bc-eval",
    description=(
        "Run policy-only closed-loop evaluation of the formal BC baseline "
        "on expert-disjoint Isaac city seeds."
    ),
)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--checkpoint")
parser.add_argument(
    "--dataset",
    default=str(REPOSITORY_ROOT / "artifacts/datasets/bc_expert_highrise_v1"),
)
parser.add_argument("--encoder")
parser.add_argument("--output")
parser.add_argument("--seed-base", type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from uav_ml.isaac.fixed_height_city_env import (  # noqa: E402
    IsaacFixedHeightCityEnv,
)
from uav_ml.tools.bc_evaluation import (  # noqa: E402
    BcPolicyRuntime,
    resolve_checkpoint,
    run_closed_loop_evaluation,
    unseen_evaluation_seeds,
)
from uav_ml.train_bc import resolve_device  # noqa: E402


def main() -> None:
    """Load the trained baseline and give it exclusive environment control."""
    dataset_root = Path(args.dataset).resolve()
    checkpoint = resolve_checkpoint(
        Path(args.checkpoint) if args.checkpoint else None,
        REPOSITORY_ROOT,
    )
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = REPOSITORY_ROOT.joinpath(
            "artifacts",
            "experiments",
            "bc_evaluation",
            time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()),
        )
    device = resolve_device(args.device)
    runtime = BcPolicyRuntime(
        checkpoint,
        device,
        Path(args.encoder) if args.encoder else None,
    )
    if runtime.training_dataset_root != dataset_root:
        raise ValueError(
            "evaluation dataset differs from the checkpoint dataset reference: "
            f"{runtime.training_dataset_root} != {dataset_root}"
        )
    seeds = unseen_evaluation_seeds(
        dataset_root,
        args.episodes,
        args.seed_base,
    )
    environment = IsaacFixedHeightCityEnv(device=str(device))
    run_closed_loop_evaluation(
        environment=environment,
        policy=runtime.act,
        seeds=seeds,
        output_dir=output_dir,
        checkpoint=runtime.checkpoint_path,
        checkpoint_sha256=runtime.checkpoint_sha256,
        encoder_checkpoint=runtime.encoder_path,
        encoder_sha256=runtime.encoder_sha256,
        dataset_root=dataset_root,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
