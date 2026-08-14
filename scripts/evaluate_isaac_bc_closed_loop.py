"""Evaluate the learned BC actor closed-loop on unseen Isaac city seeds."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default="training_runs/latent_bc_city_v0/best.pt")
parser.add_argument("--output", default="training_runs/latent_bc_city_v0/closed_loop.json")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed-base", type=int, default=30000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from uav_ml.inference.latent_city_policy import load_bc_actor  # noqa: E402
from uav_ml.isaac.fixed_height_city_env import IsaacFixedHeightCityEnv  # noqa: E402


def main() -> None:
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    device = torch.device(args.device)
    actor, input_pipeline, _ = load_bc_actor(args.checkpoint, device)
    env = IsaacFixedHeightCityEnv(device=args.device)
    records = []
    for offset in range(args.episodes):
        seed = args.seed_base + offset
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        while True:
            with torch.inference_mode():
                action = actor(input_pipeline.encode(observation))[0].cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        record = {
            "seed": seed,
            "success": bool(info["success"]),
            "collision": bool(info["collision"] or info["out_of_bounds"]),
            "timeout": bool(info["timeout"]),
            "steps": int(info["step_index"]),
            "return": episode_return,
            "final_distance_m": float(info["goal_distance_m"]),
        }
        records.append(record)
        print(
            f"seed={seed} success={record['success']} collision={record['collision']} "
            f"timeout={record['timeout']} distance={record['final_distance_m']:.3f}",
            flush=True,
        )
    summary = {
        "policy": "latent_bc_closed_loop",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed_base": args.seed_base,
        "episodes": args.episodes,
        "success_rate": sum(item["success"] for item in records) / len(records),
        "collision_rate": sum(item["collision"] for item in records) / len(records),
        "timeout_rate": sum(item["timeout"] for item in records) / len(records),
        "mean_return": sum(item["return"] for item in records) / len(records),
        "mean_final_distance_m": sum(item["final_distance_m"] for item in records) / len(records),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
