"""Launch Isaac and validate clean synchronous A* reset/step rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=2000)
parser.add_argument("--output", default="run_logs/isaac_city_smoke.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from uav_ml.isaac.fixed_height_city_env import IsaacFixedHeightCityEnv  # noqa: E402


def main() -> None:
    env = IsaacFixedHeightCityEnv(device=args.device)
    observation, reset_info = env.reset(seed=args.seed)
    first_rgb = observation["rgb"].copy()
    steps = 0
    final_info = {}
    while True:
        action = env.expert_action()
        observation, _, terminated, truncated, final_info = env.step(action)
        steps += 1
        if steps % 10 == 0:
            print(
                f"smoke_step={steps} distance={final_info['goal_distance_m']:.3f}",
                flush=True,
            )
        if terminated or truncated:
            break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_path = output.with_suffix(".png")
    Image.fromarray(first_rgb).save(image_path)
    payload = {
        "seed": args.seed,
        "reset_info": reset_info,
        "rgb_shape": list(first_rgb.shape),
        "rgb_dtype": str(first_rgb.dtype),
        "rgb_min": int(first_rgb.min()),
        "rgb_max": int(first_rgb.max()),
        "steps": steps,
        "final": final_info,
        "image": str(image_path.resolve()),
        "success": bool(final_info.get("success")),
        "synchronized": bool(final_info.get("synchronized")),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    env.close()
    if not payload["success"] or not payload["synchronized"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
