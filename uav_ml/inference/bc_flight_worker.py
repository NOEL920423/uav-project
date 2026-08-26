"""Line-delimited local worker for PyTorch BC inference."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from uav_ml.inference.bc_flight import TopRgbBcPolicy, resolve_checkpoint


def _write(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local PyTorch worker used by BC flight."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        checkpoint = resolve_checkpoint(
            args.repository_root, args.checkpoint
        )
        policy = TopRgbBcPolicy(
            checkpoint, args.image_source, torch.device(args.device)
        )
        _write({"ready": True, "identity": asdict(policy.identity)})
    except Exception as error:  # noqa: BLE001
        _write({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return 2
    for line in sys.stdin:
        try:
            request = json.loads(line)
            image = base64.b64decode(request["jpeg_base64"], validate=True)
            state = np.asarray(request["state8"], dtype=np.float32)
            action = policy.act(image, state)
            _write({"action": [float(value) for value in action]})
        except Exception as error:  # noqa: BLE001
            _write({"error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
