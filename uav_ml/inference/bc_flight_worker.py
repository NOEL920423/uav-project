"""Line-delimited local worker for PyTorch BC inference."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from uav_ml.inference.bc_flight import BcFlightPolicy, resolve_checkpoint


def _write(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local PyTorch worker used by BC flight."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image-source")
    source_group.add_argument("--image-sources", nargs="+")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        checkpoint = resolve_checkpoint(
            args.repository_root, args.checkpoint
        )
        policy = BcFlightPolicy(
            checkpoint,
            args.image_sources or args.image_source,
            torch.device(args.device),
        )
        runtime = {"device": str(policy.device)}
        if policy.device.type == "cuda":
            runtime["device_name"] = torch.cuda.get_device_name(policy.device)
        _write({
            "ready": True,
            "identity": asdict(policy.identity),
            "runtime": runtime,
        })
    except Exception as error:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        _write({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return 2
    for line in sys.stdin:
        try:
            request_started_ns = time.perf_counter_ns()
            request = json.loads(line)
            encoded_images = request.get("images_base64")
            if encoded_images is not None:
                if not isinstance(encoded_images, dict):
                    raise ValueError("images_base64 must be an object")
                image = {
                    source: base64.b64decode(value, validate=True)
                    for source, value in encoded_images.items()
                }
            else:
                encoded_image = request.get(
                    "image_base64", request.get("jpeg_base64")
                )
                image = base64.b64decode(encoded_image, validate=True)
            state = np.asarray(request["state8"], dtype=np.float32)
            input_decode_ms = (time.perf_counter_ns() - request_started_ns) / 1e6
            profile = bool(request.get("profile", False))
            if profile:
                action, stage_ms = policy.act_profiled(image, state)
            else:
                action = policy.act(image, state)
                stage_ms = {}
            _write({
                "action": [float(value) for value in action],
                "profile": {
                    "worker_input_decode_ms": input_decode_ms,
                    **stage_ms,
                    "device": str(policy.device),
                    **({"device_name": torch.cuda.get_device_name(policy.device)}
                       if policy.device.type == "cuda" else {}),
                } if profile else None,
            })
        except Exception as error:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            _write({"error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
