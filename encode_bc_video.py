#!/usr/bin/env python3
"""Encode synchronized UAV FPV/TOP PNG frames as a side-by-side MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fpv_dir = episode_dir / "images" / "fpv"
    top_dir = episode_dir / "images" / "top"

    pairs = []
    for fpv_path in sorted(fpv_dir.glob("frame_*.png")):
        top_path = top_dir / fpv_path.name
        if top_path.is_file():
            pairs.append((fpv_path, top_path))
    if not pairs:
        raise RuntimeError(f"No synchronized frame pairs found in {episode_dir}")

    first_fpv = cv2.imread(str(pairs[0][0]), cv2.IMREAD_COLOR)
    first_top = cv2.imread(str(pairs[0][1]), cv2.IMREAD_COLOR)
    if first_fpv is None or first_top is None:
        raise RuntimeError("Could not read the first synchronized image pair")
    height = min(first_fpv.shape[0], first_top.shape[0])
    width = min(first_fpv.shape[1], first_top.shape[1])
    frame_size = (width * 2, height)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer: {temporary}")

    written = 0
    try:
        for frame_index, (fpv_path, top_path) in enumerate(pairs, start=1):
            fpv = cv2.imread(str(fpv_path), cv2.IMREAD_COLOR)
            top = cv2.imread(str(top_path), cv2.IMREAD_COLOR)
            if fpv is None or top is None:
                continue
            if fpv.shape[:2] != (height, width):
                fpv = cv2.resize(fpv, (width, height), interpolation=cv2.INTER_AREA)
            if top.shape[:2] != (height, width):
                top = cv2.resize(top, (width, height), interpolation=cv2.INTER_AREA)
            cv2.putText(
                fpv,
                "BC FLIGHT - FPV",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                top,
                "BC FLIGHT - TOP",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            combined = cv2.hconcat((fpv, top))
            cv2.putText(
                combined,
                f"frame {frame_index:03d}/{len(pairs):03d}",
                (frame_size[0] - 260, frame_size[1] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(combined)
            written += 1
    finally:
        writer.release()

    if written == 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("MP4 encoding produced no usable output")
    temporary.replace(output)
    capture = cv2.VideoCapture(str(output))
    metadata = {
        "output": str(output),
        "codec": "mp4v",
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": written / float(args.fps),
        "size_bytes": output.stat().st_size,
    }
    capture.release()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
