"""Offline rendering of the exact images consumed by a BC flight policy."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw


VIDEO_METADATA_SCHEMA = "uav_bc_policy_input_video/v1"
FRAME_MANIFEST_FILENAME = "frames.jsonl"


@dataclass(frozen=True)
class PolicyInputFrame:
    """One image and action that were consumed by a successful inference."""

    inference_count: int
    timestamp_ns: int
    image_filename: str
    action_body: tuple[float, float, float]


def _image_suffix(image_bytes: bytes) -> str:
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        if image.format == "JPEG":
            return ".jpg"
        if image.format == "PNG":
            return ".png"
    raise ValueError("policy input image must be JPEG or PNG")


def capture_policy_input_frame(
    spool_dir: Path,
    *,
    image_bytes: bytes,
    image_source: str,
    inference_count: int,
    timestamp_ns: int,
    action_body: tuple[float, float, float],
) -> None:
    """Persist one exact policy input locally, after inference has succeeded."""
    spool_dir = spool_dir.expanduser().resolve()
    frames_dir = spool_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    suffix = _image_suffix(image_bytes)
    filename = f"{inference_count:06d}{suffix}"
    target = frames_dir / filename
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(image_bytes)
    temporary.replace(target)
    row = {
        "inference_count": int(inference_count),
        "timestamp_ns": int(timestamp_ns),
        "image_filename": filename,
        "image_source": image_source,
        "action_body": [float(value) for value in action_body],
    }
    with (spool_dir / FRAME_MANIFEST_FILENAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _load_frames(spool_dir: Path) -> tuple[str, list[PolicyInputFrame]]:
    source = ""
    frames = []
    manifest = spool_dir / FRAME_MANIFEST_FILENAME
    if not manifest.is_file():
        return source, frames
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source = str(row["image_source"])
        action = tuple(float(value) for value in row["action_body"])
        if len(action) != 3:
            raise ValueError("policy video action must contain three values")
        image = spool_dir / "frames" / str(row["image_filename"])
        if not image.is_file():
            raise FileNotFoundError(f"policy video frame is missing: {image}")
        frames.append(PolicyInputFrame(
            inference_count=int(row["inference_count"]),
            timestamp_ns=int(row["timestamp_ns"]),
            image_filename=str(row["image_filename"]),
            action_body=action,
        ))
    return source, frames


def _annotate_frame(
    source: Path, target: Path, frame: PolicyInputFrame,
) -> None:
    with Image.open(source) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    forward, right, _ = frame.action_body
    center_x = 46
    center_y = 46
    draw.rounded_rectangle(
        (6, 6, 86, 86), radius=12, fill=(0, 0, 0), outline=(90, 90, 90)
    )

    magnitude = max(abs(forward), abs(right))
    if magnitude > 0.01:
        scale = 28.0 / magnitude
        end_x = center_x + right * scale
        end_y = center_y - forward * scale
        draw.line(
            (center_x, center_y, end_x, end_y),
            fill=(255, 220, 0),
            width=6,
        )
        angle_x = end_x - center_x
        angle_y = end_y - center_y
        length = (angle_x**2 + angle_y**2) ** 0.5
        head_length = 11.0
        base_x = end_x - head_length * angle_x / length
        base_y = end_y - head_length * angle_y / length
        wing_x = 5.0 * -angle_y / length
        wing_y = 5.0 * angle_x / length
        draw.polygon(
            [
                (end_x, end_y),
                (base_x + wing_x, base_y + wing_y),
                (base_x - wing_x, base_y - wing_y),
            ],
            fill=(255, 220, 0),
        )
    else:
        draw.ellipse(
            (center_x - 4, center_y - 4, center_x + 4, center_y + 4),
            fill=(180, 180, 180),
        )
    canvas.save(target, format="PNG")


def render_policy_input_video(
    spool_dir: Path, output_path: Path, *, fps: float = 20.0,
    metadata_path: Path | None = None,
) -> dict:
    """Render annotated policy inputs into an H.264 MP4 on local storage."""
    if fps <= 0.0:
        raise ValueError("video FPS must be positive")
    image_source, frames = _load_frames(spool_dir)
    if not frames:
        raise ValueError("no successful BC inferences were recorded")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to render policy_input.mp4")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uav_bc_video_") as directory:
        rendered = Path(directory)
        for index, frame in enumerate(frames):
            _annotate_frame(
                spool_dir / "frames" / frame.image_filename,
                rendered / f"{index:06d}.png", frame,
            )
        concat = rendered / "frames.txt"
        lines = []
        default_duration = 1.0 / fps
        for index, frame in enumerate(frames):
            filename = (rendered / f"{index:06d}.png").as_posix()
            lines.append(f"file '{filename}'")
            if index + 1 < len(frames):
                gap = (frames[index + 1].timestamp_ns - frame.timestamp_ns) / 1e9
                duration = gap if gap > 0.0 else default_duration
            else:
                duration = default_duration
            lines.append(f"duration {duration:.9f}")
        # The concat demuxer applies the final duration only when the last file
        # is repeated once.
        lines.append(f"file '{(rendered / f'{len(frames) - 1:06d}.png').as_posix()}'")
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary = output_path.with_name(
            output_path.stem + ".partial" + output_path.suffix
        )
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat), "-vsync", "vfr", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg policy video encoding failed: "
                f"{completed.stderr.strip() or completed.returncode}"
            )
        temporary.replace(output_path)
    metadata = {
        "schema": VIDEO_METADATA_SCHEMA,
        "video_filename": output_path.name,
        "image_source": image_source,
        "frame_count": len(frames),
        "inference_count_first": frames[0].inference_count,
        "inference_count_last": frames[-1].inference_count,
        "fps": fps,
        "timing": "source timestamps; one annotated image per successful inference",
    }
    metadata_path = metadata_path or output_path.with_name(
        "policy_input_video.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**metadata, "metadata_filename": metadata_path.name}
