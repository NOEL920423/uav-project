"""Offline video of BC policy inputs, the TOP camera, and body actions."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from uav_ml.inference.bc_flight_contract import body_action_to_ned


VIDEO_METADATA_SCHEMA = "uav_bc_policy_input_video/v1"
FRAME_MANIFEST_FILENAME = "frames.jsonl"
VIDEO_SCALE = 2
VIDEO_SIZE = (1280 * VIDEO_SCALE, 720 * VIDEO_SCALE)
HUD_SUPERSAMPLING = 2
VIDEO_CRF = 16


@dataclass(frozen=True)
class PolicyInputFrame:
    """One exact source image and normalized action from an accepted inference."""

    inference_count: int
    timestamp_ns: int
    image_filename: str
    action_body: tuple[float, float, float]
    top_image_filename: str | None = None
    top_time_delta_ns: int | None = None


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
    top_image_bytes: bytes | None = None,
    top_time_delta_ns: int | None = None,
) -> None:
    """Persist the exact primary input and optional synchronized TOP image."""
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
    if top_image_bytes is not None:
        top_filename = f"{inference_count:06d}_top{_image_suffix(top_image_bytes)}"
        top_target = frames_dir / top_filename
        top_temporary = top_target.with_suffix(top_target.suffix + ".tmp")
        top_temporary.write_bytes(top_image_bytes)
        top_temporary.replace(top_target)
        row["top_image_filename"] = top_filename
        row["top_time_delta_ns"] = top_time_delta_ns
    elif image_source == "top_rgb":
        row["top_image_filename"] = filename
        row["top_time_delta_ns"] = 0
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
        top_filename = row.get("top_image_filename")
        if top_filename is None and source == "top_rgb":
            top_filename = str(row["image_filename"])
        if top_filename is not None and not (spool_dir / "frames" / top_filename).is_file():
            raise FileNotFoundError(f"TOP video frame is missing: {top_filename}")
        frames.append(PolicyInputFrame(
            inference_count=int(row["inference_count"]),
            timestamp_ns=int(row["timestamp_ns"]),
            image_filename=str(row["image_filename"]),
            action_body=action,
            top_image_filename=top_filename,
            top_time_delta_ns=row.get("top_time_delta_ns"),
        ))
    return source, frames


@lru_cache(maxsize=8)
def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _paste_view(
    canvas: Image.Image, source: Path | None, box: tuple[int, int, int, int],
    *, depth: bool = False,
    resampling: Image.Resampling = Image.Resampling.BILINEAR,
) -> None:
    """Fit a complete camera image without cropping or stretching it."""
    x0, y0, x1, y1 = box
    if source is None:
        ImageDraw.Draw(canvas).text(
            ((x0 + x1) / 2, (y0 + y1) / 2), "No synchronized TOP frame",
            font=_font(18 * VIDEO_SCALE), fill="#9bb1c9", anchor="mm",
        )
        return
    with Image.open(source) as image:
        if depth:
            # Raw FPV depth is uint16 millimetres. Display 0..30 m as grayscale;
            # the saved bytes and model preprocessing remain untouched.
            values = np.asarray(image, dtype=np.float32)
            pixels = np.clip(values * (255.0 / 30000.0), 0, 255).astype(np.uint8)
            view = Image.fromarray(pixels).convert("RGB")
        else:
            view = image.convert("RGB")
    view = ImageOps.contain(view, (x1 - x0, y1 - y0), resampling)
    canvas.paste(view, (x0 + (x1 - x0 - view.width) // 2,
                      y0 + (y1 - y0 - view.height) // 2))


def _annotate_frame(
    source: Path, target: Path, frame: PolicyInputFrame,
    *, image_source: str = "top_rgb", top_source: Path | None = None,
) -> None:
    """Compose model input, TOP view, direction pointer, and physical actions."""
    render_scale = VIDEO_SCALE * HUD_SUPERSAMPLING

    def scaled(values, factor=render_scale):
        return tuple(int(value * factor) for value in values)

    canvas = Image.new("RGB", tuple(value * HUD_SUPERSAMPLING for value in VIDEO_SIZE), "#070d16")
    draw = ImageDraw.Draw(canvas)
    for box in ((16, 16, 772, 704), (788, 16, 1264, 492), (788, 508, 1264, 704)):
        draw.rounded_rectangle(scaled(box), radius=12 * render_scale, fill="#101d2e",
                               outline="#36516b", width=render_scale)
    label = {"fpv_rgb": "FPV RGB", "fpv_depth": "FPV DEPTH", "top_rgb": "TOP RGB"}.get(
        image_source, image_source.upper()
    )
    draw.text(scaled((32, 30)), f"MODEL INPUT / {label}",
              font=_font(28 * render_scale, True), fill="#edf3fb")
    draw.text(
        scaled((32, 72)), f"Image used by inference #{frame.inference_count:06d}",
        font=_font(18 * render_scale), fill="#9bb1c9",
    )
    draw.text(scaled((804, 30)), "TOP VIEW", font=_font(24 * render_scale, True), fill="#edf3fb")
    draw.text(scaled((804, 66)), "Observer camera", font=_font(16 * render_scale), fill="#9bb1c9")

    draw.text(scaled((804, 524)), "BODY-FRAME ACTION", font=_font(20 * render_scale, True), fill="#edf3fb")
    forward, right, _, yaw_rate = body_action_to_ned(frame.action_body, 0.0)
    for y, label, value, unit in (
        (564, "Forward", forward, "m/s"),
        (606, "Right", right, "m/s"),
        (648, "Yaw rate", yaw_rate, "rad/s"),
    ):
        draw.text(scaled((804, y)), label, font=_font(18 * render_scale), fill="#9bb1c9")
        draw.text(scaled((918, y)), f"{value:+.2f} {unit}",
                  font=_font(20 * render_scale, True), fill="#36d5f4")
    center_x, center_y, radius = 1180, 619, 54
    draw.ellipse(scaled((center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius)),
                 outline="#9bb1c9", width=2 * render_scale)
    draw.line(scaled((center_x - 42, center_y, center_x + 42, center_y)),
              fill="#36516b", width=render_scale)
    draw.line(scaled((center_x, center_y - 42, center_x, center_y + 42)),
              fill="#36516b", width=render_scale)
    for label, x, y in (("F", center_x, center_y - 65), ("B", center_x, center_y + 65),
                        ("L", center_x - 65, center_y), ("R", center_x + 65, center_y)):
        draw.text(scaled((x, y)), label, font=_font(15 * render_scale), fill="#edf3fb", anchor="mm")
    magnitude = (forward**2 + right**2) ** 0.5
    if magnitude > 0.01:
        scale = 43.0 / magnitude
        end_x = center_x + right * scale
        end_y = center_y - forward * scale
        draw.line(
            scaled((center_x, center_y, end_x, end_y)),
            fill=(255, 220, 0),
            width=6 * render_scale,
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
                scaled((end_x, end_y)),
                scaled((base_x + wing_x, base_y + wing_y)),
                scaled((base_x - wing_x, base_y - wing_y)),
            ],
            fill=(255, 220, 0),
        )
    else:
        draw.ellipse(
            scaled((center_x - 4, center_y - 4, center_x + 4, center_y + 4)),
            fill=(180, 180, 180),
        )
    canvas = canvas.resize(VIDEO_SIZE, Image.Resampling.LANCZOS)
    # Camera images are fitted after HUD antialiasing, preserving the primary
    # input's existing display interpolation and its exact saved source bytes.
    _paste_view(canvas, source, scaled((32, 112, 756, 672), VIDEO_SCALE),
                depth=image_source == "fpv_depth")
    _paste_view(canvas, top_source, scaled((804, 92, 1248, 476), VIDEO_SCALE),
                resampling=Image.Resampling.LANCZOS)
    canvas.save(target, format="PNG")


def render_policy_input_video(
    spool_dir: Path, output_path: Path, *, fps: float = 20.0,
    metadata_path: Path | None = None,
) -> dict:
    """Render the four-part dashboard into an H.264 MP4 on local storage."""
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
            top_source = (
                spool_dir / "frames" / frame.top_image_filename
                if frame.top_image_filename is not None else None
            )
            _annotate_frame(
                spool_dir / "frames" / frame.image_filename,
                rendered / f"{index:06d}.png", frame,
                image_source=image_source, top_source=top_source,
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
            "-crf", str(VIDEO_CRF),
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
        "layout": "model_input_top_pointer_action/v1",
        "video_size": list(VIDEO_SIZE),
        "hud_supersampling": HUD_SUPERSAMPLING,
        "top_resampling": "lanczos",
        "h264_crf": VIDEO_CRF,
        "top_frame_count": sum(frame.top_image_filename is not None for frame in frames),
        "pointer_frame": "body: forward up, right right; direction only",
        "action_units": ["m/s", "m/s", "rad/s"],
        "displayed_action": "clipped policy output scaled by runtime action limits",
        "timing": "inference completion timestamps; one composite frame per recorded inference",
    }
    metadata_path = metadata_path or output_path.with_name(
        "policy_input_video.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**metadata, "metadata_filename": metadata_path.name}
