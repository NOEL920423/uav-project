"""Offline video of BC policy inputs, the TOP camera, and body actions."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from uav_ml.inference.bc_flight_contract import ACTION_LIMITS, body_action_to_ned


VIDEO_METADATA_SCHEMA = "uav_bc_policy_input_video/v1"
FRAME_MANIFEST_FILENAME = "frames.jsonl"
VIDEO_SCALE = 2
VIDEO_SIZE = (1280 * VIDEO_SCALE, 720 * VIDEO_SCALE)
HUD_SUPERSAMPLING = 2
VIDEO_CRF = 16
ACTION_HISTORY_LENGTH = 10
POINTER_STYLE = "needle"
POINTER_STYLES = ("arrow", "triangle", "chevron", "needle")
POINTER_SPEED_FULL_SCALE_MPS = math.hypot(*ACTION_LIMITS[:2])
YAW_ARC_MAX_SWEEP_DEG = 180.0
NEGATIVE_FORWARD_COLOR = "#ff5353"
YAW_COLOR = "#36d5f4"


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


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False, monospace: bool = False):
    family = "DejaVuSansMono" if monospace else "DejaVuSans"
    name = f"{family}-Bold.ttf" if bold else f"{family}.ttf"
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
    action_history: list[PolicyInputFrame] | None = None,
    pointer_style: str = POINTER_STYLE,
) -> None:
    """Compose camera views and commanded translation/yaw visualizations."""
    if pointer_style not in POINTER_STYLES:
        raise ValueError(f"unsupported pointer style: {pointer_style}")
    render_scale = VIDEO_SCALE * HUD_SUPERSAMPLING

    def scaled(values, factor=render_scale):
        return tuple(int(value * factor) for value in values)

    canvas = Image.new("RGB", tuple(value * HUD_SUPERSAMPLING for value in VIDEO_SIZE), "#070d16")
    draw = ImageDraw.Draw(canvas)
    for box in ((16, 16, 772, 704), (788, 16, 1264, 400), (788, 416, 1264, 704)):
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

    draw.text(scaled((804, 432)), "RECENT ACTIONS", font=_font(20 * render_scale, True), fill="#edf3fb")
    draw.text(scaled((804, 468)), "[forward m/s, right m/s, yaw rad/s]",
              font=_font(12 * render_scale), fill="#9bb1c9")
    history = (action_history or [frame])[-ACTION_HISTORY_LENGTH:]
    for index, entry in enumerate(history):
        forward, right, _, yaw_rate = body_action_to_ned(entry.action_body, 0.0)
        y = 496 + index * 18
        current = entry.inference_count == frame.inference_count
        if current:
            draw.rounded_rectangle(scaled((800, y - 2, 1092, y + 16)),
                                   radius=3 * render_scale, fill="#173d4a")
        font = _font(14 * render_scale, bold=current, monospace=True)
        color = "#36d5f4" if current else "#9bb1c9"
        label = f"step{entry.inference_count}: [{forward:.2f}, {right:.2f}, {yaw_rate:.2f}]"
        if forward < 0.0:
            x = 804 * render_scale
            for text, fill in (
                (f"step{entry.inference_count}: [", color),
                (f"{forward:.2f}", NEGATIVE_FORWARD_COLOR),
                (f", {right:.2f}, {yaw_rate:.2f}]", color),
            ):
                draw.text((x, y * render_scale), text, font=font, fill=fill)
                x += draw.textlength(text, font=font)
        else:
            draw.text(scaled((804, y)), label, font=font, fill=color)
    forward, right, _, yaw_rate = body_action_to_ned(frame.action_body, 0.0)
    center_x, center_y, radius = 1180, 585, 54
    draw.ellipse(scaled((center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius)),
                 outline="#9bb1c9", width=2 * render_scale)
    draw.line(scaled((center_x - 42, center_y, center_x + 42, center_y)),
              fill="#36516b", width=render_scale)
    draw.line(scaled((center_x, center_y - 42, center_x, center_y + 42)),
              fill="#36516b", width=render_scale)
    for label, x, y in (("F", center_x, center_y - 76), ("B", center_x, center_y + 76),
                        ("L", center_x - 76, center_y), ("R", center_x + 76, center_y)):
        draw.text(scaled((x, y)), label, font=_font(15 * render_scale), fill="#edf3fb", anchor="mm")
    magnitude = math.hypot(forward, right)
    speed_fraction = min(magnitude / POINTER_SPEED_FULL_SCALE_MPS, 1.0)
    draw.text(scaled((center_x, 686)), f"CMD {magnitude:.2f} m/s",
              font=_font(11 * render_scale, True), fill=YAW_COLOR, anchor="mm")

    yaw_radius = 67
    yaw_box = scaled((center_x - yaw_radius, center_y - yaw_radius,
                      center_x + yaw_radius, center_y + yaw_radius))
    draw.ellipse(yaw_box, outline="#36516b", width=2 * render_scale)
    yaw_label = "YAW 0.00 rad/s"
    if abs(yaw_rate) > 0.005:
        sign = 1.0 if yaw_rate > 0.0 else -1.0
        sweep = YAW_ARC_MAX_SWEEP_DEG * min(abs(yaw_rate) / ACTION_LIMITS[2], 1.0)
        end_angle = -90.0 + sign * sweep
        draw.arc(yaw_box, start=min(-90.0, end_angle), end=max(-90.0, end_angle),
                 fill=YAW_COLOR, width=3 * render_scale)
        angle = math.radians(end_angle)
        tip_x = center_x + yaw_radius * math.cos(angle)
        tip_y = center_y + yaw_radius * math.sin(angle)
        tangent_x, tangent_y = -sign * math.sin(angle), sign * math.cos(angle)
        head_length = min(8.0, yaw_radius * math.radians(sweep))
        half_width = head_length * 0.5
        draw.polygon([
            scaled((tip_x, tip_y)),
            scaled((tip_x - head_length * tangent_x - half_width * tangent_y,
                    tip_y - head_length * tangent_y + half_width * tangent_x)),
            scaled((tip_x - head_length * tangent_x + half_width * tangent_y,
                    tip_y - head_length * tangent_y - half_width * tangent_x)),
        ], fill=YAW_COLOR)
        yaw_label = f"{'RIGHT' if yaw_rate > 0.0 else 'LEFT'} {yaw_rate:+.2f} rad/s"
    draw.text(scaled((center_x, 480)), yaw_label,
              font=_font(10 * render_scale, True), fill=YAW_COLOR, anchor="mm")

    if magnitude > 0.01:
        scale = 43.0 * speed_fraction / magnitude
        end_x = center_x + right * scale
        end_y = center_y - forward * scale
        angle_x = end_x - center_x
        angle_y = end_y - center_y
        length = (angle_x**2 + angle_y**2) ** 0.5
        direction_x, direction_y = angle_x / length, angle_y / length
        side_x, side_y = -direction_y, direction_x
        # Keep slow commands visible while preserving the linear length scale.
        width_fraction = max(speed_fraction, 0.35)

        def pointer_point(along, across=0.0):
            return scaled((center_x + speed_fraction * along * direction_x
                           + width_fraction * across * side_x,
                           center_y + speed_fraction * along * direction_y
                           + width_fraction * across * side_y))

        if pointer_style == "arrow":
            draw.line(scaled((center_x, center_y, end_x, end_y)),
                      fill=(255, 220, 0),
                      width=max(render_scale, round(6 * render_scale * speed_fraction)))
            draw.polygon([pointer_point(43), pointer_point(32, 5), pointer_point(32, -5)],
                         fill=(255, 220, 0))
        elif pointer_style == "triangle":
            draw.polygon([pointer_point(43), pointer_point(-14, 18), pointer_point(-14, -18)],
                         fill=(255, 220, 0))
        elif pointer_style == "chevron":
            draw.polygon([pointer_point(43), pointer_point(-16, 23),
                          pointer_point(3), pointer_point(-16, -23)], fill=(255, 220, 0))
        else:
            draw.polygon([pointer_point(-43), pointer_point(0, 8), pointer_point(0, -8)],
                         fill="#687c91")
            draw.polygon([pointer_point(43), pointer_point(0, 8), pointer_point(0, -8)],
                         fill=(255, 220, 0))
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
    _paste_view(canvas, top_source, scaled((804, 92, 1248, 384), VIDEO_SCALE),
                resampling=Image.Resampling.LANCZOS)
    canvas.save(target, format="PNG")


def render_policy_input_video(
    spool_dir: Path, output_path: Path, *, fps: float = 20.0,
    metadata_path: Path | None = None,
    pointer_style: str = POINTER_STYLE,
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
                action_history=frames[max(0, index - ACTION_HISTORY_LENGTH + 1):index + 1],
                pointer_style=pointer_style,
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
        "pointer_frame": "body: forward up, right right; length encodes horizontal command speed",
        "pointer_style": pointer_style,
        "pointer_length_scaling": "linear in horizontal commanded speed",
        "pointer_speed_full_scale_mps": POINTER_SPEED_FULL_SCALE_MPS,
        "yaw_indicator": "outer arc: positive clockwise/right; negative counterclockwise/left",
        "yaw_arc_full_scale_radps": ACTION_LIMITS[2],
        "yaw_arc_max_sweep_deg": YAW_ARC_MAX_SWEEP_DEG,
        "yaw_display_deadband_radps": 0.005,
        "negative_forward_color": NEGATIVE_FORWARD_COLOR,
        "action_history_length": ACTION_HISTORY_LENGTH,
        "action_history_order": "oldest to newest; current step highlighted",
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
