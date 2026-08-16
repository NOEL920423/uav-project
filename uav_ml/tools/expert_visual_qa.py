"""Build lightweight contact sheets from completed expert episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CELL_WIDTH = 320
CELL_HEIGHT = 210
IMAGE_HEIGHT = 180
CONTACT_COLUMNS = 3
CONTACT_ROWS = 3


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _indices(count: int) -> tuple[int, int, int]:
    if count < 1:
        raise ValueError("contact sheet source episode has no samples")
    return 0, count // 2, count - 1


def _nearest_available(
    rows: list[dict[str, str]], index: int, field: str
) -> dict[str, str] | None:
    candidates = [row for row in rows if row.get(field)]
    if not candidates:
        return None
    target = index + 1
    return min(candidates, key=lambda row: abs(int(row["sample_id"]) - target))


def _depth_preview(path: Path) -> Image.Image:
    with Image.open(path) as image:
        depth = np.asarray(image, dtype=np.uint16)
    valid = depth[depth > 0]
    if not valid.size:
        pixels = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(valid, (2, 98))
        scale = max(float(high - low), 1.0)
        pixels = np.clip((depth.astype(np.float32) - low) / scale, 0, 1)
        pixels = (255 * (1.0 - pixels)).astype(np.uint8)
        pixels[depth == 0] = 0
    return Image.fromarray(pixels, mode="L").convert("RGB")


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((CELL_WIDTH, IMAGE_HEIGHT))


def _cell(image: Image.Image | None, label: str) -> Image.Image:
    cell = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), (20, 24, 30))
    if image is not None:
        cell.paste(image.resize((CELL_WIDTH, IMAGE_HEIGHT)), (0, 0))
    else:
        draw = ImageDraw.Draw(cell)
        draw.text((12, 76), "stream unavailable", fill=(220, 180, 80))
    ImageDraw.Draw(cell).text((8, IMAGE_HEIGHT + 7), label, fill="white")
    return cell


def create_contact_sheet(
    dataset_root: Path,
    through_episode: int,
    source_episode: str | None = None,
) -> dict:
    """Create start/mid/near-goal FPV, observer, and depth QA previews."""
    dataset_root = dataset_root.resolve()
    if source_episode is None:
        for index in range(through_episode, 0, -1):
            candidate = f"episode_{index:06d}"
            metadata_path = dataset_root / candidate / "episode.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("success") and int(
                metadata.get("sample_count", 0)
            ):
                source_episode = candidate
                break
    if source_episode is None:
        raise ValueError("no successful episode is available for visual QA")
    episode_dir = dataset_root / source_episode
    samples = _load_rows(episode_dir / "samples.csv")
    auxiliary = _load_rows(episode_dir / "auxiliary.csv")
    selected_indices = _indices(len(samples))
    positions = ("start", "mid-flight", "near-goal")

    cells: list[Image.Image] = []
    source_paths: dict[str, list[str | None]] = {
        "fpv_rgb": [], "observer_rgb": [], "fpv_depth": []
    }
    for index, position in zip(selected_indices, positions):
        relative = samples[index]["image_path"]
        source_paths["fpv_rgb"].append(relative)
        cells.append(_cell(
            _open_rgb(dataset_root / relative), f"FPV {position}"
        ))
    for index, position in zip(selected_indices, positions):
        row = _nearest_available(auxiliary, index, "observer_rgb_path")
        relative = None if row is None else row["observer_rgb_path"]
        source_paths["observer_rgb"].append(relative)
        image = (
            None if relative is None
            else _open_rgb(dataset_root / relative)
        )
        cells.append(_cell(image, f"Observer {position}"))
    for index, position in zip(selected_indices, positions):
        row = _nearest_available(auxiliary, index, "fpv_depth_path")
        relative = None if row is None else row["fpv_depth_path"]
        source_paths["fpv_depth"].append(relative)
        image = (
            None if relative is None
            else _depth_preview(dataset_root / relative)
        )
        cells.append(_cell(image, f"Depth {position}"))

    canvas = Image.new(
        "RGB",
        (CELL_WIDTH * CONTACT_COLUMNS, CELL_HEIGHT * CONTACT_ROWS),
        (0, 0, 0),
    )
    for index, cell in enumerate(cells):
        canvas.paste(
            cell,
            (
                (index % CONTACT_COLUMNS) * CELL_WIDTH,
                (index // CONTACT_COLUMNS) * CELL_HEIGHT,
            ),
        )
    output_dir = dataset_root / "visual_qa"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"contact_sheet_{through_episode:06d}"
    output_path = output_dir / f"{stem}.jpg"
    canvas.save(output_path, format="JPEG", quality=90)
    result = {
        "through_episode": through_episode,
        "source_episode": source_episode,
        "contact_sheet": str(output_path.relative_to(dataset_root)),
        "layout": [CONTACT_COLUMNS, CONTACT_ROWS],
        "source_paths": source_paths,
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    """Generate one requested contact sheet from an existing dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--through-episode", required=True, type=int)
    parser.add_argument("--source-episode")
    args = parser.parse_args()
    result = create_contact_sheet(
        Path(args.dataset), args.through_episode, args.source_episode
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
