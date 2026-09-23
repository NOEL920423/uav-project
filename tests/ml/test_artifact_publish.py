"""Unit tests for local artifact publication and policy input capture."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import tempfile

from PIL import Image

from uav_ml.tools.artifact_publish import (
    MANIFEST_FILENAME,
    publish_artifact_tree,
)
from uav_ml.tools.bc_flight_video import (
    FRAME_MANIFEST_FILENAME,
    capture_policy_input_frame,
)


def _jpeg_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 3), color=(1, 2, 3)).save(stream, format="JPEG")
    return stream.getvalue()


def test_capture_preserves_one_exact_inference_input() -> None:
    with tempfile.TemporaryDirectory() as directory:
        spool = Path(directory) / "spool"
        image = _jpeg_bytes()
        capture_policy_input_frame(
            spool,
            image_bytes=image,
            image_source="fpv_rgb",
            inference_count=7,
            timestamp_ns=123,
            action_body=(0.2, -0.3, 0.4),
        )
        row = json.loads((spool / FRAME_MANIFEST_FILENAME).read_text())
        assert row["inference_count"] == 7
        assert row["action_body"] == [0.2, -0.3, 0.4]
        assert (spool / "frames" / row["image_filename"]).read_bytes() == image


def test_publish_verifies_before_local_deletion() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "local" / "run_001"
        source.mkdir(parents=True)
        (source / "result.json").write_text('{"ok":true}\n', encoding="utf-8")
        destination = publish_artifact_tree(
            source, root / "nas", "evaluations/bc_flight", delete_local=True
        )
        assert not source.exists()
        assert (destination / "result.json").is_file()
        manifest = json.loads((destination / MANIFEST_FILENAME).read_text())
        assert manifest["status"] == "verified"
        assert manifest["files"][0]["relative_path"] == "result.json"
