"""Unit tests for local artifact publication and policy input capture."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import tempfile
import os
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from PIL import Image

from uav_ml.tools.artifact_publish import (
    MANIFEST_FILENAME,
    publish_artifact_tree,
    nas_project_root,
    nas_output_path,
    require_nas_root,
    stored_experiment_path,
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


def test_nas_guard_rejects_missing_local_and_read_only_roots(tmp_path) -> None:
    with patch("uav_ml.tools.artifact_publish.subprocess.run") as probe:
        with pytest.raises(FileNotFoundError, match="NAS project root is unavailable"):
            require_nas_root(tmp_path / "absent")
        probe.assert_not_called()
        probe.return_value = SimpleNamespace(returncode=0, stdout="ext4\n", stderr="")
        with pytest.raises(RuntimeError, match="mounted network filesystem"):
            require_nas_root(tmp_path)
        probe.return_value.stdout = "fuse.raidrive\n"
        with patch("uav_ml.tools.artifact_publish.os.access", return_value=False):
            with pytest.raises(PermissionError, match="not writable"):
                require_nas_root(tmp_path)
    assert not (tmp_path / "absent").exists()


def test_outputs_reject_local_paths_and_symlink_escapes(tmp_path, monkeypatch) -> None:
    nas = tmp_path / "nas"
    nas.mkdir()
    local = tmp_path / "local"
    local.mkdir()
    (nas / "escape").symlink_to(local, target_is_directory=True)
    monkeypatch.setenv("UAV_NAS_ROOT", str(nas))
    assert nas_project_root() == nas
    with patch("uav_ml.tools.artifact_publish.subprocess.run", return_value=SimpleNamespace(
        returncode=0, stdout="fuse.raidrive\n", stderr="",
    )):
        assert nas_output_path(nas / "experiments/run") == nas / "experiments/run"
        for output in (local / "run", nas / "escape/run", nas):
            with pytest.raises(ValueError, match="inside NAS project root"):
                nas_output_path(output)
    assert not (local / "run").exists()
    assert not (nas / "experiments").exists()


def test_archived_reference_prefers_nas_without_changing_local_copy(tmp_path) -> None:
    project, nas = tmp_path / "project", tmp_path / "nas"
    original = project / "experiments/bc/dataset/top/run/best.pt"
    archived = nas / original.relative_to(project)
    for path, content in ((original, b"original"), (archived, b"archived")):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    assert stored_experiment_path(original, project, artifact_root=nas) == archived
    assert original.read_bytes() == b"original"


def test_clean_wrapper_preserves_nas_root_and_ros_log_destination(tmp_path) -> None:
    wrapper = Path(__file__).resolve().parents[2] / "uav"
    source = wrapper.read_text()
    start = source.index("clean_environment_command() {")
    end = source.index("\n}\n", start) + 3
    function = source[start:end]
    stub = tmp_path / "environment.sh"
    stub.write_text('printf "%s\\n" "$UAV_NAS_ROOT" "$ROS_LOG_DIR"\n')
    nas = str(tmp_path / "nas")
    logs = str(tmp_path / "nas/evaluations/run/ros_logs")
    result = subprocess.run(
        ["bash", "-c", function + '\nSCRIPT_PATH="$1"\nclean_environment_command fixture\n',
         "nas-wrapper-test", str(stub)],
        env={**os.environ, "UAV_NAS_ROOT": nas, "ROS_LOG_DIR": logs},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [nas, logs]


@pytest.mark.parametrize("complete,conflict", [(True, False), (False, False), (True, True)])
def test_dataset_backup_requires_completion_preserves_conflicts_and_sources(
    tmp_path, complete, conflict,
) -> None:
    if shutil.which("rsync") is None:
        pytest.skip("rsync is not installed")
    project, nas = tmp_path / "project", tmp_path / "nas"
    tools = project / "tools"
    tools.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[2]
    shutil.copy2(source_root / "tools/backup.sh", tools / "backup.sh")
    (project / "uav_ml").symlink_to(source_root / "uav_ml", target_is_directory=True)
    dataset = project / "datasets/collection"
    dataset.mkdir(parents=True)
    (dataset / "collection_manifest.json").write_text(json.dumps({
        "status": "complete" if complete else "collecting",
    }))
    (dataset / "collection_validation.json").write_text('{"valid":true}')
    (dataset / "sample.jpg").write_bytes(b"new evidence")
    nas.mkdir()
    if conflict:
        destination = nas / "datasets/collection"
        destination.mkdir(parents=True)
        (destination / "sample.jpg").write_bytes(b"old evidence")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    probe = binaries / "findmnt"
    probe.write_text("#!/bin/sh\nprintf 'fuse.raidrive\\n'\n")
    probe.chmod(0o755)
    result = subprocess.run(
        ["bash", str(tools / "backup.sh")], input="yes\n", text=True,
        capture_output=True, timeout=20,
        env={**os.environ, "UAV_NAS_ROOT":str(nas), "PATH":f"{binaries}:{os.environ['PATH']}"},
    )
    assert (dataset / "sample.jpg").read_bytes() == b"new evidence"
    if not complete:
        assert result.returncode != 0
        assert "incomplete" in result.stderr
        assert not (nas / "datasets").exists()
    elif conflict:
        assert result.returncode != 0
        assert "backup content differs" in result.stderr
        assert (nas / "datasets/collection/sample.jpg").read_bytes() == b"old evidence"
    else:
        assert result.returncode == 0, result.stderr
        assert "verified by checksums" in result.stdout
        assert (nas / "datasets/collection/sample.jpg").read_bytes() == b"new evidence"
    assert not (nas / "experiments").exists()
    assert not (nas / "evaluations").exists()
