"""Publish completed local experiment artifacts to a NAS safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


MANIFEST_FILENAME = "transfer_manifest.json"
MANIFEST_SCHEMA = "uav_artifact_transfer/v1"
DEFAULT_NAS_ROOT = Path.home() / "NAS/614/614420090_ChenHao/uav-project"
NETWORK_FILESYSTEMS = {
    "nfs", "nfs4", "cifs", "smb3", "ceph", "glusterfs",
    "fuse.raidrive", "fuse.rclone", "fuse.sshfs", "fuse.davfs", "davfs",
}


def nas_project_root(value: str | Path | None = None) -> Path:
    """Resolve the shared NAS project root without creating directories."""
    selected = value if value is not None else os.environ.get("UAV_NAS_ROOT", DEFAULT_NAS_ROOT)
    if not str(selected).strip():
        raise ValueError("NAS project root cannot be empty")
    return Path(selected).expanduser().resolve()


def require_nas_root(value: str | Path | None = None) -> Path:
    """Reject absent, local, or unwritable NAS roots before any output is opened."""
    root = nas_project_root(value)
    if not root.is_dir():
        raise FileNotFoundError(f"NAS project root is unavailable: {root}")
    result = subprocess.run(
        ["findmnt", "--noheadings", "--output", "FSTYPE", "--target", str(root)],
        capture_output=True, text=True, timeout=5, check=False,
    )
    filesystem = result.stdout.strip()
    if result.returncode or filesystem not in NETWORK_FILESYSTEMS:
        raise RuntimeError(
            f"NAS project root is not on a mounted network filesystem: {root} "
            f"(filesystem={filesystem or 'unknown'}; {result.stderr.strip()})"
        )
    if not os.access(root, os.W_OK | os.X_OK):
        raise PermissionError(f"NAS project root is not writable: {root}")
    return root


def nas_output_path(value: str | Path, *, nas_root: Path | None = None) -> Path:
    """Require every explicit or default output to remain inside the mounted NAS."""
    root = require_nas_root(nas_root)
    output = Path(value).expanduser().resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError(f"output must be a directory inside NAS project root {root}: {output}")
    return output


def stored_experiment_path(
    value: str | Path, project_root: Path, *, artifact_root: Path | None = None,
) -> Path:
    """Resolve archived experiment references without changing checkpoint metadata."""
    project = project_root.resolve()
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else project / path).resolve()
    try:
        relative = path.relative_to(project / "artifacts/experiments")
    except ValueError:
        pass
    else:
        if relative.parts and relative.parts[0] == "bc_baseline":
            return path
        path = project / "experiments" / relative
    try:
        relative = path.relative_to(project / "experiments")
    except ValueError:
        return path
    archived = nas_project_root(artifact_root) / "experiments" / relative
    return archived if archived.is_file() else path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _copy_verified(source: Path, destination: Path) -> list[dict]:
    records = []
    for path in _files(source):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = _sha256(path)
        if not target.is_file() or _sha256(target) != source_hash:
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copy2(path, temporary)
            temporary.replace(target)
        target_hash = _sha256(target)
        if target_hash != source_hash:
            raise RuntimeError(f"checksum mismatch after copy: {relative}")
        records.append({
            "relative_path": str(relative),
            "size_bytes": path.stat().st_size,
            "sha256": source_hash,
        })
    return records


def _safe_delete_source(source: Path) -> None:
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    if source in (cwd, home) or source == source.parent:
        raise ValueError(f"refusing to delete unsafe artifact source: {source}")
    shutil.rmtree(source)


def publish_artifact_tree(
    source: Path,
    nas_root: Path,
    category: str,
    *,
    delete_local: bool,
) -> Path:
    """Copy a completed artifact tree, verify it, then optionally delete local."""
    source = source.expanduser().resolve()
    nas_root = nas_root.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"artifact source directory is missing: {source}")
    if not category or Path(category).is_absolute() or ".." in Path(category).parts:
        raise ValueError("artifact category must be a relative path")
    destination = nas_root / category / source.name
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists():
        raise FileExistsError(f"NAS artifact destination already exists: {destination}")
    partial.mkdir(parents=True, exist_ok=True)
    records = _copy_verified(source, partial)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_name": source.name,
        "category": category,
        "status": "verified",
        "files": records,
    }
    (partial / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(destination)
    if delete_local:
        _safe_delete_source(source)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a completed local artifact tree to NAS with SHA-256 verification."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--nas-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--delete-local", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination = publish_artifact_tree(
        args.source, args.nas_root, args.category,
        delete_local=args.delete_local,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
