"""Publish completed local experiment artifacts to a NAS safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


MANIFEST_FILENAME = "transfer_manifest.json"
MANIFEST_SCHEMA = "uav_artifact_transfer/v1"


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
