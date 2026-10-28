#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOURCE_ROOT"
command -v rsync >/dev/null || { echo "ERROR: rsync is required" >&2; exit 1; }

# Validate the whole collection batch before opening any backup destination.
BACKUP_ROOT="$(python3 - <<'PY'
import json
from pathlib import Path
import sys

from uav_ml.tools.artifact_publish import require_nas_root

root = require_nas_root()
datasets = Path("datasets")
if not datasets.is_dir():
    raise SystemExit("ERROR: local datasets directory is missing")
collections = sorted(datasets.iterdir())
if not collections:
    raise SystemExit("ERROR: no datasets are available for backup")
for dataset in collections:
    if dataset.is_symlink() or not dataset.is_dir():
        raise SystemExit(f"ERROR: expected a local dataset directory: {dataset}")
    manifest = json.loads((dataset / "collection_manifest.json").read_text())
    validation = json.loads((dataset / "collection_validation.json").read_text())
    if manifest.get("status") != "complete" or validation.get("valid") is not True:
        raise SystemExit(f"ERROR: dataset is incomplete or has not passed validation: {dataset}")
    print(f"Completed and validated: {dataset}", file=sys.stderr)
print(root)
PY
)"

printf 'Source: %s/datasets\nDestination: %s/datasets\n' "$SOURCE_ROOT" "$BACKUP_ROOT"
# Existing NAS files are preserved. Missing files can be filled in on a rerun.
rsync -rt --omit-dir-times --ignore-existing --dry-run --stats \
    "$SOURCE_ROOT/datasets" "$BACKUP_ROOT/"
read -r -p "Back up all completed datasets to NAS? Type yes: " ANSWER
if [[ "$ANSWER" != "yes" ]]; then
    echo "Backup cancelled."
    exit 0
fi

# Recheck the mount immediately before copying; never delete local data.
python3 - "$BACKUP_ROOT" <<'PY'
import sys
from uav_ml.tools.artifact_publish import require_nas_root
require_nas_root(sys.argv[1])
PY
rsync -rt --omit-dir-times --ignore-existing \
    "$SOURCE_ROOT/datasets" "$BACKUP_ROOT/"

# This is a read-only comparison. --delete only reports NAS-only paths here.
DIFFERENCES="$(rsync -rcn --delete --itemize-changes \
    "$SOURCE_ROOT/datasets/" "$BACKUP_ROOT/datasets/")"
if [[ -n "$DIFFERENCES" ]]; then
    printf 'ERROR: backup content differs; all existing files were preserved:\n%s\n' "$DIFFERENCES" >&2
    exit 1
fi
echo "Dataset backup verified by checksums. Local datasets were preserved."
