#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || -z $1 || $1 == /* || $1 == */ || $1 == *//* || $1 == '.' || $1 == '..' || $1 == */../* || $1 == ../* || $1 == */.. || $1 == */./* || $1 == ./* ]]; then
    echo "Usage: $0 <scene_id>" >&2
    echo "Error: provide exactly one valid scene ID or scene path." >&2
    exit 1
fi

scene_id=$1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
scene_dir="$repo_root/data/scene_graphs/$scene_id"
archive_name="${scene_id##*/}_scene_graphs.zip"
archive_path="$repo_root/$archive_name"

if [[ ! -d $scene_dir ]]; then
    echo "Error: scene results directory does not exist: $scene_dir" >&2
    exit 1
fi

json_files=()
while IFS= read -r -d '' json_file; do
    json_file=${json_file#./}
    json_files+=("$json_file")
done < <(cd -- "$scene_dir" && find . -type f -name '*.json' -print0)

if [[ ${#json_files[@]} -eq 0 ]]; then
    echo "Error: no .json files found in scene directory: $scene_dir" >&2
    exit 1
fi

# Remove an older archive first so it cannot retain stale or non-JSON entries.
rm -f -- "$archive_path"

(
    if command -v zip >/dev/null 2>&1; then
        cd -- "$scene_dir"
        zip -q "$archive_path" -- "${json_files[@]}"
    elif command -v python3 >/dev/null 2>&1; then
        python3 - "$scene_dir" "$archive_path" "${json_files[@]}" <<'PY'
import os
import sys
import zipfile

scene_dir, archive_path, *json_files = sys.argv[1:]
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for relative_path in json_files:
        archive.write(
            os.path.join(scene_dir, relative_path),
            arcname=relative_path,
        )
PY
    else
        echo "Error: neither 'zip' nor 'python3' is available to create the archive." >&2
        exit 1
    fi
)

echo "Created archive: $archive_path"
