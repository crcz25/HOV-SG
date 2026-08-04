#!/usr/bin/env bash
# Convert every HM3DSEM validation walk into a ROS 2 bag and place its
# validated semantic-instance legend beside metadata.yaml in the bag directory.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/convert_all_hm3dsem_val_to_rosbags.sh [options]

Options:
  --walks-root PATH  HM3DSEM validation walks directory.
                     Default: data/hm3dsem_walks/val
  --bags-root PATH   Parent directory for ROS 2 bag directories.
                     Default: data/bags
  --fps FPS          Synthetic FPS passed to the converter. Default: 1
  --overwrite        Replace existing per-scene bags and CSV legends.
  -h, --help         Show this help.

For each <scene> under --walks-root, this creates:
  <bags-root>/<scene>/                 ROS 2 sqlite3 bag and metadata.yaml
  <bags-root>/<scene>/semantic_label_map.csv

Without --overwrite, a valid existing bag is reused and its CSV legend is
generated or refreshed. This makes interrupted batch runs resumable.
EOF
}

walks_root="data/hm3dsem_walks/val"
bags_root="data/bags"
fps="1"
overwrite=0

while (($#)); do
    case "$1" in
        --walks-root)
            walks_root="$2"
            shift 2
            ;;
        --bags-root)
            bags_root="$2"
            shift 2
            ;;
        --fps)
            fps="$2"
            shift 2
            ;;
        --overwrite)
            overwrite=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$walks_root" ]]; then
    echo "Walks root does not exist: $walks_root" >&2
    exit 1
fi
if ! python - "$fps" <<'PY'
import math
import sys

try:
    fps = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(fps) and fps > 0.0 else 1)
PY
then
    echo "--fps must be a finite positive number: $fps" >&2
    exit 2
fi

mkdir -p "$bags_root"
mapfile -d '' scenes < <(find "$walks_root" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
if ((${#scenes[@]} == 0)); then
    echo "No scene directories found under $walks_root" >&2
    exit 1
fi

converter="scripts/convert_hm3dsem_walks_to_rosbags.py"
legend_generator="scripts/generate_hm3dsem_semantic_label_csv.py"
if [[ ! -f "$converter" || ! -f "$legend_generator" ]]; then
    echo "Run this script from the repository root; converter scripts are missing." >&2
    exit 1
fi

for scene_dir in "${scenes[@]}"; do
    scene_name="$(basename "$scene_dir")"
    bag_dir="$bags_root/$scene_name"
    echo "=== $scene_name ==="

    csv_args=(python "$legend_generator" "$scene_dir" --output "$bag_dir/semantic_label_map.csv")
    if ((overwrite)); then
        bag_args=(python "$converter" "$scene_dir" --fps "$fps" --ros2-out "$bag_dir" --overwrite)
        csv_args+=(--overwrite)
        "${bag_args[@]}"
    elif [[ -e "$bag_dir" ]]; then
        if [[ ! -d "$bag_dir" ]] || [[ ! -f "$bag_dir/metadata.yaml" ]] || ! compgen -G "$bag_dir/*.db3" >/dev/null; then
            echo "Existing output is not a complete ROS 2 bag: $bag_dir. Pass --overwrite to replace it." >&2
            exit 1
        fi
        echo "Reusing existing ROS 2 bag: $bag_dir"
        if [[ -e "$bag_dir/semantic_label_map.csv" ]]; then
            csv_args+=(--overwrite)
        fi
    else
        bag_args=(python "$converter" "$scene_dir" --fps "$fps" --ros2-out "$bag_dir")
        "${bag_args[@]}"
    fi
    "${csv_args[@]}"
done

echo "Completed ${#scenes[@]} scene(s). ROS 2 bags and semantic_label_map.csv files are under $bags_root."
