#!/usr/bin/env python3
"""Create a validated semantic-instance label legend from an HM3DSEM walk."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


HEX_COLOR = re.compile(r"^[0-9A-Fa-f]{6}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write an HM3DSEM semantic-instance ID, GT colour, and class-label CSV for one walk scene."
    )
    parser.add_argument("scene_dir", type=Path, help="HM3DSEM walk scene directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path (default: <scene_dir>/semantic_label_map.csv)",
    )
    parser.add_argument(
        "--semantic-annotations",
        type=Path,
        help="Authoritative <scene>.semantic.txt GT file; auto-detected under the sibling data/hm3d tree when available.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing CSV")
    return parser.parse_args()


def load_gt_objects(scene_info_path: Path) -> dict[int, dict[str, object]]:
    try:
        scene_info = json.loads(scene_info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read ground-truth scene metadata {scene_info_path}: {exc}") from exc
    objects = scene_info.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"Ground-truth scene metadata has no objects list: {scene_info_path}")

    result: dict[int, dict[str, object]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid non-object entry in {scene_info_path}")
        try:
            label = int(obj["id"])
            category = str(obj["category"])
            color = str(obj["hex"]).upper().removeprefix("#")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed ground-truth object entry: {obj!r}") from exc
        if label <= 0 or label in result:
            raise ValueError(f"Ground-truth object IDs must be unique positive integers; found {label}")
        if not category or not HEX_COLOR.fullmatch(color):
            raise ValueError(f"Invalid class label or six-digit hex colour for GT object {label}")
        result[label] = {
            "class_label": category,
            "hex_color": color,
            "region_id": obj.get("region_id", ""),
            "floor_id": obj.get("floor_id", ""),
            "gt_source": "walk_scene_info",
        }
    return result


def discover_semantic_annotations(scene_dir: Path) -> Path | None:
    """Locate the raw HM3D semantic annotation beside a standard walks tree."""
    try:
        data_root = scene_dir.parents[2]
        split = scene_dir.parent.name
        scene_name = scene_dir.name.split("-", maxsplit=1)[1]
    except (IndexError, ValueError):
        return None
    candidate = data_root / "hm3d" / split / scene_dir.name / f"{scene_name}.semantic.txt"
    return candidate if candidate.is_file() else None


def merge_authoritative_annotations(
    gt_objects: dict[int, dict[str, object]], annotation_path: Path | None
) -> dict[int, dict[str, object]]:
    """Fill omissions in scene_info and reject conflicts with source HM3D GT."""
    if annotation_path is None:
        return gt_objects
    try:
        with annotation_path.open(newline="") as file:
            reader = csv.reader(file)
            next(reader)  # "HM3D Semantic Annotations"
            annotations = list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"Could not read semantic annotations {annotation_path}: {exc}") from exc
    for row in annotations:
        if len(row) != 4:
            raise ValueError(f"Malformed semantic annotation row in {annotation_path}: {row!r}")
        label_text, color, category, region_id = row
        try:
            label = int(label_text)
        except ValueError as exc:
            raise ValueError(f"Invalid semantic ID {label_text!r} in {annotation_path}") from exc
        color = color.upper().removeprefix("#")
        if label <= 0 or not category or not HEX_COLOR.fullmatch(color):
            raise ValueError(f"Invalid semantic annotation for ID {label} in {annotation_path}")
        metadata = gt_objects.get(label)
        if metadata is None:
            gt_objects[label] = {
                "class_label": category,
                "hex_color": color,
                "region_id": region_id,
                "floor_id": "",
                "gt_source": "raw_semantic_annotation_fallback",
            }
        elif metadata["class_label"] != category or metadata["hex_color"] != color:
            raise ValueError(f"GT conflict for semantic ID {label} between scene_info.json and {annotation_path}")
    return gt_objects


def count_semantic_labels(semantic_dir: Path) -> tuple[Counter[int], Counter[int], int]:
    files = sorted(semantic_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No semantic .npy frames found in {semantic_dir}")

    pixel_counts: Counter[int] = Counter()
    frame_counts: Counter[int] = Counter()
    for path in tqdm(files, desc="Validating semantic labels", unit="frame"):
        try:
            image = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read semantic frame {path}: {exc}") from exc
        if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
            raise ValueError(f"Semantic frame must be a 2-D integer array: {path}")
        labels, counts = np.unique(image, return_counts=True)
        for label_value, count in zip(labels, counts):
            label = int(label_value)
            if label < 0:
                raise ValueError(f"Semantic frame contains a negative label ({label}): {path}")
            pixel_counts[label] += int(count)
            frame_counts[label] += 1
    return pixel_counts, frame_counts, len(files)


def write_csv(
    output: Path,
    gt_objects: dict[int, dict[str, object]],
    pixel_counts: Counter[int],
    frame_counts: Counter[int],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "semantic_id",
                "hex_color",
                "red",
                "green",
                "blue",
                "class_label",
                "region_id",
                "floor_id",
                "gt_source",
                "pixel_count_in_walk",
                "frame_count_in_walk",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "semantic_id": 0,
                "hex_color": "#000000",
                "red": 0,
                "green": 0,
                "blue": 0,
                "class_label": "background/unlabeled",
                "region_id": "",
                "floor_id": "",
                "gt_source": "reserved_background",
                "pixel_count_in_walk": pixel_counts[0],
                "frame_count_in_walk": frame_counts[0],
            }
        )
        labels = sorted(set(gt_objects) | (set(pixel_counts) - {0}))
        for label in labels:
            metadata = gt_objects.get(label)
            if metadata is None:
                # Do not invent a class or colour for labels that Habitat emits
                # but which do not occur in either supplied GT source.
                writer.writerow(
                    {
                        "semantic_id": label,
                        "hex_color": "",
                        "red": "",
                        "green": "",
                        "blue": "",
                        "class_label": "unmapped/no GT annotation",
                        "region_id": "",
                        "floor_id": "",
                        "gt_source": "semantic_unmapped_no_gt_annotation",
                        "pixel_count_in_walk": pixel_counts[label],
                        "frame_count_in_walk": frame_counts[label],
                    }
                )
                continue
            color = str(metadata["hex_color"])
            writer.writerow(
                {
                    "semantic_id": label,
                    "hex_color": f"#{color}",
                    "red": int(color[0:2], 16),
                    "green": int(color[2:4], 16),
                    "blue": int(color[4:6], 16),
                    "class_label": metadata["class_label"],
                    "region_id": metadata["region_id"],
                    "floor_id": metadata["floor_id"],
                    "gt_source": metadata["gt_source"],
                    "pixel_count_in_walk": pixel_counts[label],
                    "frame_count_in_walk": frame_counts[label],
                }
            )


def generate_semantic_label_map(
    scene_dir: Path,
    semantic_annotations: Path | None = None,
    output: Path | None = None,
) -> Path:
    """Create a scene-local label map for a completed, validated walk.

    This reusable entry point lets dataset preparation create the mandatory CSV
    without shelling out to this CLI or duplicating GT reconciliation logic.
    """
    scene_dir = Path(scene_dir)
    scene_info = scene_dir / "scene_info.json"
    semantic_dir = scene_dir / "semantic"
    if not scene_dir.is_dir() or not scene_info.is_file() or not semantic_dir.is_dir():
        raise FileNotFoundError("Expected a walk scene directory containing scene_info.json and semantic/")
    output = Path(output) if output is not None else scene_dir / "semantic_label_map.csv"
    gt_objects = load_gt_objects(scene_info)
    annotation_path = semantic_annotations or discover_semantic_annotations(scene_dir)
    if annotation_path is not None and not annotation_path.is_file():
        raise FileNotFoundError(f"Semantic annotation file does not exist: {annotation_path}")
    gt_objects = merge_authoritative_annotations(gt_objects, annotation_path)
    pixel_counts, frame_counts, frame_total = count_semantic_labels(semantic_dir)
    write_csv(output, gt_objects, pixel_counts, frame_counts)
    observed_instances = len(set(pixel_counts) - {0})
    unmapped_labels = sorted((set(pixel_counts) - {0}) - set(gt_objects))
    print(
        f"Wrote {output}: validated {frame_total} semantic frames, "
        f"{observed_instances} observed GT instance labels, and {len(gt_objects)} GT objects"
        f" ({'with' if annotation_path else 'without'} raw semantic-annotation reconciliation)."
    )
    if unmapped_labels:
        print(
            "WARNING: semantic labels with no entry in scene_info.json or the raw semantic annotation: "
            + ", ".join(map(str, unmapped_labels))
        )
    return output


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir
    scene_info = scene_dir / "scene_info.json"
    semantic_dir = scene_dir / "semantic"
    if not scene_dir.is_dir() or not scene_info.is_file() or not semantic_dir.is_dir():
        raise FileNotFoundError("Expected a walk scene directory containing scene_info.json and semantic/")
    output = args.output or scene_dir / "semantic_label_map.csv"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    annotation_path = args.semantic_annotations or discover_semantic_annotations(scene_dir)
    if args.semantic_annotations and not args.semantic_annotations.is_file():
        raise FileNotFoundError(f"Semantic annotation file does not exist: {args.semantic_annotations}")
    generate_semantic_label_map(scene_dir, annotation_path, output)


if __name__ == "__main__":
    main()
