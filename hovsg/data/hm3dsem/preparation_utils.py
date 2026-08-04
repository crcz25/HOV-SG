"""Shared discovery, validation, and floor-selection helpers for HM3DSEM walks.

The helpers deliberately depend only on the on-disk dataset contract.  They do
not contain a list of scene identifiers, split names, image dimensions, or
repository-relative paths.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ScenePaths:
    """Resolved source and output paths for one scene."""

    scene_id: str
    split: str
    raw_scene_dir: Path
    scene_name: str
    basis_mesh: Path
    semantic_mesh: Path
    semantic_annotations: Path
    pose_file: Path | None
    walk_scene_dir: Path


@dataclass(frozen=True)
class FloorBounds:
    """Inclusive world-Y bounds for a selected zero-based floor."""

    floor_id: int
    lower: float
    upper: float
    source: str


@dataclass(frozen=True)
class FramePaths:
    """One RGB-D-semantic-pose observation, keyed by its common frame stem."""

    stem: str
    rgb: Path
    depth: Path
    semantic: Path
    pose: Path


def scene_name_from_dir(scene_dir: Path) -> str:
    """Infer the mesh stem without assuming a numerical scene-ID prefix."""
    meshes = sorted(scene_dir.glob("*.basis.glb"))
    if len(meshes) == 1:
        return meshes[0].name.removesuffix(".basis.glb")
    if "-" in scene_dir.name:
        return scene_dir.name.split("-", maxsplit=1)[1]
    return scene_dir.name


def discover_scene_paths(
    dataset_dir: str | Path,
    walks_dir: str | Path,
    pose_dir: str | Path | None = None,
    splits: Sequence[str] | None = None,
    scene_ids: Iterable[str] | None = None,
) -> list[ScenePaths]:
    """Discover scene directories below ``<dataset_dir>/<split>/<scene_id>``.

    If no split is supplied, every immediate child directory that contains scene
    directories is used.  This keeps the format compatible with the existing
    train/val tree while avoiding a fixed split list.
    """
    dataset_root = Path(dataset_dir)
    walks_root = Path(walks_dir)
    pose_root = Path(pose_dir) if pose_dir is not None else None
    requested = set(scene_ids) if scene_ids is not None else None

    if splits is None:
        split_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    else:
        split_dirs = [dataset_root / split for split in splits]

    scenes: list[ScenePaths] = []
    for split_dir in split_dirs:
        if not split_dir.is_dir():
            continue
        for raw_scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if requested is not None and raw_scene_dir.name not in requested:
                continue
            scene_name = scene_name_from_dir(raw_scene_dir)
            pose_file = pose_root / f"{raw_scene_dir.name}.txt" if pose_root else None
            scenes.append(
                ScenePaths(
                    scene_id=raw_scene_dir.name,
                    split=split_dir.name,
                    raw_scene_dir=raw_scene_dir,
                    scene_name=scene_name,
                    basis_mesh=raw_scene_dir / f"{scene_name}.basis.glb",
                    semantic_mesh=raw_scene_dir / f"{scene_name}.semantic.glb",
                    semantic_annotations=raw_scene_dir / f"{scene_name}.semantic.txt",
                    pose_file=pose_file,
                    walk_scene_dir=walks_root / split_dir.name / raw_scene_dir.name,
                )
            )
    return scenes


def resolve_scene_config(dataset_dir: str | Path, split: str, explicit_path: str | Path | None = None) -> Path | None:
    """Find a scene-dataset config, preferring one explicitly matching ``split``."""
    if explicit_path is not None:
        path = Path(explicit_path)
        return path if path.is_file() else None
    candidates = sorted(Path(dataset_dir).glob("*.scene_dataset_config.json"))
    split_candidates = [path for path in candidates if split.lower() in path.name.lower()]
    return (split_candidates or candidates or [None])[0]


def validate_raw_scene(scene: ScenePaths, scene_config: Path | None, require_poses: bool) -> list[str]:
    """Return precise missing source-data labels for a scene."""
    requirements: list[tuple[str, Path | None]] = [
        ("Habitat basis mesh (.basis.glb)", scene.basis_mesh),
        ("semantic mesh (.semantic.glb)", scene.semantic_mesh),
        ("semantic annotations (.semantic.txt)", scene.semantic_annotations),
        ("scene dataset config (.scene_dataset_config.json)", scene_config),
    ]
    if require_poses:
        requirements.append(("trajectory pose file", scene.pose_file))
    return [f"{label}: {path}" for label, path in requirements if path is None or not path.is_file()]


def validate_pose_file(pose_file: Path) -> list[str]:
    """Validate the legacy tab/whitespace separated 4x4 pose-file format."""
    try:
        lines = [line.strip() for line in pose_file.read_text().splitlines() if line.strip()]
    except OSError as exc:
        return [f"trajectory pose file cannot be read: {pose_file} ({exc})"]
    if not lines:
        return [f"trajectory pose file contains no poses: {pose_file}"]
    for line_number, line in enumerate(lines, start=1):
        values = np.fromstring(line, sep=" ")
        if values.size != 16 or not np.isfinite(values).all():
            return [f"invalid 4x4 pose at {pose_file}:{line_number}"]
    return []


def load_pose_matrices(pose_file: Path) -> list[np.ndarray]:
    """Read validated 4x4 pose matrices in source coordinate conventions."""
    errors = validate_pose_file(pose_file)
    if errors:
        raise ValueError("; ".join(errors))
    return [
        np.fromstring(line, sep=" ").reshape(4, 4)
        for line in pose_file.read_text().splitlines()
        if line.strip()
    ]


def floor_bounds_from_metadata(metadata_path: str | Path | None, scene_id: str, floor_id: int = 0) -> FloorBounds | None:
    """Read ``Scene Name,Separation Heights`` metadata when it is supplied."""
    if metadata_path is None:
        return None
    path = Path(metadata_path)
    if not path.is_file():
        return None
    try:
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                if row.get("Scene Name") != scene_id:
                    continue
                heights = ast.literal_eval(row.get("Separation Heights", ""))
                if not isinstance(heights, (list, tuple)) or len(heights) <= floor_id + 1:
                    return None
                lower, upper = float(heights[floor_id]), float(heights[floor_id + 1])
                if lower >= upper:
                    return None
                return FloorBounds(floor_id, lower, upper, f"metadata:{path}")
    except (OSError, csv.Error, SyntaxError, ValueError, TypeError):
        return None
    return None


def floor_bounds_from_semantic_scene(semantic_scene: object, floor_id: int = 0) -> FloorBounds | None:
    """Derive bounds from Habitat's semantic level with the requested ID."""
    for level in getattr(semantic_scene, "levels", []):
        if int(getattr(level, "id", -1)) != floor_id:
            continue
        aabb = getattr(level, "aabb", None)
        if aabb is None:
            continue
        center = np.asarray(aabb.center, dtype=float)
        size = np.asarray(aabb.sizes if hasattr(aabb, "sizes") else aabb.size, dtype=float)
        lower, upper = float(center[1] - size[1] / 2), float(center[1] + size[1] / 2)
        if lower < upper:
            return FloorBounds(floor_id, lower, upper, "habitat_semantic_level")
    return None


def pose_is_on_floor(pose: np.ndarray, floor: FloorBounds, tolerance: float = 1e-4) -> bool:
    """Return whether a source-coordinate camera pose belongs to ``floor``."""
    height = float(pose[1, 3])
    return floor.lower - tolerance <= height <= floor.upper + tolerance


def discover_aligned_walk_frames(scene_dir: str | Path) -> tuple[list[FramePaths], list[str]]:
    """Return aligned walk frames or exact directory/file mismatch messages."""
    root = Path(scene_dir)
    directories = {
        "RGB directory": root / "rgb",
        "depth directory": root / "depth",
        "semantic directory": root / "semantic",
        "pose directory": root / "pose",
    }
    missing = [f"{label}: {path}" for label, path in directories.items() if not path.is_dir()]
    if missing:
        return [], missing
    files = {
        "rgb": {path.stem: path for path in directories["RGB directory"].glob("*.png")},
        "depth": {path.stem: path for path in directories["depth directory"].glob("*.png")},
        "semantic": {path.stem: path for path in directories["semantic directory"].glob("*.npy")},
        "pose": {path.stem: path for path in directories["pose directory"].glob("*.txt")},
    }
    names = {kind: set(paths) for kind, paths in files.items()}
    common = set.intersection(*names.values()) if names else set()
    if not common:
        return [], [f"no common RGB/depth/semantic/pose frames under {root}"]
    errors: list[str] = []
    for kind, stems in names.items():
        unmatched = sorted(stems - common)
        if unmatched:
            errors.append(f"unaligned {kind} frame stems: {', '.join(unmatched[:10])}" + (" ..." if len(unmatched) > 10 else ""))
    if errors:
        return [], errors
    return [FramePaths(stem, files["rgb"][stem], files["depth"][stem], files["semantic"][stem], files["pose"][stem]) for stem in sorted(common)], []


def missing_output_paths(scene_dir: str | Path) -> list[str]:
    """List required final artifacts absent from a processed walk scene."""
    root = Path(scene_dir)
    expected = [
        root / "rgb",
        root / "depth",
        root / "semantic",
        root / "pose",
        root / "objects",
        root / "regions",
        root / "semantic_label_map.csv",
        root / "scene_info.json",
        root / "scene_panoptic.ply",
        root / "scene_rgb.ply",
    ]
    return [str(path) for path in expected if not path.exists()]
