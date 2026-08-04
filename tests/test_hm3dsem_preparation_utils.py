from pathlib import Path

import numpy as np

from hovsg.data.hm3dsem.preparation_utils import (
    FloorBounds,
    discover_aligned_walk_frames,
    discover_scene_paths,
    floor_bounds_from_metadata,
    pose_is_on_floor,
    validate_pose_file,
    validate_raw_scene,
)


def make_raw_scene(root: Path, split: str, scene_id: str, scene_name: str) -> Path:
    scene_dir = root / split / scene_id
    scene_dir.mkdir(parents=True)
    for suffix in (".basis.glb", ".semantic.glb", ".semantic.txt"):
        (scene_dir / f"{scene_name}{suffix}").write_text("source")
    return scene_dir


def test_discovery_and_raw_validation_are_scene_agnostic(tmp_path):
    dataset = tmp_path / "raw"
    walks = tmp_path / "walks"
    poses = tmp_path / "poses"
    poses.mkdir()
    (dataset / "any.scene_dataset_config.json").parent.mkdir(parents=True, exist_ok=True)
    (dataset / "any.scene_dataset_config.json").write_text("{}")
    make_raw_scene(dataset, "custom_split", "alpha", "render_mesh")
    (poses / "alpha.txt").write_text(" ".join(map(str, np.eye(4).ravel())) + "\n")

    scenes = discover_scene_paths(dataset, walks, poses)

    assert len(scenes) == 1
    scene = scenes[0]
    assert scene.scene_id == "alpha"
    assert scene.scene_name == "render_mesh"
    assert validate_raw_scene(scene, dataset / "any.scene_dataset_config.json", require_poses=True) == []


def test_validation_reports_each_missing_source(tmp_path):
    dataset = tmp_path / "raw"
    walks = tmp_path / "walks"
    scene_dir = dataset / "split" / "scene"
    scene_dir.mkdir(parents=True)
    scene = discover_scene_paths(dataset, walks, tmp_path / "poses")[0]

    missing = validate_raw_scene(scene, None, require_poses=True)

    assert any("basis mesh" in item for item in missing)
    assert any("semantic mesh" in item for item in missing)
    assert any("semantic annotations" in item for item in missing)
    assert any("scene dataset config" in item for item in missing)
    assert any("trajectory pose file" in item for item in missing)


def test_frame_alignment_and_floor_metadata(tmp_path):
    scene_dir = tmp_path / "walk" / "val" / "scene"
    for directory in ("rgb", "depth", "semantic", "pose"):
        (scene_dir / directory).mkdir(parents=True)
    for stem in ("mesh_000000", "mesh_000001"):
        (scene_dir / "rgb" / f"{stem}.png").write_bytes(b"")
        (scene_dir / "depth" / f"{stem}.png").write_bytes(b"")
        (scene_dir / "semantic" / f"{stem}.npy").write_bytes(b"")
        (scene_dir / "pose" / f"{stem}.txt").write_text("pose")

    frames, errors = discover_aligned_walk_frames(scene_dir)
    assert not errors
    assert [frame.stem for frame in frames] == ["mesh_000000", "mesh_000001"]

    (scene_dir / "rgb" / "extra.png").write_bytes(b"")
    _, errors = discover_aligned_walk_frames(scene_dir)
    assert errors == ["unaligned rgb frame stems: extra"]

    floors = tmp_path / "floors.csv"
    floors.write_text("Scene Name,Separation Heights\nscene,\"[0.0, 2.5, 5.0]\"\n")
    floor = floor_bounds_from_metadata(floors, "scene")
    assert floor == FloorBounds(0, 0.0, 2.5, f"metadata:{floors}")
    pose = np.eye(4)
    pose[1, 3] = 1.5
    assert pose_is_on_floor(pose, floor)
    pose[1, 3] = 3.0
    assert not pose_is_on_floor(pose, floor)


def test_pose_validation_accepts_whitespace_and_rejects_bad_matrix(tmp_path):
    pose_file = tmp_path / "poses.txt"
    pose_file.write_text("\t".join(map(str, np.eye(4).ravel())) + "\n")
    assert validate_pose_file(pose_file) == []
    pose_file.write_text("1 2 3\n")
    assert validate_pose_file(pose_file) == [f"invalid 4x4 pose at {pose_file}:1"]
