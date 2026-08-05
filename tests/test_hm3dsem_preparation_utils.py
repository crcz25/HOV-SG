import json
from pathlib import Path

import numpy as np
import pytest

from hovsg.data.hm3dsem.preparation_utils import (
    FloorBounds,
    discover_aligned_walk_frames,
    discover_scene_paths,
    floor_index_for_height,
    floor_separations_from_camera_info,
    floor_separations_from_metadata,
    pose_is_on_floor,
    validate_pose_file,
    validate_raw_scene,
    walkable_levels_from_heights,
)


def make_raw_scene(root: Path, split: str, scene_id: str, scene_name: str) -> Path:
    scene_dir = root / split / scene_id
    scene_dir.mkdir(parents=True)
    for suffix in (".basis.glb", ".basis.navmesh", ".semantic.glb", ".semantic.txt"):
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
    assert validate_raw_scene(scene, dataset / "any.scene_dataset_config.json", require_poses=True, require_navmesh=True) == []


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
    assert any("navigation mesh" in item for item in validate_raw_scene(scene, None, require_poses=False, require_navmesh=True))


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
    separations = floor_separations_from_metadata(floors, "scene")
    assert separations == [0.0, 2.5, 5.0]
    ground = FloorBounds(0, separations[0], separations[1], f"metadata:{floors}")
    pose = np.eye(4)
    pose[1, 3] = 1.5
    assert pose_is_on_floor(pose, ground)
    pose[1, 3] = 3.0
    assert not pose_is_on_floor(pose, ground)


def test_pose_validation_accepts_whitespace_and_rejects_bad_matrix(tmp_path):
    pose_file = tmp_path / "poses.txt"
    pose_file.write_text("\t".join(map(str, np.eye(4).ravel())) + "\n")
    assert validate_pose_file(pose_file) == []
    pose_file.write_text("1 2 3\n")
    assert validate_pose_file(pose_file) == [f"invalid 4x4 pose at {pose_file}:1"]


def test_walkable_levels_separate_storeys_joined_by_a_staircase():
    generator = np.random.default_rng(0)
    ground = generator.normal(0.0, 0.01, 4000)
    upper = generator.normal(2.9, 0.01, 4000)
    # A staircase contributes few samples spread across the whole gap, which is
    # what defeats nearest-neighbour gap clustering.
    stairs = np.linspace(0.0, 2.9, 120)

    levels = walkable_levels_from_heights(np.concatenate([ground, upper, stairs]))

    assert len(levels) == 2
    assert levels[0] == pytest.approx(0.0, abs=0.05)
    assert levels[1] == pytest.approx(2.9, abs=0.05)


def test_walkable_levels_report_a_single_storey_once():
    generator = np.random.default_rng(1)
    levels = walkable_levels_from_heights(generator.normal(0.07, 0.02, 4000))

    assert len(levels) == 1
    assert levels[0] == pytest.approx(0.07, abs=0.05)


def test_walkable_levels_tolerate_degenerate_input():
    assert walkable_levels_from_heights([]) == []
    assert walkable_levels_from_heights([np.nan, np.inf]) == []
    assert walkable_levels_from_heights([1.25]) == [pytest.approx(1.25, abs=0.05)]


def test_camera_info_separations_are_reused_and_bad_records_rejected(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    info = scene_dir / "camera_info.json"

    info.write_text(json.dumps({"floor_separations": [-0.2, 3.1, 6.0]}))
    assert floor_separations_from_camera_info(scene_dir) == [-0.2, 3.1, 6.0]

    # Non-monotonic or single-value records describe no storey at all.
    info.write_text(json.dumps({"floor_separations": [3.1, -0.2]}))
    assert floor_separations_from_camera_info(scene_dir) is None
    info.write_text(json.dumps({"floor_separations": [3.1]}))
    assert floor_separations_from_camera_info(scene_dir) is None

    info.write_text("not json")
    assert floor_separations_from_camera_info(scene_dir) is None

    info.unlink()
    assert floor_separations_from_camera_info(scene_dir) is None


def test_metadata_separations_keep_every_storey(tmp_path):
    floors = tmp_path / "floors.csv"
    # Three boundaries describe two storeys; the old reader kept only the first.
    floors.write_text('Scene Name,Separation Heights\nscene,"[-0.05, 3.05, 5.93]"\n')

    assert floor_separations_from_metadata(floors, "scene") == [-0.05, 3.05, 5.93]
    assert floor_separations_from_metadata(floors, "absent") is None

    floors.write_text('Scene Name,Separation Heights\nscene,"[3.0, 1.0]"\n')
    assert floor_separations_from_metadata(floors, "scene") is None


def test_every_annotated_scene_reports_its_real_storey_count():
    """The shipped CSV must survive the reader that replaced the floor-0 one."""
    csv_path = Path("hovsg/data/hm3dsem/metadata/Per_Scene_Floor_Sep.csv")
    expected = {"00824-Dd4bFSTQ8gi": 1, "00829-QaLdnwvtxbs": 1, "00843-DYehNKdT76V": 2,
                "00847-bCPU9suPUw9": 3, "00862-LT9Jq6dN3Ea": 3}
    for scene_id, floors in expected.items():
        separations = floor_separations_from_metadata(csv_path, scene_id)
        assert separations is not None, scene_id
        assert len(separations) - 1 == floors, scene_id


@pytest.mark.parametrize(
    "height,expected",
    [(-0.1, 0), (1.0, 0), (3.0, 0), (3.2, 1), (5.5, 1), (-5.0, None), (99.0, None)],
)
def test_floor_index_lookup(height, expected):
    assert floor_index_for_height(height, [-0.2, 3.1, 6.0]) == expected
