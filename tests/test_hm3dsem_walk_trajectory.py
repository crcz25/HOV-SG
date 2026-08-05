"""Trajectory-source selection and camera placement for floor-0 walks."""

import argparse

import numpy as np
import pytest

from hovsg.data.hm3dsem.gen_hm3dsem_walks_from_poses import (
    EXCLUDED_TRAJECTORY_SCENES,
    camera_pose_matrix,
    eye_height_on_floor,
    has_supplied_trajectory,
    should_generate_trajectory,
)
from hovsg.data.hm3dsem.preparation_utils import FloorBounds, discover_scene_paths, pose_is_on_floor


FLOOR = FloorBounds(0, -0.2, 3.1, "test")


def make_scene(tmp_path, scene_id, *, with_pose_file):
    """Build a ScenePaths through the real discovery helper."""
    raw = tmp_path / "raw" / "val" / scene_id
    raw.mkdir(parents=True, exist_ok=True)
    for suffix in (".basis.glb", ".basis.navmesh", ".semantic.glb", ".semantic.txt"):
        (raw / f"mesh{suffix}").write_text("source")
    poses = tmp_path / "poses"
    poses.mkdir(exist_ok=True)
    if with_pose_file:
        (poses / f"{scene_id}.txt").write_text("\t".join(map(str, np.eye(4).ravel())) + "\n")
    scenes = discover_scene_paths(tmp_path / "raw", tmp_path / "walks", poses, scene_ids=[scene_id])
    return scenes[0]


ARGS = argparse.Namespace(generate_missing_poses=True)


def test_a_supplied_trajectory_is_always_preferred(tmp_path):
    scene = make_scene(tmp_path, "00900-ordinary", with_pose_file=True)

    assert has_supplied_trajectory(scene)
    assert should_generate_trajectory(scene, ARGS) is False


def test_a_scene_without_a_supplied_trajectory_synthesizes_one(tmp_path):
    scene = make_scene(tmp_path, "00900-ordinary", with_pose_file=False)

    assert not has_supplied_trajectory(scene)
    assert should_generate_trajectory(scene, ARGS) is True


@pytest.mark.parametrize("scene_id", sorted(EXCLUDED_TRAJECTORY_SCENES))
def test_readme_evaluation_scenes_use_their_supplied_trajectory(tmp_path, scene_id):
    scene = make_scene(tmp_path, scene_id, with_pose_file=True)

    assert should_generate_trajectory(scene, ARGS) is False


@pytest.mark.parametrize("scene_id", sorted(EXCLUDED_TRAJECTORY_SCENES))
def test_readme_evaluation_scenes_are_never_synthesized(tmp_path, scene_id):
    # Their walks are manually curated; a missing file is a data error to
    # report, not a gap to fill with a different trajectory.
    scene = make_scene(tmp_path, scene_id, with_pose_file=False)

    assert should_generate_trajectory(scene, ARGS) is False


def test_generation_can_be_disabled_entirely(tmp_path):
    scene = make_scene(tmp_path, "00900-ordinary", with_pose_file=False)

    assert should_generate_trajectory(scene, argparse.Namespace(generate_missing_poses=False)) is False


def test_every_shipped_pose_file_wins_over_generation(tmp_path):
    """The ten trajectories tracked in the repository must all be honoured."""
    from pathlib import Path

    shipped = sorted(p.stem for p in Path("hovsg/data/hm3dsem/metadata/poses").glob("*.txt"))
    assert len(shipped) >= 1
    for scene_id in shipped:
        scene = make_scene(tmp_path, scene_id, with_pose_file=True)
        assert should_generate_trajectory(scene, ARGS) is False, scene_id


def test_navmesh_point_is_raised_to_sensor_height():
    # get_random_navigable_point returns the walkable surface, but the renderer
    # assigns the pose straight to the sensor, so an unraised point films
    # the floor instead of the room.
    assert eye_height_on_floor(0.07, 1.5, FLOOR) == pytest.approx(1.57)


def test_raised_height_is_clamped_within_the_storey_it_stands_on():
    # Three storeys; a point on the middle one must not be lifted into the top.
    separations = [-2.48, -0.25, 2.42, 5.72]
    assert eye_height_on_floor(-2.21, 1.5, separations) == pytest.approx(-0.71)
    assert eye_height_on_floor(0.0, 1.5, separations) == pytest.approx(1.5)
    # A 1.0 m-tall crawlspace cannot host a 1.5 m camera; clamp, do not spill.
    tight = [0.0, 1.0, 4.0]
    assert 0.0 <= eye_height_on_floor(0.05, 1.5, tight) <= 1.0


def test_height_outside_every_storey_is_left_unclamped():
    assert eye_height_on_floor(99.0, 1.5, [-0.2, 3.1]) == pytest.approx(100.5)


def test_raised_height_stays_inside_the_floor_slab():
    # create_hm3dsem_walks_gt.py rejects a whole walk containing an off-floor
    # pose, so the lift must never cross the ceiling of a low storey.
    low = FloorBounds(0, -0.2, 1.0, "test")
    height = eye_height_on_floor(0.07, 1.5, low)

    assert low.lower <= height <= low.upper
    pose = np.eye(4)
    pose[1, 3] = height
    assert pose_is_on_floor(pose, low)


def test_camera_pose_is_a_rigid_transform_looking_along_the_path():
    pose = camera_pose_matrix(np.array([0.0, 0.07, 0.0]), np.array([0.0, 0.07, -1.0]), 1.5, FLOOR)

    rotation = pose[:3, :3]
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0])
    assert pose[1, 3] == pytest.approx(1.57)
    # Habitat cameras look down -Z, so a path heading to -Z needs no yaw.
    assert np.allclose(rotation, np.eye(3), atol=1e-9)


def test_pose_height_is_unchanged_when_no_sensor_height_is_requested():
    pose = camera_pose_matrix(np.array([1.0, 0.07, 2.0]), np.array([1.0, 0.07, 1.0]))

    assert pose[1, 3] == pytest.approx(0.07)
    assert np.allclose(pose[:3, 3], [1.0, 0.07, 2.0])


def test_supplied_and_synthesized_trajectories_agree_on_camera_height():
    # The supplied HM3DSEM trajectories sit one sensor height above the navmesh;
    # synthesized ones must land in the same band, not at floor level.
    assert eye_height_on_floor(0.069, 1.5, FLOOR) == pytest.approx(1.569, abs=1e-3)
