#!/usr/bin/env python3
"""Render whole-building RGB-D-semantic walks for every valid scene in a dataset tree."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable

import habitat_sim
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from hovsg.data.hm3dsem.habitat_utils import make_cfg, save_obs
from hovsg.data.hm3dsem.preparation_utils import (
    discover_scene_paths,
    floor_index_for_height,
    floor_separations_from_metadata,
    floor_separations_from_navmesh,
    load_pose_matrices,
    resolve_scene_config,
    validate_pose_file,
    validate_raw_scene,
)


LOG = logging.getLogger("hm3dsem.walks")


# These are the README evaluation scenes.  Their supplied trajectories are
# preserved; missing trajectories for them are intentionally not synthesized.
EXCLUDED_TRAJECTORY_SCENES = frozenset(
    {
        "00824-Dd4bFSTQ8gi",
        "00829-QaLdnwvtxbs",
        "00843-DYehNKdT76V",
        "00861-GLAQ4DNUx5U",
        "00862-LT9Jq6dN3Ea",
        "00873-bxsVRursffK",
        "00877-4ok3usBNeis",
        "00890-6s7QHgap2fW",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Raw dataset root containing split directories")
    parser.add_argument("--save-dir", required=True, type=Path, help="Walk-output root")
    parser.add_argument("--pose-dir", required=True, type=Path, help="Directory containing <scene_id>.txt trajectories")
    parser.add_argument("--split", action="append", dest="splits", help="Split directory to process; repeatable; defaults to all")
    parser.add_argument("--scene-id", action="append", dest="scene_ids", help="Scene ID to process; repeatable; defaults to all")
    parser.add_argument("--scene-config", type=Path, help="Explicit Habitat scene dataset config")
    parser.add_argument("--floor-metadata", type=Path, help="Optional CSV with Scene Name and Separation Heights")
    parser.add_argument("--width", type=int, default=1080, help="Rendered image width")
    parser.add_argument("--height", type=int, default=720, help="Rendered image height")
    parser.add_argument("--hfov", type=float, default=90.0, help="Horizontal camera field of view in degrees")
    parser.add_argument("--sensor-height", type=float, default=1.5, help="Sensor height used by Habitat sensor specs")
    parser.add_argument(
        "--generate-missing-poses",
        dest="generate_missing_poses",
        action="store_true",
        default=True,
        help="Synthesize a whole-building navmesh trajectory when a non-excluded scene has no pose file (default)",
    )
    parser.add_argument(
        "--no-generate-missing-poses",
        dest="generate_missing_poses",
        action="store_false",
        help="Require existing pose files for every scene",
    )
    parser.add_argument(
        "--generated-pose-dir",
        type=Path,
        help="Directory for synthesized trajectories; defaults to the scene output directory so --pose-dir stays read-only",
    )
    parser.add_argument("--trajectory-step", type=float, default=0.25, help="Generated trajectory spacing in metres")
    parser.add_argument("--trajectory-targets", type=int, default=80, help="Navmesh targets sampled per storey for a generated trajectory")
    parser.add_argument("--max-trajectory-poses", type=int, default=5000, help="Maximum generated poses per scene")
    parser.add_argument("--trajectory-seed", type=int, default=42, help="Base deterministic seed for navmesh trajectory sampling")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing scene output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report scenes without rendering")
    return parser.parse_args()


def pose_vector(pose_matrix: np.ndarray) -> np.ndarray:
    quat = R.from_matrix(pose_matrix[:3, :3]).as_quat()
    return np.concatenate([pose_matrix[:3, 3], quat])


def trajectory_generation_enabled(scene_id: str, args: argparse.Namespace) -> bool:
    return args.generate_missing_poses and scene_id not in EXCLUDED_TRAJECTORY_SCENES


def has_supplied_trajectory(scene) -> bool:
    """Whether ``--pose-dir`` provides a trajectory file for this scene."""
    return scene.pose_file is not None and scene.pose_file.is_file()


def should_generate_trajectory(scene, args: argparse.Namespace) -> bool:
    """Decide between the supplied trajectory and a synthesized one.

    A trajectory under ``--pose-dir`` always wins: those files are the curated
    HM3DSEM walks and must be reproduced exactly.  Only a scene without one
    falls back to the navmesh planner.  The README evaluation scenes are never
    synthesized - a missing trajectory there is a data error to report, not a
    gap to fill with a different walk.
    """
    if has_supplied_trajectory(scene):
        return False
    return trajectory_generation_enabled(scene.scene_id, args)


def interpolate_path(points: Iterable[np.ndarray], step: float, maximum: int) -> list[np.ndarray]:
    """Densify Habitat path points while preserving their world coordinates."""
    points = [np.asarray(point, dtype=float) for point in points]
    if len(points) < 2:
        return points
    poses: list[np.ndarray] = []
    for start, end in zip(points, points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(distance / step)))
        for index in range(count):
            if len(poses) >= maximum:
                return poses
            poses.append(start + (end - start) * (index / count))
    if len(poses) < maximum:
        poses.append(points[-1])
    return poses


def eye_height_on_floor(navmesh_height: float, sensor_height: float, separations=None, margin: float = 0.05) -> float:
    """Raise a navmesh-level point to camera height without leaving its storey.

    ``get_random_navigable_point`` returns walkable-surface heights, but the
    renderer assigns the pose directly to the sensor, so an unlifted trajectory
    films the floor.  The supplied trajectories sit one ``sensor_height`` above
    the navmesh.  Clamping keeps the camera inside the storey it stands on so a
    low ceiling cannot push it into the floor above.
    """
    height = float(navmesh_height) + float(sensor_height)
    if separations is None:
        return height
    # Accept a FloorBounds for callers that still work with a single storey.
    if hasattr(separations, "lower") and hasattr(separations, "upper"):
        lower, upper = float(separations.lower), float(separations.upper)
    else:
        boundaries = [float(value) for value in separations]
        index = floor_index_for_height(float(navmesh_height), boundaries)
        if index is None:
            return height
        lower, upper = boundaries[index], boundaries[index + 1]
    return max(lower + margin, min(height, upper - margin))


def camera_pose_matrix(
    position: np.ndarray, next_position: np.ndarray, sensor_height: float = 0.0, separations=None
) -> np.ndarray:
    """Create a Habitat camera-to-world pose looking along the path direction."""
    direction = np.asarray(next_position, dtype=float) - np.asarray(position, dtype=float)
    direction[1] = 0.0
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([0.0, 0.0, -1.0])
    direction /= np.linalg.norm(direction)
    yaw = float(np.arctan2(-direction[0], -direction[2]))
    rotation = np.array(
        [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]],
        dtype=float,
    )
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    if sensor_height:
        pose[1, 3] = eye_height_on_floor(position[1], sensor_height, separations)
    return pose


def write_pose_matrices(pose_file: Path, poses: Iterable[np.ndarray]) -> None:
    pose_file.parent.mkdir(parents=True, exist_ok=True)
    with pose_file.open("w") as file:
        for pose in poses:
            file.write("\t".join(map(str, np.asarray(pose).reshape(4, 4).ravel())))
            file.write("\n")


def generate_trajectory(sim, scene_id: str, separations: list[float], args: argparse.Namespace) -> list[np.ndarray]:
    """Plan a deterministic whole-building coverage walk from the scene navmesh.

    Targets are sampled per storey so a multi-storey scene is covered evenly
    instead of leaving the largest floor to dominate the random sample.
    """
    if args.trajectory_step <= 0 or args.trajectory_targets < 1 or args.max_trajectory_poses < 2:
        raise ValueError("trajectory-step must be > 0, trajectory-targets >= 1, and max-trajectory-poses >= 2")
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        raise ValueError("navigation mesh is not loaded by Habitat-Sim")
    scene_seed = args.trajectory_seed + sum(ord(character) for character in scene_id)
    pathfinder.seed(scene_seed)
    floor_count = max(1, len(separations) - 1)

    def sample_point(floor_index: int | None):
        for _ in range(args.trajectory_targets * 20):
            point = np.asarray(pathfinder.get_random_navigable_point(), dtype=float)
            if floor_index is None or floor_index_for_height(float(point[1]), separations) == floor_index:
                return point
        return None

    # Walk one storey at a time.  Storeys are frequently disconnected on the
    # navmesh, so interleaving targets across storeys would make almost every
    # path query fail and leave the upper floors uncovered.
    positions: list[np.ndarray] = []
    budget = max(2, args.max_trajectory_poses // floor_count)
    for floor_index in range(floor_count):
        targets = [point for point in (sample_point(floor_index) for _ in range(args.trajectory_targets)) if point is not None]
        if len(targets) < 2:
            continue
        limit = min(args.max_trajectory_poses, len(positions) + budget)
        current = targets[0]
        positions.append(current)
        for target in targets[1:]:
            shortest_path = habitat_sim.ShortestPath()
            shortest_path.requested_start = current
            shortest_path.requested_end = target
            if not pathfinder.find_path(shortest_path) or len(shortest_path.points) < 2:
                current = target
                continue
            segment = interpolate_path(shortest_path.points, args.trajectory_step, limit - len(positions))
            if len(segment) > 1:
                positions.extend(segment[1:])
            current = target
            if len(positions) >= limit:
                break
        if len(positions) >= args.max_trajectory_poses:
            break
    if len(positions) < 2:
        raise ValueError("could not plan a multi-pose trajectory from the navigation mesh")
    return [
        camera_pose_matrix(position, positions[min(index + 1, len(positions) - 1)], args.sensor_height, separations)
        for index, position in enumerate(positions)
    ]


def simulator_settings(args: argparse.Namespace, scene_mesh: Path, output_dir: Path) -> dict:
    """Keep all render parameters explicit and recordable rather than scene-specific."""
    return {
        "scene": str(scene_mesh),
        "default_agent": 0,
        "sensor_height": args.sensor_height,
        "color_sensor": True,
        "depth_sensor": True,
        "semantic_sensor": True,
        "lidar_sensor": False,
        "move_forward": 0.2,
        "move_backward": 0.2,
        "turn_left": 5,
        "turn_right": 5,
        "look_up": 5,
        "look_down": 5,
        "look_left": 5,
        "look_right": 5,
        "width": args.width,
        "height": args.height,
        "hfov": args.hfov,
        "enable_physics": False,
        "seed": 42,
        "lidar_fov": 360,
        "depth_img_for_lidar_n": 20,
        "img_save_dir": str(output_dir),
    }


def write_camera_info(output_dir: Path, args: argparse.Namespace, separations: list[float], source: str) -> None:
    payload = {
        "width": args.width,
        "height": args.height,
        "hfov_degrees": args.hfov,
        "floor_count": len(separations) - 1,
        "floor_separations": list(separations),
        "floor_separations_source": source,
    }
    (output_dir / "camera_info.json").write_text(json.dumps(payload, indent=2) + "\n")


def process_scene(scene, args: argparse.Namespace) -> bool:
    config = resolve_scene_config(args.dataset_dir, scene.split, args.scene_config)
    generate_poses = should_generate_trajectory(scene, args)
    missing = validate_raw_scene(scene, config, require_poses=not generate_poses, require_navmesh=generate_poses)
    # A supplied trajectory is never silently replaced by a synthesized one: a
    # malformed file is reported so the source data gets fixed.
    if not missing and has_supplied_trajectory(scene):
        missing.extend(validate_pose_file(scene.pose_file))
    if missing:
        LOG.warning("SKIPPED %s: %s", scene.scene_id, "; ".join(missing))
        return False
    source = "synthesized from navmesh" if generate_poses else f"supplied trajectory {scene.pose_file}"
    if args.dry_run:
        LOG.info("VALID %s: %s", scene.scene_id, source)
        return True

    output_dir = scene.walk_scene_dir
    created_output = False
    sim = None
    try:
        if output_dir.exists():
            if not args.overwrite:
                LOG.warning("SKIPPED %s: output already exists: %s (pass --overwrite to replace it)", scene.scene_id, output_dir)
                return False
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        created_output = True
        settings = simulator_settings(args, scene.basis_mesh, output_dir)
        sim = habitat_sim.Simulator(
            make_cfg(settings, str(args.dataset_dir), str(scene.raw_scene_dir), scene.scene_name, str(config))
        )
        # Separation heights describe the whole building.  The annotated scenes
        # carry them in the CSV; the rest derive them from the navmesh.
        separations = floor_separations_from_metadata(args.floor_metadata, scene.scene_id)
        separation_source = f"metadata:{args.floor_metadata}"
        if separations is None:
            separations = floor_separations_from_navmesh(sim.pathfinder)
            separation_source = "navmesh_height_histogram"
        if separations is None:
            LOG.warning(
                "SKIPPED %s: floor separation heights are unavailable from metadata or the navmesh", scene.scene_id
            )
            shutil.rmtree(output_dir)
            return False
        LOG.info(
            "%s: %d floor(s), separations %s from %s",
            scene.scene_id,
            len(separations) - 1,
            [round(value, 3) for value in separations],
            separation_source,
        )

        if generate_poses:
            poses = generate_trajectory(sim, scene.scene_id, separations, args)
            # Never write into --pose-dir: it holds the tracked supplied trajectories.
            generated_pose_file = (
                args.generated_pose_dir / f"{scene.scene_id}.txt" if args.generated_pose_dir else output_dir / "trajectory.txt"
            )
            write_pose_matrices(generated_pose_file, poses)
            LOG.info("GENERATED %s: wrote %d trajectory poses to %s", scene.scene_id, len(poses), generated_pose_file)
        else:
            # A supplied trajectory is rendered in full: it is the curated walk
            # and it deliberately spans every storey of the building.
            poses = load_pose_matrices(scene.pose_file)
            LOG.info("SUPPLIED %s: rendering all %d poses from %s", scene.scene_id, len(poses), scene.pose_file)
        if not poses:
            LOG.warning("SKIPPED %s: trajectory contains no camera poses", scene.scene_id)
            shutil.rmtree(output_dir)
            return False
        write_camera_info(output_dir, args, separations, separation_source)
        agent = sim.initialize_agent(settings["default_agent"])
        # Place the agent body on the navmesh exactly as the upstream
        # gen_hm3dsem_from_poses.py does.  The sensors are positioned absolutely
        # below, but the body transform still participates in composing the
        # sensor world transform, so omitting this shifts sub-pixel rasterization
        # and the output stops being bit-identical to the released walks.
        body_state = habitat_sim.AgentState()
        body_state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(body_state)
        for index, pose_matrix in enumerate(tqdm(poses, desc=f"Rendering {scene.scene_id}", unit="frame")):
            pose = pose_vector(pose_matrix)
            state = agent.get_state()
            for sensor_name in ("color_sensor", "depth_sensor", "semantic"):
                state.sensor_states[sensor_name].position = pose[:3]
                state.sensor_states[sensor_name].rotation = pose[3:]
            agent.set_state(state, reset_sensors=True, infer_sensor_states=False)
            save_obs(str(output_dir), settings, sim.get_sensor_observations(0), pose, index)
        heights = np.array([float(pose[1, 3]) for pose in poses])
        covered = sorted({index for index in (floor_index_for_height(h, separations) for h in heights) if index is not None})
        LOG.info(
            "PROCESSED %s: rendered %d frames covering storey(s) %s of %d",
            scene.scene_id,
            len(poses),
            covered,
            len(separations) - 1,
        )
        return True
    except Exception as exc:  # Continue with independent scenes after simulator/data failures.
        LOG.exception("FAILED %s: %s", scene.scene_id, exc)
        if created_output and output_dir.exists():
            shutil.rmtree(output_dir)
            LOG.info("Removed incomplete output for %s", scene.scene_id)
        return False
    finally:
        if sim is not None:
            sim.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    scenes = discover_scene_paths(args.dataset_dir, args.save_dir, args.pose_dir, args.splits, args.scene_ids)
    if not scenes:
        LOG.error("No scene directories discovered under %s", args.dataset_dir)
        raise SystemExit(1)
    succeeded = sum(process_scene(scene, args) for scene in scenes)
    failed = len(scenes) - succeeded
    if args.dry_run:
        LOG.info("Walk validation complete: %d valid, %d invalid", succeeded, failed)
    else:
        LOG.info("Walk generation complete: %d processed, %d skipped or failed", succeeded, failed)
    # A skipped or failed scene must be visible to the caller's exit status.
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
