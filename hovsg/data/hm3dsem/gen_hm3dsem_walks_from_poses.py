#!/usr/bin/env python3
"""Render floor-0 RGB-D-semantic walks for every valid scene in a dataset tree."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import habitat_sim
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from hovsg.data.hm3dsem.habitat_utils import make_cfg, save_obs
from hovsg.data.hm3dsem.preparation_utils import (
    discover_scene_paths,
    floor_bounds_from_metadata,
    floor_bounds_from_semantic_scene,
    load_pose_matrices,
    pose_is_on_floor,
    resolve_scene_config,
    validate_pose_file,
    validate_raw_scene,
)


LOG = logging.getLogger("hm3dsem.walks")


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
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing scene output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report scenes without rendering")
    return parser.parse_args()


def pose_vector(pose_matrix: np.ndarray) -> np.ndarray:
    quat = R.from_matrix(pose_matrix[:3, :3]).as_quat()
    return np.concatenate([pose_matrix[:3, 3], quat])


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


def write_camera_info(output_dir: Path, args: argparse.Namespace, floor) -> None:
    payload = {
        "width": args.width,
        "height": args.height,
        "hfov_degrees": args.hfov,
        "floor_id": floor.floor_id,
        "floor_y_bounds": [floor.lower, floor.upper],
        "floor_bounds_source": floor.source,
    }
    (output_dir / "camera_info.json").write_text(json.dumps(payload, indent=2) + "\n")


def process_scene(scene, args: argparse.Namespace) -> bool:
    config = resolve_scene_config(args.dataset_dir, scene.split, args.scene_config)
    missing = validate_raw_scene(scene, config, require_poses=True)
    if not missing and scene.pose_file is not None:
        missing.extend(validate_pose_file(scene.pose_file))
    if missing:
        LOG.warning("SKIPPED %s: %s", scene.scene_id, "; ".join(missing))
        return False
    if args.dry_run:
        LOG.info("VALID %s", scene.scene_id)
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
        floor = floor_bounds_from_metadata(args.floor_metadata, scene.scene_id)
        if floor is None:
            floor = floor_bounds_from_semantic_scene(sim.semantic_scene, floor_id=0)
        if floor is None:
            LOG.warning("SKIPPED %s: floor 0 bounds are unavailable from metadata or Habitat semantic levels", scene.scene_id)
            shutil.rmtree(output_dir)
            return False

        poses = [pose for pose in load_pose_matrices(scene.pose_file) if pose_is_on_floor(pose, floor)]
        if not poses:
            LOG.warning("SKIPPED %s: trajectory contains no camera poses on floor 0", scene.scene_id)
            shutil.rmtree(output_dir)
            return False
        write_camera_info(output_dir, args, floor)
        agent = sim.initialize_agent(settings["default_agent"])
        for index, pose_matrix in enumerate(tqdm(poses, desc=f"Rendering {scene.scene_id}", unit="frame")):
            pose = pose_vector(pose_matrix)
            state = agent.get_state()
            for sensor_name in ("color_sensor", "depth_sensor", "semantic"):
                state.sensor_states[sensor_name].position = pose[:3]
                state.sensor_states[sensor_name].rotation = pose[3:]
            agent.set_state(state, reset_sensors=True, infer_sensor_states=False)
            save_obs(str(output_dir), settings, sim.get_sensor_observations(0), pose, index)
        LOG.info("PROCESSED %s: rendered %d aligned floor-0 frames", scene.scene_id, len(poses))
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
        return
    succeeded = sum(process_scene(scene, args) for scene in scenes)
    if args.dry_run:
        LOG.info("Walk validation complete: %d valid, %d invalid", succeeded, len(scenes) - succeeded)
    else:
        LOG.info("Walk generation complete: %d processed, %d skipped or failed", succeeded, len(scenes) - succeeded)


if __name__ == "__main__":
    main()
