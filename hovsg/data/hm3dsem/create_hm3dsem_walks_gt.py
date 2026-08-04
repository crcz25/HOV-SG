import argparse
import ast
from collections import defaultdict
import json
import logging
import math
import os
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import open3d as o3d
import pandas as pd

from PIL import Image, ImageColor
from tqdm import tqdm
from scipy.spatial import cKDTree

from hovsg.data.hm3dsem.habitat_utils import make_cfg
from hovsg.data.hm3dsem.preparation_utils import (
    FloorBounds,
    discover_aligned_walk_frames,
    discover_scene_paths,
    floor_bounds_from_metadata,
    floor_bounds_from_semantic_scene,
    missing_output_paths,
    pose_is_on_floor,
    resolve_scene_config,
    validate_raw_scene,
)
from scripts.generate_hm3dsem_semantic_label_csv import generate_semantic_label_map


LOG = logging.getLogger("hm3dsem.ground_truth")


class PanopticObject:
    def __init__(self, line):
        # semantics object info
        self.id = int(line[0])
        self.hex = line[1]
        try:
            self.category = ast.literal_eval(line[2])
        except (SyntaxError, ValueError):
            self.category = line[2]
        self.region_id = int(line[3])
        self.floor_id = None
        self.rgb = np.array(ImageColor.getcolor("#" + self.hex, "RGB"))
        self.type = "object"
        self.mapped = False

        # habitat object info
        self.aabb_center = None
        self.aabb_dims = None
        self.obb_center = None
        self.obb_dims = None
        self.obb_rotation = None
        self.obb_local_to_world = None
        self.obb_world_to_local = None
        self.obb_volume = None
        self.obb_half_extents = None

        # point cloud data
        self.points = None
        self.colors = None

    def __print__(self):
        print("id:", self.id, "category:", self.category, "hex_color:", self.hex, "region_id:", self.region_id)

    def hex2id(self, hex_color):
        return self.id

    def __str__(self) -> str:
        return f"{self.floor_id}_{self.region_id}_{self.id}"


def habitat_attr_value(value):
    value = value() if callable(value) else value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: habitat_attr_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [habitat_attr_value(item) for item in value]
    if hasattr(value, "tolist"):
        return habitat_attr_value(value.tolist())
    if hasattr(value, "__len__") and hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
        return [habitat_attr_value(value[idx]) for idx in range(len(value))]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return [habitat_attr_value(item) for item in value]
    return value


def rgb2hex(color_array):
    color_array = color_array * 255
    return "#%02x%02x%02x" % (int(color_array[0]), int(color_array[1]), int(color_array[2]))


class PanopticRegion:
    def __init__(self, region_id):
        self.id = region_id
        self.graph_id = None
        self.floor_id = None
        self.objects = []
        self.type = "room"
        self.voted_category = None
        self.category = None

        self.min_height = None
        self.max_height = None
        self.mean_height = None
        self.region_points = None
        self.bev_region_points = None

    def project_regions(self):
        # Aggregate region point cloud
        print("Projecting region ", self.id, " w/ ", len(self.objects), " objects")
        self.region_points = np.concatenate([obj.points for obj in self.objects if obj.mapped], axis=0)
        self.region_colors = np.concatenate([obj.colors for obj in self.objects if obj.mapped], axis=0)
        self.min_height = np.min(self.region_points[:, 1])
        self.max_height = np.max(self.region_points[:, 1])
        self.mean_height = np.mean(self.region_points[:, 1])

        # project region point cloud to xz plane and use min height as region height
        self.region_point_cloud = o3d.geometry.PointCloud()
        self.region_point_cloud.points = o3d.utility.Vector3dVector(self.region_points)
        self.region_point_cloud.colors = o3d.utility.Vector3dVector(self.region_colors)
        self.bev_region_point_cloud = o3d.geometry.PointCloud()
        self.bev_region_point_cloud.points = o3d.utility.Vector3dVector(
            np.stack(
                [
                    self.region_points[:, 0],
                    np.repeat(self.min_height, self.region_points.shape[0]),
                    self.region_points[:, 2],
                ],
                axis=1,
            )
        )
        self.bev_region_point_cloud = self.bev_region_point_cloud.voxel_down_sample(0.05)

    def __print__(self) -> str:
        return f"{self.floor_id}_{self.id}"


class PanopticLevel:
    def __init__(self, level_id, lower, upper):
        self.id = level_id
        self.lower = lower
        self.upper = upper
        self.type = "floor"

        self.regions = []
        self.objects = []

    def __print__(self) -> str:
        return f"{self.id}"


class PanopticScene:
    def __init__(self, scene_dir_name, habitat_scene, panoptic_object_list):
        self.scene = habitat_scene
        self.scene_dir_name = scene_dir_name
        self.objects = panoptic_object_list
        self.regions = defaultdict(PanopticRegion)
        self.floors = defaultdict(PanopticLevel)

        self.scene_info = {"levels": [], "regions": [], "objects": []}

        self.id2obj_idx = {}
        for i, obj in enumerate(self.objects):
            self.id2obj_idx[obj.id] = i

        self.hex2obj_idx = {}
        for i, obj in enumerate(self.objects):
            self.hex2obj_idx[obj.hex] = i

        self.hex2id = {}
        for i, obj in enumerate(self.objects):
            self.hex2id[obj.hex] = obj.id
        self.id2hex = {v: k for k, v in self.hex2id.items()}
        self.id2rgb = {id: ImageColor.getcolor("#" + hex, "RGB") for id, hex in self.id2hex.items()}

        self.append_habitat_infos()

    def append_habitat_infos(self):
        for obj in self.scene.objects:
            try:
                obj_id = int(str(obj.id).rsplit("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                LOG.warning("Ignoring Habitat semantic object with an unparseable ID: %s", obj.id)
                continue

            if obj_id in self.id2obj_idx:
                self.objects[self.id2obj_idx[obj_id]].aabb_center = habitat_attr_value(obj.aabb.center)
                self.objects[self.id2obj_idx[obj_id]].aabb_dims = habitat_attr_value(obj.aabb.size)
                self.objects[self.id2obj_idx[obj_id]].obb_center = habitat_attr_value(obj.obb.center)
                self.objects[self.id2obj_idx[obj_id]].obb_dims = habitat_attr_value(obj.obb.sizes)
                self.objects[self.id2obj_idx[obj_id]].obb_rotation = habitat_attr_value(obj.obb.rotation)
                self.objects[self.id2obj_idx[obj_id]].obb_local_to_world = habitat_attr_value(obj.obb.local_to_world)
                self.objects[self.id2obj_idx[obj_id]].obb_world_to_local = habitat_attr_value(obj.obb.world_to_local)
                self.objects[self.id2obj_idx[obj_id]].obb_volume = habitat_attr_value(obj.obb.volume)
                self.objects[self.id2obj_idx[obj_id]].obb_half_extents = habitat_attr_value(obj.obb.half_extents)

    def get_object(self, key):
        if isinstance(key, str):
            return self.objects[self.hex2obj_idx[key]]
        elif isinstance(key, int):
            return self.objects[self.id2obj_idx[key]]
        else:
            raise NotImplementedError

    def construct_regions(self):
        for obj in self.objects:
            if obj.mapped:
                if (obj.region_id) not in self.regions:
                    self.regions[int(obj.region_id)] = PanopticRegion(int(obj.region_id))
                self.regions[int(obj.region_id)].objects.append(obj)

        for region_id in self.regions.keys():
            self.regions[region_id].project_regions()

    def label_mapped_objects(self):
        for obj in self.objects:
            if obj.points is not None:
                obj.mapped = True

    def label_regions(self, region_votes_file_path=None, region_labels_file_path=None):
        # load region labels
        if region_votes_file_path and os.path.isfile(region_votes_file_path):
            region_votes = pd.read_csv(
                region_votes_file_path, header=0, usecols=["Scene Name", "Region #", "Weighted Room Proposal"], sep=","
            )
            for ind in region_votes.index:
                if region_votes["Scene Name"][ind] == self.scene_dir_name and int(region_votes["Region #"][ind]) in self.regions:
                    self.regions[int(region_votes["Region #"][ind])].voted_category = region_votes["Weighted Room Proposal"][ind].strip().lower()

        if region_labels_file_path and os.path.isfile(region_labels_file_path):
            region_labels = pd.read_csv(
                region_labels_file_path, header=0, usecols=["Scene Name", "Region #", "Region Category"], sep=","
            )
            for ind in region_labels.index:
                if region_labels["Scene Name"][ind] == self.scene_dir_name and int(region_labels["Region #"][ind]) in self.regions:
                    self.regions[int(region_labels["Region #"][ind])].category = region_labels["Region Category"][ind].strip().lower()

    def get_region_objects(self, region_id):
        if isinstance(region_id, str):
            region_id = int(region_id)
        return self.regions[region_id]
    
    def select_floor(self, floor: FloorBounds):
        """Keep only mapped regions and objects whose geometry is on floor 0."""
        selected = PanopticLevel(floor.floor_id, floor.lower, floor.upper)
        for region in self.regions.values():
            if not (floor.lower <= region.mean_height <= floor.upper):
                continue
            region.floor_id = floor.floor_id
            selected.regions.append(region)
            for obj in region.objects:
                obj.floor_id = floor.floor_id
                selected.objects.append(obj)
        self.floors = {floor.floor_id: selected}
        self.regions = {region.id: region for region in selected.regions}

    def write_metadata(self, save_dir):
        # write level information
        for floor_idx, floor_obj in self.floors.items():
            floor_item = {"id": floor_idx, "lower": floor_obj.lower, "upper": floor_obj.upper}
            floor_item["regions"] = [region.id for region in floor_obj.regions]
            floor_item["objects"] = [obj.id for obj in floor_obj.objects]
            self.scene_info["levels"].append(floor_item)

        # write region information
        for region_id, region_obj in self.regions.items():
            region_item = {
                "id": region_id,
                "floor_id": region_obj.floor_id,
                "voted_category": region_obj.voted_category,
                "category": region_obj.category,
                "min_height": region_obj.min_height,
                "max_height": region_obj.max_height,
                "mean_height": region_obj.mean_height,
                "bev_region_points": np.array(region_obj.bev_region_point_cloud.points).tolist(),
            }
            region_item["objects"] = [obj.id for obj in region_obj.objects if obj.mapped]
            self.scene_info["regions"].append(region_item)

        selected_object_ids = {obj.id for floor in self.floors.values() for obj in floor.objects}
        for obj in self.objects:
            if obj.id not in selected_object_ids:
                continue
            if obj.mapped:
                object_item = {
                    "id": obj.id,
                    "category": obj.category,
                    "hex": obj.hex,
                    "region_id": obj.region_id,
                    "floor_id": obj.floor_id,
                    "aabb_center": obj.aabb_center,
                    "aabb_dims": obj.aabb_dims,
                    "obb_center": obj.obb_center,
                    "obb_dims": obj.obb_dims,
                    "obb_rotation": obj.obb_rotation,
                    "obb_local_to_world": obj.obb_local_to_world,
                    "obb_world_to_local": obj.obb_world_to_local,
                    "obb_volume": obj.obb_volume,
                    "obb_half_extents": obj.obb_half_extents,
                    # "points": obj.points.tolist(),
                    # "colors": object.colors.tolist(),
                }
                self.scene_info["objects"].append(object_item)

        # save scene info as JSON
        with open(os.path.join(save_dir, "scene_info.json"), "w") as file:
            json.dump(self.scene_info, file, default=habitat_attr_value)


def read_camera_pose_hmp3d(file_path):
    """
    for habitat mp3d dataset, read first line of camera pose file
    16 separate by space, reshape to 4x4 matrix
    """
    with open(file_path, "r") as file:
        line = file.readline().strip()
        values = line.split()
        values = [float(val) for val in values]
        transformation_matrix = np.array(values).reshape((4, 4))
        C = np.eye(4)
        C[1, 1] = -1
        C[2, 2] = -1
        transformation_matrix = np.matmul(transformation_matrix, C)
    return transformation_matrix


def create_pcd_hmp3d(rgb, depth, camera_pose=None, hfov_degrees=90.0):
    """
    for habitat mp3d dataset, create point cloud from RGBD images
    params:
        rgb_img: numpy array of shape (H, W, 3)
        depth_img: numpy array of shape (H, W)
        camera_matrix: numpy array of shape (3, 3)
        depth_scale: depth scale factor
    return:
        pcd: Open3D point cloud
    """
    H = rgb.shape[0]
    W = rgb.shape[1]

    hfov = float(hfov_degrees) * np.pi / 180
    vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
    fx = W / (2.0 * np.tan(hfov / 2.0))
    fy = H / (2.0 * np.tan(vfov / 2.0))
    cx = W / 2
    cy = H / 2
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    depth = depth.astype(np.float32) / 1000.0
    mask = depth > 0
    x = x[mask]
    y = y[mask]
    depth = depth[mask]

    # convert to 3D
    X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
    Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
    Z = depth

    # convert to open3d point cloud
    points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = rgb[mask]
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd.transform(camera_pose)
    return pcd


def crop_point_cloud_to_floor(point_cloud, floor: FloorBounds):
    """Crop a world-coordinate point cloud to the selected floor's Y interval."""
    points = np.asarray(point_cloud.points)
    if not len(points):
        return point_cloud
    indices = np.flatnonzero((points[:, 1] >= floor.lower) & (points[:, 1] <= floor.upper)).tolist()
    return point_cloud.select_by_index(indices)


def rgb_colors_for_panoptic_points(rgb_pcd, panoptic_pcd):
    """Return RGB colors aligned to panoptic points after independent voxelization."""
    rgb_points = np.asarray(rgb_pcd.points)
    panoptic_points = np.asarray(panoptic_pcd.points)
    rgb_colors = np.asarray(rgb_pcd.colors)
    if len(rgb_points) == len(panoptic_points) and np.allclose(rgb_points, panoptic_points):
        return rgb_colors
    if not len(rgb_points):
        return np.zeros((len(panoptic_points), 3), dtype=float)
    _, indices = cKDTree(rgb_points).query(panoptic_points, k=1)
    return rgb_colors[indices]


def update_object_geometry_from_points(obj, point_cloud) -> None:
    """Replace source-wide boxes with boxes computed from floor-0 geometry."""
    aabb = point_cloud.get_axis_aligned_bounding_box()
    obj.aabb_center = np.asarray(aabb.get_center()).tolist()
    obj.aabb_dims = np.asarray(aabb.get_extent()).tolist()
    try:
        obb = point_cloud.get_oriented_bounding_box()
    except RuntimeError:
        return
    obj.obb_center = np.asarray(obb.center).tolist()
    obj.obb_dims = np.asarray(obb.extent).tolist()
    obj.obb_rotation = np.asarray(obb.R).tolist()
    obj.obb_local_to_world = None
    obj.obb_world_to_local = None
    obj.obb_volume = float(np.prod(obb.extent))
    obj.obb_half_extents = (np.asarray(obb.extent) / 2).tolist()


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def parse_semantics(scene_dir, scene_mesh, txt_path, raw_scene_dir, dataset_dir, scene_name, scene_config):

    sim_settings = {
        "scene": scene_mesh,
        "default_agent": 0,
        "sensor_height": 1.5,
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
        "width": 1080,
        "height": 720,
        "enable_physics": False,
        "seed": 42,
        "lidar_fov": 360,
        "depth_img_for_lidar_n": 20,
        # "img_save_dir": save_dir,
        "raw_data_dir": raw_scene_dir,
        "dataset_dir": dataset_dir,
        "scene_name": scene_name,
    }

    sim_cfg = make_cfg(sim_settings, dataset_dir, raw_scene_dir, scene_name, scene_config)
    sim = habitat_sim.Simulator(sim_cfg)
    scene = sim.semantic_scene

    panoptic_object_list = []

    # load txt file
    with open(txt_path, "r") as file:
        lines = file.readlines()[1:]

    for i, line in enumerate(lines):
        object_desc = line.strip().split(",")
        panoptic_object_list.append(PanopticObject(object_desc))

    return PanopticScene(scene_dir, scene, panoptic_object_list), sim


def parse_args():
    parser = argparse.ArgumentParser(description="Create floor-0 HM3DSEM ground truth for all valid walk scenes.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Raw dataset root containing split directories")
    parser.add_argument("--walks-dir", required=True, type=Path, help="Rendered walk-output root")
    parser.add_argument("--split", action="append", dest="splits", help="Split directory to process; repeatable; defaults to all")
    parser.add_argument("--scene-id", action="append", dest="scene_ids", help="Scene ID to process; repeatable; defaults to all")
    parser.add_argument("--scene-config", type=Path, help="Explicit Habitat scene dataset config")
    parser.add_argument("--floor-metadata", type=Path, help="Optional CSV with Scene Name and Separation Heights")
    parser.add_argument("--region-votes", type=Path, help="Optional CSV with region vote labels")
    parser.add_argument("--region-labels", type=Path, help="Optional CSV with manual region labels")
    parser.add_argument("--hfov", type=float, default=90.0, help="Fallback horizontal field of view in degrees")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Point-cloud voxel size in metres")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth aligned frame")
    return parser.parse_args()


def load_hfov(scene_dir: Path, fallback: float) -> float:
    info_path = scene_dir / "camera_info.json"
    if not info_path.is_file():
        return fallback
    try:
        value = float(json.loads(info_path.read_text())["hfov_degrees"])
        return value if 0 < value < 180 else fallback
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return fallback


def validate_ground_truth_scene(scene, config):
    missing = validate_raw_scene(scene, config, require_poses=False)
    frames, frame_errors = discover_aligned_walk_frames(scene.walk_scene_dir)
    return frames, missing + frame_errors


def clear_derived_outputs(scene_dir: Path) -> None:
    """Remove only artifacts owned by this generator before replacing them."""
    for directory_name in ("objects", "regions"):
        directory = scene_dir / directory_name
        directory.mkdir(exist_ok=True)
        for path in directory.glob("*.ply"):
            path.unlink()
    for filename in ("scene_rgb.ply", "scene_panoptic.ply", "scene_info.json", "semantic_label_map.csv"):
        path = scene_dir / filename
        if path.exists():
            path.unlink()


def process_scene(scene, args) -> bool:
    config = resolve_scene_config(args.dataset_dir, scene.split, args.scene_config)
    frames, missing = validate_ground_truth_scene(scene, config)
    if missing:
        LOG.warning("SKIPPED %s: %s", scene.scene_id, "; ".join(missing))
        return False
    if args.frame_step < 1:
        LOG.warning("SKIPPED %s: --frame-step must be >= 1", scene.scene_id)
        return False

    sim = None
    try:
        panoptic_scene, sim = parse_semantics(
            scene.scene_id,
            str(scene.basis_mesh),
            str(scene.semantic_annotations),
            str(scene.raw_scene_dir),
            str(args.dataset_dir),
            scene.scene_name,
            str(config),
        )
        floor = floor_bounds_from_metadata(args.floor_metadata, scene.scene_id)
        if floor is None:
            floor = floor_bounds_from_semantic_scene(panoptic_scene.scene, floor_id=0)
        if floor is None:
            LOG.warning("SKIPPED %s: floor 0 bounds are unavailable from metadata or Habitat semantic levels", scene.scene_id)
            return False

        frame_poses = [(frame, read_camera_pose_hmp3d(frame.pose)) for frame in frames]
        non_floor_frames = [frame.stem for frame, pose in frame_poses if not pose_is_on_floor(pose, floor)]
        if non_floor_frames:
            LOG.warning(
                "SKIPPED %s: walk contains %d non-floor-0 frames (%s); re-render with gen_hm3dsem_walks_from_poses.py",
                scene.scene_id,
                len(non_floor_frames),
                ", ".join(non_floor_frames[:10]),
            )
            return False

        selected = frame_poses[:: args.frame_step]
        if not selected:
            LOG.warning("SKIPPED %s: no aligned floor-0 frames remain after --frame-step", scene.scene_id)
            return False
        clear_derived_outputs(scene.walk_scene_dir)
        hfov = load_hfov(scene.walk_scene_dir, args.hfov)
        rgb_pcd = o3d.geometry.PointCloud()
        panoptic_pcd = o3d.geometry.PointCloud()
        for index, (frame, camera_pose) in enumerate(tqdm(selected, desc=f"GT {scene.scene_id}", unit="frame")):
            rgb = cv2.imread(str(frame.rgb), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(frame.depth), cv2.IMREAD_ANYDEPTH)
            panoptic_ids = np.load(frame.semantic, allow_pickle=False)
            if rgb is None or depth is None:
                raise ValueError(f"unreadable RGB or depth frame: {frame.stem}")
            if panoptic_ids.ndim != 2 or panoptic_ids.shape != depth.shape or rgb.shape[:2] != depth.shape:
                raise ValueError(f"unaligned RGB/depth/semantic dimensions for frame {frame.stem}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            panoptic_rgb = np.zeros((*panoptic_ids.shape, 3), dtype=np.uint8)
            for object_id, object_rgb in panoptic_scene.id2rgb.items():
                panoptic_rgb[panoptic_ids == object_id] = object_rgb
            rgb_pcd += crop_point_cloud_to_floor(create_pcd_hmp3d(rgb, depth, camera_pose, hfov), floor)
            panoptic_pcd += crop_point_cloud_to_floor(create_pcd_hmp3d(panoptic_rgb, depth, camera_pose, hfov), floor)
            if (index + 1) % 500 == 0:
                rgb_pcd = rgb_pcd.voxel_down_sample(args.voxel_size)
                panoptic_pcd = panoptic_pcd.voxel_down_sample(args.voxel_size)

        rgb_pcd = rgb_pcd.voxel_down_sample(args.voxel_size)
        panoptic_pcd = panoptic_pcd.voxel_down_sample(args.voxel_size)
        if not len(panoptic_pcd.points):
            LOG.warning("SKIPPED %s: floor 0 produced no valid point-cloud points", scene.scene_id)
            return False
        o3d.io.write_point_cloud(str(scene.walk_scene_dir / "scene_rgb.ply"), rgb_pcd)
        o3d.io.write_point_cloud(str(scene.walk_scene_dir / "scene_panoptic.ply"), panoptic_pcd)

        objects_dir = scene.walk_scene_dir / "objects"
        regions_dir = scene.walk_scene_dir / "regions"
        point_colors = rgb_colors_for_panoptic_points(rgb_pcd, panoptic_pcd)
        pan_colors = np.asarray(panoptic_pcd.colors)
        points = np.asarray(panoptic_pcd.points)
        for obj in panoptic_scene.objects:
            object_mask = np.all(np.isclose(pan_colors, obj.rgb / 255.0), axis=1)
            if not np.any(object_mask):
                continue
            obj.mapped = True
            obj.points = points[object_mask]
            obj.colors = point_colors[object_mask]

        panoptic_scene.construct_regions()
        panoptic_scene.select_floor(floor)
        panoptic_scene.label_regions(args.region_votes, args.region_labels)
        selected_object_ids = {obj.id for floor_data in panoptic_scene.floors.values() for obj in floor_data.objects}
        for obj in panoptic_scene.objects:
            if obj.id not in selected_object_ids or not obj.mapped:
                continue
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(obj.points)
            object_pcd.colors = o3d.utility.Vector3dVector(obj.colors)
            update_object_geometry_from_points(obj, object_pcd)
            o3d.io.write_point_cloud(str(objects_dir / f"{obj.id}.ply"), object_pcd)
        for region in panoptic_scene.regions.values():
            o3d.io.write_point_cloud(str(regions_dir / f"{region.id}.ply"), region.region_point_cloud)
        panoptic_scene.write_metadata(str(scene.walk_scene_dir))
        generate_semantic_label_map(scene.walk_scene_dir, scene.semantic_annotations)
        absent = missing_output_paths(scene.walk_scene_dir)
        if absent:
            raise RuntimeError("required outputs were not written: " + ", ".join(absent))
        LOG.info("PROCESSED %s: %d aligned floor-0 frames and %d mapped objects", scene.scene_id, len(selected), len(panoptic_scene.scene_info["objects"]))
        return True
    except Exception as exc:
        LOG.exception("FAILED %s: %s", scene.scene_id, exc)
        return False
    finally:
        if sim is not None:
            sim.close()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scenes = discover_scene_paths(args.dataset_dir, args.walks_dir, splits=args.splits, scene_ids=args.scene_ids)
    if not scenes:
        LOG.error("No scene directories discovered under %s", args.dataset_dir)
        return
    succeeded = sum(process_scene(scene, args) for scene in scenes)
    LOG.info("Ground-truth generation complete: %d processed, %d skipped or failed", succeeded, len(scenes) - succeeded)


if __name__ == "__main__":
    main()
