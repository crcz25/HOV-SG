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
    discover_aligned_walk_frames,
    discover_scene_paths,
    floor_index_for_height,
    floor_separations_from_camera_info,
    floor_separations_from_metadata,
    floor_separations_from_navmesh,
    missing_output_paths,
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
        # Whether the annotation could be matched to Habitat geometry in the
        # complete semantic scene, independently of the rendered walk.
        self.habitat_matched = False

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


def is_finite_vector(value, size=3):
    """Return whether ``value`` is a finite numeric vector of ``size`` entries."""
    if value is None:
        return False
    try:
        array = np.asarray(habitat_attr_value(value), dtype=float)
    except (TypeError, ValueError):
        return False
    return array.shape == (size,) and bool(np.isfinite(array).all())


def object_centroid(obj):
    """Return the representative world position of an object.

    Observed objects use the mean of their mapped points, which is the geometry
    the walk actually saw.  Unobserved objects fall back to the complete-scene
    OBB centre and then to its AABB centre, since no point cloud exists for
    them.  The same value drives the height-based floor fallback.
    """
    points = getattr(obj, "points", None)
    if points is not None and len(points):
        return np.mean(np.asarray(points, dtype=float), axis=0).tolist()
    for box_center in (obj.obb_center, obj.aabb_center):
        if is_finite_vector(box_center):
            return [float(value) for value in np.asarray(habitat_attr_value(box_center), dtype=float)]
    return None


def has_complete_box_geometry(obj):
    """Return whether both boxes of an object are exportable 3-D extents."""
    return all(
        is_finite_vector(value)
        for value in (obj.aabb_center, obj.aabb_dims, obj.obb_center, obj.obb_dims)
    )


def serialize_object(obj):
    """Serialize one object for ``scene_info.json``.

    Shared by the trajectory-visible ``objects`` collection and the
    complete-scene ``all_objects`` collection so both stay identical field for
    field; ``observed_in_walk`` is the only thing that distinguishes them.
    """
    return {
        "id": obj.id,
        "category": obj.category,
        "hex": obj.hex,
        "region_id": obj.region_id,
        "floor_id": obj.floor_id,
        "observed_in_walk": bool(obj.mapped),
        "centroid": object_centroid(obj),
        "aabb_center": obj.aabb_center,
        "aabb_dims": obj.aabb_dims,
        "obb_center": obj.obb_center,
        "obb_dims": obj.obb_dims,
        "obb_rotation": obj.obb_rotation,
        "obb_local_to_world": obj.obb_local_to_world,
        "obb_world_to_local": obj.obb_world_to_local,
        "obb_volume": obj.obb_volume,
        "obb_half_extents": obj.obb_half_extents,
    }


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
        # ``region_points`` are reconstructed directly in Habitat world
        # coordinates (see ``read_camera_pose_hmp3d``).  Keep the centroid in
        # that same frame so it is comparable to a HOV-SG room centroid.
        self.centroid = None
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
        self.centroid = np.mean(self.region_points, axis=0).tolist()

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

        self.scene_info = {"levels": [], "regions": [], "objects": [], "all_objects": []}
        # Population statistics of the complete semantic scene, reported once the
        # walk has been mapped.
        self.habitat_object_count = 0

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
        """Copy complete-scene Habitat geometry onto the annotated objects.

        Every object of the semantic scene is visited, not only the ones the
        walk observed, so unobserved annotations still carry their source AABB
        and OBB.
        """
        for obj in self.scene.objects:
            if obj is None:
                LOG.warning("Ignoring an empty Habitat semantic object slot")
                continue
            self.habitat_object_count += 1
            try:
                obj_id = int(str(obj.id).rsplit("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                LOG.warning("Ignoring Habitat semantic object with an unparseable ID: %s", obj.id)
                continue

            if obj_id in self.id2obj_idx:
                panoptic_object = self.objects[self.id2obj_idx[obj_id]]
                panoptic_object.aabb_center = habitat_attr_value(obj.aabb.center)
                panoptic_object.aabb_dims = habitat_attr_value(obj.aabb.size)
                panoptic_object.obb_center = habitat_attr_value(obj.obb.center)
                panoptic_object.obb_dims = habitat_attr_value(obj.obb.sizes)
                panoptic_object.obb_rotation = habitat_attr_value(obj.obb.rotation)
                panoptic_object.obb_local_to_world = habitat_attr_value(obj.obb.local_to_world)
                panoptic_object.obb_world_to_local = habitat_attr_value(obj.obb.world_to_local)
                panoptic_object.obb_volume = habitat_attr_value(obj.obb.volume)
                panoptic_object.obb_half_extents = habitat_attr_value(obj.obb.half_extents)
                panoptic_object.habitat_matched = has_complete_box_geometry(panoptic_object)
                if not panoptic_object.habitat_matched:
                    LOG.warning(
                        "Habitat geometry for semantic object %s is incomplete or non-finite", obj_id
                    )

        unmatched = sorted(obj.id for obj in self.objects if not obj.habitat_matched)
        if unmatched:
            LOG.warning(
                "%d semantic annotation entries have no valid Habitat geometry: %s",
                len(unmatched),
                ", ".join(str(obj_id) for obj_id in unmatched[:20]) + (" ..." if len(unmatched) > 20 else ""),
            )

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
    
    def assign_regions_to_floors(self, separations):
        """Distribute mapped regions and objects across every storey.

        ``separations`` holds N+1 boundaries for N storeys.  A region belongs to
        the storey containing its mean height; regions outside every storey
        (annotation noise, roof geometry) are dropped rather than forced onto a
        floor they do not belong to.
        """
        floor_count = len(separations) - 1
        for floor_idx in range(floor_count):
            self.floors[floor_idx] = PanopticLevel(floor_idx, separations[floor_idx], separations[floor_idx + 1])

        unassigned = []
        for region in list(self.regions.values()):
            floor_idx = floor_index_for_height(region.mean_height, separations)
            if floor_idx is None:
                unassigned.append(region.id)
                continue
            region.floor_id = floor_idx
            floor = self.floors[floor_idx]
            floor.regions.append(region)
            floor.objects.extend(region.objects)
            for region_obj in region.objects:
                self.objects[self.id2obj_idx[region_obj.id]].floor_id = floor_idx
        if unassigned:
            LOG.warning("Regions outside every storey were dropped: %s", unassigned)
        self.regions = {region.id: region for region in self.regions.values() if region.floor_id is not None}

        # Replace the nominal boundaries with the extent of the regions actually
        # on each storey, and drop storeys the walk never observed.
        for floor_idx in list(self.floors):
            regions = self.floors[floor_idx].regions
            if not regions:
                del self.floors[floor_idx]
                continue
            self.floors[floor_idx].lower = float(np.mean([region.min_height for region in regions]))
            self.floors[floor_idx].upper = float(np.mean([region.max_height for region in regions]))

    def assign_remaining_object_floors(self, separations):
        """Give every remaining annotated object a storey, or leave it unset.

        ``assign_regions_to_floors`` only reaches objects that the walk observed,
        because it works through regions reconstructed from observed points.  An
        object that was never observed still has an annotated region and a
        complete-scene position, so its storey is recovered from the storey of
        its own region first and from its centroid height second.  Nothing is
        guessed: an object that matches neither is reported and left unassigned.
        """
        region_floors = {
            int(region.id): region.floor_id for region in self.regions.values() if region.floor_id is not None
        }
        from_region, from_height, unassigned = 0, 0, []
        for obj in self.objects:
            if obj.floor_id is not None:
                continue
            region_floor = region_floors.get(int(obj.region_id))
            if region_floor is not None:
                obj.floor_id = region_floor
                from_region += 1
                continue
            centroid = object_centroid(obj)
            floor_idx = floor_index_for_height(centroid[1], separations) if centroid is not None else None
            if floor_idx is None:
                unassigned.append(obj.id)
                continue
            obj.floor_id = floor_idx
            from_height += 1
        if from_region or from_height:
            LOG.info(
                "Floors recovered for %d unassigned object(s): %d from their annotated region, %d from their height",
                from_region + from_height,
                from_region,
                from_height,
            )
        if unassigned:
            LOG.warning(
                "%d object(s) keep floor_id=null: neither their annotated region nor their height falls on a storey: %s",
                len(unassigned),
                ", ".join(str(obj_id) for obj_id in unassigned[:20]) + (" ..." if len(unassigned) > 20 else ""),
            )

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
                "centroid": region_obj.centroid,
                "voted_category": region_obj.voted_category,
                "category": region_obj.category,
                "min_height": region_obj.min_height,
                "max_height": region_obj.max_height,
                "mean_height": region_obj.mean_height,
                "bev_region_points": np.array(region_obj.bev_region_point_cloud.points).tolist(),
            }
            region_item["objects"] = [obj.id for obj in region_obj.objects if obj.mapped]
            self.scene_info["regions"].append(region_item)

        # "objects" keeps its meaning: the trajectory-visible objects that the
        # HOV-SG evaluator consumes, each backed by an <id>.ply point cloud.
        # "all_objects" is a superset holding every annotated object of the
        # complete semantic scene, observed or not, and is evaluation-only
        # metadata that no HOV-SG graph stage reads.
        selected_object_ids = {obj.id for floor in self.floors.values() for obj in floor.objects}
        skipped = []
        for obj in self.objects:
            if not (obj.mapped or obj.habitat_matched):
                skipped.append(obj.id)
                continue
            object_item = serialize_object(obj)
            self.scene_info["all_objects"].append(object_item)
            if obj.mapped and obj.id in selected_object_ids:
                self.scene_info["objects"].append(object_item)
        if skipped:
            LOG.warning(
                "%d annotated object(s) excluded from all_objects: never observed and without valid Habitat geometry: %s",
                len(skipped),
                ", ".join(str(obj_id) for obj_id in skipped[:20]) + (" ..." if len(skipped) > 20 else ""),
            )

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
    """Replace source-wide boxes with boxes computed from the observed geometry."""
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
    parser = argparse.ArgumentParser(description="Create multi-storey HM3DSEM ground truth for all valid walk scenes.")
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
        # The annotated scenes carry authoritative separation heights; prefer
        # them, then whatever the renderer recorded, then the navmesh.
        separations = floor_separations_from_metadata(args.floor_metadata, scene.scene_id)
        separation_source = f"metadata:{args.floor_metadata}"
        if separations is None:
            separations = floor_separations_from_camera_info(scene.walk_scene_dir)
            separation_source = "camera_info.json"
        if separations is None:
            separations = floor_separations_from_navmesh(sim.pathfinder)
            separation_source = "navmesh_height_histogram"
        if separations is None:
            LOG.warning(
                "SKIPPED %s: floor separation heights are unavailable from metadata, camera_info.json, or the navmesh",
                scene.scene_id,
            )
            return False
        LOG.info(
            "%s: %d floor(s), separations %s from %s",
            scene.scene_id,
            len(separations) - 1,
            [round(value, 3) for value in separations],
            separation_source,
        )

        frame_poses = [(frame, read_camera_pose_hmp3d(frame.pose)) for frame in frames]
        selected = frame_poses[:: args.frame_step]
        if not selected:
            LOG.warning("SKIPPED %s: no aligned frames remain after --frame-step", scene.scene_id)
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
            # The whole building is kept: cropping to one storey is what made
            # multi-storey walks unusable.
            rgb_pcd += create_pcd_hmp3d(rgb, depth, camera_pose, hfov)
            panoptic_pcd += create_pcd_hmp3d(panoptic_rgb, depth, camera_pose, hfov)
            if (index + 1) % 500 == 0:
                rgb_pcd = rgb_pcd.voxel_down_sample(args.voxel_size)
                panoptic_pcd = panoptic_pcd.voxel_down_sample(args.voxel_size)

        rgb_pcd = rgb_pcd.voxel_down_sample(args.voxel_size)
        panoptic_pcd = panoptic_pcd.voxel_down_sample(args.voxel_size)
        if not len(panoptic_pcd.points):
            LOG.warning("SKIPPED %s: the walk produced no valid point-cloud points", scene.scene_id)
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
        panoptic_scene.assign_regions_to_floors(separations)
        # Objects the walk never saw have no region point cloud, so their storey
        # is recovered separately before the metadata is written.
        panoptic_scene.assign_remaining_object_floors(separations)
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
        all_objects = panoptic_scene.scene_info["all_objects"]
        LOG.info(
            "PROCESSED %s: %d frames, %d floor(s), %d regions, %d mapped objects",
            scene.scene_id,
            len(selected),
            len(panoptic_scene.scene_info["levels"]),
            len(panoptic_scene.scene_info["regions"]),
            len(panoptic_scene.scene_info["objects"]),
        )
        LOG.info(
            "%s complete-scene GT: %d Habitat semantic objects, %d annotation entries, "
            "%d in all_objects (%d observed in the walk), %d without a floor",
            scene.scene_id,
            panoptic_scene.habitat_object_count,
            len(panoptic_scene.objects),
            len(all_objects),
            sum(item["observed_in_walk"] for item in all_objects),
            sum(item["floor_id"] is None for item in all_objects),
        )
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
        raise SystemExit(1)
    succeeded = sum(process_scene(scene, args) for scene in scenes)
    failed = len(scenes) - succeeded
    LOG.info("Ground-truth generation complete: %d processed, %d skipped or failed", succeeded, failed)
    # A skipped or failed scene must be visible to the caller's exit status.
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
