#!/usr/bin/env python3
"""Convert one hm3dsem_walks scene directory to ROS1 and/or ROS2 bags.

The generator stores frame files but not timestamps or camera-info files. This
script therefore requires an FPS and derives pinhole intrinsics from the HOV-SG
HM3DSem loader convention: 90 degree horizontal FOV, centered principal point.
Depth PNG values are millimetres. Saved camera poses are camera-to-Habitat-world
transforms; they are converted to a ROS map frame before serialization.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from rosbags.rosbag1 import Writer as Rosbag1Writer
from rosbags.rosbag2 import Writer as Rosbag2Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from tqdm import tqdm


TOPICS = {
    "color": "/camera/color/image_raw",
    "depth": "/camera/depth/image_raw",
    "color_info": "/camera/color/camera_info",
    "depth_info": "/camera/depth/camera_info",
    "semantic": "/camera/semantic/image_raw",
    "semantic_info": "/camera/semantic/camera_info",
    "odom": "/odom",
    "tf": "/tf",
    "tf_static": "/tf_static",
}

TFMESSAGE_TYPE = "tf2_msgs/msg/TFMessage"
TFMESSAGE_DEFINITION = "geometry_msgs/TransformStamped[] transforms\n"

# The generator writes Habitat world coordinates (x right, y up, z back).
# ROS map uses x forward, y left, z up.  This proper rotation changes the
# *world* convention, while CAMERA_HABITAT_TO_OPTICAL changes the camera's
# local convention to REP-103 optical (x right, y down, z forward).
WORLD_HABITAT_TO_ROS = np.array(
    [[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
CAMERA_HABITAT_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

# rosbag2 stores these as YAML strings.  Supplying them is important for
# /tf_static: late-joining TF consumers must receive the static calibration.
# rosbag2_transport parses a complete QoS profile from this YAML string; a
# partial profile (for example history/depth/reliability/durability only) is
# rejected by current ROS 2 releases while reading metadata.yaml.
def ros2_qos(depth: int, durability: str) -> str:
    return (
        f"- history: keep_last\n"
        f"  depth: {depth}\n"
        "  reliability: reliable\n"
        f"  durability: {durability}\n"
        "  deadline:\n"
        "    sec: 0\n"
        "    nsec: 0\n"
        "  lifespan:\n"
        "    sec: 0\n"
        "    nsec: 0\n"
        "  liveliness: system_default\n"
        "  liveliness_lease_duration:\n"
        "    sec: 0\n"
        "    nsec: 0\n"
        "  avoid_ros_namespace_conventions: false\n"
    )


ROS2_QOS_RELIABLE_VOLATILE = ros2_qos(depth=10, durability="volatile")
ROS2_QOS_TF_STATIC = ros2_qos(depth=1, durability="transient_local")
MAX_COMMON_ROS_TIME_SEC = 2_147_483_647  # ROS 2 builtin_interfaces/Time.sec is int32.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path, help="Path like data/hm3dsem_walks/val/00824-Dd4bFSTQ8gi")
    parser.add_argument("--ros1-out", type=Path, help="Output ROS1 .bag path")
    parser.add_argument("--ros2-out", type=Path, help="Output ROS2 bag directory")
    parser.add_argument("--fps", type=float, default=10.0, help="Synthetic frame rate for timestamps")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Synthetic start time in seconds")
    parser.add_argument("--frame-id", default="map", help="World frame for poses")
    parser.add_argument("--color-frame-id", default="camera_color_optical_frame")
    parser.add_argument("--depth-frame-id", default="camera_depth_optical_frame")
    parser.add_argument("--semantic-frame-id", default="camera_semantic_optical_frame")
    parser.add_argument(
        "--pose-frame",
        choices=("optical", "habitat"),
        default="optical",
        help="Use REP-103 optical camera axes (default), or keep Habitat camera axes in a non-optical child frame",
    )
    parser.add_argument(
        "--semantic-encoding",
        choices=("32SC1", "16UC1"),
        default="32SC1",
        help="Encoding for semantic label images loaded from .npy",
    )
    parser.add_argument("--max-frames", type=int, help="Limit frames for testing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.ros1_out and not args.ros2_out:
        parser.error("Provide --ros1-out and/or --ros2-out.")
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    if not math.isfinite(args.fps):
        parser.error("--fps must be finite.")
    if not math.isfinite(args.start_sec) or args.start_sec < 0:
        parser.error("--start-sec must be a finite, non-negative value.")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive.")
    return args


def prepare_output(path: Path, overwrite: bool) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {path.parent}")
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def collect_frames(scene_dir: Path, max_frames: int | None) -> list[dict[str, Path | None]]:
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

    def files(directory: str, pattern: str) -> dict[str, Path]:
        path = scene_dir / directory
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required input directory: {path}")
        result = {p.stem: p for p in sorted(path.glob(pattern))}
        if not result:
            raise FileNotFoundError(f"No {pattern} files found in required directory: {path}")
        return result

    rgb = files("rgb", "*.png")
    depth = files("depth", "*.png")
    pose = files("pose", "*.txt")
    semantic_dir = scene_dir / "semantic"
    semantic = {p.stem: p for p in sorted(semantic_dir.glob("*.npy"))} if semantic_dir.is_dir() else {}

    expected = set(rgb)
    for name, stream in (("depth", depth), ("pose", pose)):
        if set(stream) != expected:
            missing = sorted(expected - set(stream))
            extra = sorted(set(stream) - expected)
            raise RuntimeError(
                f"Unsynchronized {name} stream: {len(missing)} missing and {len(extra)} extra frame(s) relative to rgb"
            )

    if semantic_dir.is_dir() and set(semantic) != expected:
        missing = len(expected - set(semantic))
        extra = len(set(semantic) - expected)
        raise RuntimeError(f"Unsynchronized semantic stream: {missing} missing and {extra} extra frame(s) relative to rgb")

    common = sorted(expected)
    if max_frames is not None:
        common = common[:max_frames]

    return [
        {"rgb": rgb[stem], "depth": depth[stem], "pose": pose[stem], "semantic": semantic.get(stem)}
        for stem in common
    ]


def make_intrinsics(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hfov = math.radians(90.0)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * height / width)
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = height / (2.0 * math.tan(vfov / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    k = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], dtype=np.float64)
    r = np.eye(3, dtype=np.float64).reshape(-1)
    p = np.array([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    return k, r, p


def timestamp_ns(start_sec: float, fps: float, frame_idx: int) -> int:
    return int(round((start_sec + frame_idx / fps) * 1_000_000_000))


def make_timestamps(frame_count: int, start_sec: float, fps: float) -> list[int]:
    stamps = [timestamp_ns(start_sec, fps, idx) for idx in range(frame_count)]
    if stamps[-1] // 1_000_000_000 > MAX_COMMON_ROS_TIME_SEC:
        raise ValueError(f"Timestamps exceed the ROS 1/ROS 2 common Time range ({MAX_COMMON_ROS_TIME_SEC} seconds)")
    if any(later <= earlier for earlier, later in zip(stamps, stamps[1:])):
        raise ValueError("Synthetic timestamps are not strictly increasing; choose an FPS below 1e9")
    return stamps


def matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
        return quat / np.linalg.norm(quat)

    diag = np.diag(rot)
    axis = int(np.argmax(diag))
    if axis == 0:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        quat = [0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[2, 1] - rot[1, 2]) / s]
    elif axis == 1:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        quat = [(rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s, (rot[0, 2] - rot[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        quat = [(rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s, (rot[1, 0] - rot[0, 1]) / s]
    quat_array = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(quat_array)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Rotation matrix produced an invalid quaternion")
    return quat_array / norm


def read_pose(path: Path, pose_frame: str) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not read pose {path}: {exc}") from exc
    if values.size != 16:
        raise ValueError(f"Pose must contain exactly 16 values: {path}")
    pose = values.reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError(f"Pose contains non-finite values: {path}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"Pose has an invalid homogeneous last row: {path}")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"Pose has a non-rigid rotation matrix: {path}")

    pose = WORLD_HABITAT_TO_ROS @ pose
    if pose_frame == "optical":
        pose = pose @ CAMERA_HABITAT_TO_OPTICAL
    return pose


def ensure_message_types(typestore) -> None:
    if TFMESSAGE_TYPE not in typestore.types:
        typestore.register(get_types_from_msg(TFMESSAGE_DEFINITION, TFMESSAGE_TYPE))


class MessageFactory:
    def __init__(self, store: Stores) -> None:
        # The same helper builds messages for whichever ROS type store the
        # selected bag writer uses.
        self.typestore = get_typestore(store)
        ensure_message_types(self.typestore)
        self.t = self.typestore.types

    def time(self, stamp_ns: int):
        sec, nsec = divmod(stamp_ns, 1_000_000_000)
        return self.t["builtin_interfaces/msg/Time"](sec=int(sec), nanosec=int(nsec))

    def header(self, stamp_ns: int, frame_id: str):
        header_type = self.t["std_msgs/msg/Header"]
        kwargs = {"stamp": self.time(stamp_ns), "frame_id": frame_id}
        if "seq" in getattr(header_type, "__annotations__", {}):
            kwargs["seq"] = 0
        try:
            return header_type(**kwargs)
        except TypeError as exc:
            if "seq" in str(exc) and "seq" not in kwargs:
                kwargs["seq"] = 0
                return header_type(**kwargs)
            raise

    def image(self, stamp_ns: int, frame_id: str, image: np.ndarray, encoding: str):
        height, width = image.shape[:2]
        if image.ndim == 2:
            step = width * image.dtype.itemsize
        else:
            step = width * image.shape[2] * image.dtype.itemsize
        data = np.frombuffer(np.ascontiguousarray(image).tobytes(), dtype=np.uint8)
        return self.t["sensor_msgs/msg/Image"](
            header=self.header(stamp_ns, frame_id),
            height=height,
            width=width,
            encoding=encoding,
            is_bigendian=0,
            step=step,
            data=data,
        )

    def camera_info(self, stamp_ns: int, frame_id: str, width: int, height: int):
        k, r, p = make_intrinsics(width, height)
        roi = self.t["sensor_msgs/msg/RegionOfInterest"](0, 0, 0, 0, False)
        camera_info_type = self.t["sensor_msgs/msg/CameraInfo"]
        kwargs = {
            "header": self.header(stamp_ns, frame_id),
            "height": height,
            "width": width,
            "distortion_model": "plumb_bob",
            "binning_x": 0,
            "binning_y": 0,
            "roi": roi,
        }
        ann = getattr(camera_info_type, "__annotations__", {})
        if "d" in ann or "D" not in ann:
            kwargs.update({"d": np.zeros(5, dtype=np.float64), "k": k, "r": r, "p": p})
        else:
            kwargs.update({"D": np.zeros(5, dtype=np.float64), "K": k, "R": r, "P": p})
        try:
            return camera_info_type(**kwargs)
        except TypeError as exc:
            if "unexpected keyword argument 'd'" in str(exc):
                kwargs["D"] = kwargs.pop("d")
                kwargs["K"] = kwargs.pop("k")
                kwargs["R"] = kwargs.pop("r")
                kwargs["P"] = kwargs.pop("p")
                return camera_info_type(**kwargs)
            raise

    def pose_parts(self, pose: np.ndarray):
        quat = matrix_to_quaternion(pose[:3, :3])
        point = self.t["geometry_msgs/msg/Point"](*map(float, pose[:3, 3]))
        orientation = self.t["geometry_msgs/msg/Quaternion"](*map(float, quat))
        return point, orientation

    def odom(
        self,
        stamp_ns: int,
        frame_id: str,
        child_frame_id: str,
        pose: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ):
        point, orientation = self.pose_parts(pose)
        pose_msg = self.t["geometry_msgs/msg/Pose"](point, orientation)
        pose_cov = self.t["geometry_msgs/msg/PoseWithCovariance"](pose_msg, np.zeros(36, dtype=np.float64))
        linear = self.t["geometry_msgs/msg/Vector3"](*map(float, linear_velocity))
        angular = self.t["geometry_msgs/msg/Vector3"](*map(float, angular_velocity))
        twist = self.t["geometry_msgs/msg/Twist"](linear, angular)
        twist_cov = self.t["geometry_msgs/msg/TwistWithCovariance"](twist, np.zeros(36, dtype=np.float64))
        return self.t["nav_msgs/msg/Odometry"](
            header=self.header(stamp_ns, frame_id),
            child_frame_id=child_frame_id,
            pose=pose_cov,
            twist=twist_cov,
        )

    def transform(self, stamp_ns: int, frame_id: str, child_frame_id: str, pose: np.ndarray):
        point, orientation = self.pose_parts(pose)
        trans = self.t["geometry_msgs/msg/Vector3"](point.x, point.y, point.z)
        transform = self.t["geometry_msgs/msg/Transform"](trans, orientation)
        return self.t["geometry_msgs/msg/TransformStamped"](
            header=self.header(stamp_ns, frame_id),
            child_frame_id=child_frame_id,
            transform=transform,
        )

    def identity_transform(self, stamp_ns: int, frame_id: str, child_frame_id: str):
        pose = np.eye(4, dtype=np.float64)
        return self.transform(stamp_ns, frame_id, child_frame_id, pose)

    def tf_message(self, transforms: Iterable[object]):
        return self.t[TFMESSAGE_TYPE](list(transforms))


def add_connections(writer, factory: MessageFactory, include_semantic: bool) -> dict[str, object]:
    topics_and_types = {
        "color": (TOPICS["color"], "sensor_msgs/msg/Image"),
        "depth": (TOPICS["depth"], "sensor_msgs/msg/Image"),
        "color_info": (TOPICS["color_info"], "sensor_msgs/msg/CameraInfo"),
        "depth_info": (TOPICS["depth_info"], "sensor_msgs/msg/CameraInfo"),
        "odom": (TOPICS["odom"], "nav_msgs/msg/Odometry"),
        "tf": (TOPICS["tf"], TFMESSAGE_TYPE),
        "tf_static": (TOPICS["tf_static"], TFMESSAGE_TYPE),
    }
    if include_semantic:
        topics_and_types["semantic"] = (TOPICS["semantic"], "sensor_msgs/msg/Image")
        topics_and_types["semantic_info"] = (TOPICS["semantic_info"], "sensor_msgs/msg/CameraInfo")
    connections = {}
    for key, (topic, typename) in topics_and_types.items():
        if isinstance(writer, Rosbag1Writer):
            connections[key] = writer.add_connection(
                topic,
                typename,
                typestore=factory.typestore,
                latching=1 if key == "tf_static" else None,
            )
        else:
            connections[key] = writer.add_connection(
                topic,
                typename,
                typestore=factory.typestore,
                offered_qos_profiles=ROS2_QOS_TF_STATIC if key == "tf_static" else ROS2_QOS_RELIABLE_VOLATILE,
            )
    return connections


def validate_frame_ids(args: argparse.Namespace, include_semantic: bool) -> None:
    frame_ids = [args.frame_id, args.color_frame_id, args.depth_frame_id]
    if include_semantic:
        frame_ids.append(args.semantic_frame_id)
    for frame_id in frame_ids:
        if not frame_id or frame_id.startswith("/") or any(char.isspace() for char in frame_id):
            raise ValueError(f"Invalid ROS frame id: {frame_id!r}")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("World, color, depth, and semantic frame IDs must be distinct")
    sensor_frame_ids = [args.color_frame_id, args.depth_frame_id]
    if include_semantic:
        sensor_frame_ids.append(args.semantic_frame_id)
    if args.pose_frame == "habitat" and any("optical" in frame_id for frame_id in sensor_frame_ids):
        raise ValueError(
            "--pose-frame habitat keeps Habitat camera axes; use non-optical sensor frame IDs or use --pose-frame optical"
        )


def read_images(frame: dict[str, Path | None], include_semantic: bool, semantic_encoding: str):
    color_bgr = cv2.imread(str(frame["rgb"]), cv2.IMREAD_UNCHANGED)
    if color_bgr is None:
        raise RuntimeError(f"Could not read RGB image {frame['rgb']}")
    if color_bgr.dtype != np.uint8 or color_bgr.ndim != 3 or color_bgr.shape[2] not in (3, 4):
        raise ValueError(f"RGB image must be uint8 BGR/BGRA with 3 or 4 channels: {frame['rgb']}")
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB if color_bgr.shape[2] == 3 else cv2.COLOR_BGRA2RGB)
    height, width = color.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"RGB image has invalid dimensions: {frame['rgb']}")

    depth = cv2.imread(str(frame["depth"]), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth image {frame['depth']}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Depth image must be a single-channel uint16 PNG in millimetres: {frame['depth']}")
    if depth.shape != (height, width):
        raise ValueError(f"RGB/depth dimensions differ at {frame['rgb']} and {frame['depth']}")

    semantic_image = None
    if include_semantic:
        try:
            semantic = np.load(frame["semantic"], allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read semantic labels {frame['semantic']}: {exc}") from exc
        if semantic.ndim != 2 or semantic.shape != (height, width) or not np.issubdtype(semantic.dtype, np.integer):
            raise ValueError(f"Semantic labels must be a 2-D integer image matching RGB dimensions: {frame['semantic']}")
        min_label, max_label = int(semantic.min()), int(semantic.max())
        if min_label < 0:
            raise ValueError(f"Semantic labels must be non-negative: {frame['semantic']}")
        limit = np.iinfo(np.int32).max if semantic_encoding == "32SC1" else np.iinfo(np.uint16).max
        if max_label > limit:
            raise ValueError(f"Semantic label exceeds {semantic_encoding} range: {frame['semantic']}")
        semantic_image = semantic.astype(np.int32 if semantic_encoding == "32SC1" else np.uint16, copy=False)
    return color, depth, semantic_image


def velocities_from_poses(previous: np.ndarray | None, current: np.ndarray, dt_ns: int | None) -> tuple[np.ndarray, np.ndarray]:
    if previous is None or dt_ns is None or dt_ns <= 0:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    dt = dt_ns / 1_000_000_000.0
    # nav_msgs/Odometry expresses twist in child_frame_id coordinates.
    linear = current[:3, :3].T @ ((current[:3, 3] - previous[:3, 3]) / dt)
    relative_rotation = previous[:3, :3].T @ current[:3, :3]
    quaternion = matrix_to_quaternion(relative_rotation)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    sin_half_angle = np.linalg.norm(quaternion[:3])
    if sin_half_angle < 1e-9:
        angular = np.zeros(3, dtype=np.float64)
    else:
        angle = 2.0 * math.atan2(sin_half_angle, quaternion[3])
        axis = quaternion[:3] / sin_half_angle
        angular = axis * (angle / dt)
    if not np.isfinite(linear).all() or not np.isfinite(angular).all():
        raise ValueError("Pose sequence produced a non-finite odometry twist")
    return linear, angular


def write_frames(
    writer,
    factory: MessageFactory,
    connections: dict[str, object],
    args: argparse.Namespace,
    frames: list[dict[str, Path | None]],
) -> None:
    include_semantic = frames[0]["semantic"] is not None
    stamps = make_timestamps(len(frames), args.start_sec, args.fps)
    previous_pose = None
    previous_stamp = None

    desc = f"Writing {args.scene_dir.name}"
    for idx, frame in enumerate(tqdm(frames, desc=desc, unit="frame")):
        stamp = stamps[idx]
        color, depth, semantic_image = read_images(frame, include_semantic, args.semantic_encoding)
        height, width = color.shape[:2]
        pose = read_pose(Path(frame["pose"]), args.pose_frame)
        linear_velocity, angular_velocity = velocities_from_poses(
            previous_pose, pose, None if previous_stamp is None else stamp - previous_stamp
        )

        messages = {
            "color": factory.image(stamp, args.color_frame_id, color, "rgb8"),
            "depth": factory.image(stamp, args.depth_frame_id, depth, "16UC1"),
            "color_info": factory.camera_info(stamp, args.color_frame_id, width, height),
            "depth_info": factory.camera_info(stamp, args.depth_frame_id, width, height),
            "odom": factory.odom(
                stamp, args.frame_id, args.color_frame_id, pose, linear_velocity, angular_velocity
            ),
            "tf": factory.tf_message([factory.transform(stamp, args.frame_id, args.color_frame_id, pose)]),
        }

        if include_semantic:
            messages["semantic"] = factory.image(stamp, args.semantic_frame_id, semantic_image, args.semantic_encoding)
            messages["semantic_info"] = factory.camera_info(stamp, args.semantic_frame_id, width, height)

        if idx == 0:
            static_transforms = [factory.identity_transform(stamp, args.color_frame_id, args.depth_frame_id)]
            if include_semantic:
                static_transforms.append(factory.identity_transform(stamp, args.color_frame_id, args.semantic_frame_id))
            messages = {"tf_static": factory.tf_message(static_transforms), **messages}

        for key, msg in messages.items():
            typename = msg.__msgtype__
            try:
                if isinstance(writer, Rosbag1Writer):
                    data = factory.typestore.serialize_ros1(msg, typename)
                else:
                    data = factory.typestore.serialize_cdr(msg, typename)
                writer.write(connections[key], stamp, data)
            except Exception as exc:
                raise RuntimeError(f"Failed to serialize/write {typename} on {TOPICS[key]} at {stamp} ns") from exc
        previous_pose = pose
        previous_stamp = stamp


def convert(args: argparse.Namespace) -> None:
    frames = collect_frames(args.scene_dir, args.max_frames)
    include_semantic = frames[0]["semantic"] is not None
    validate_frame_ids(args, include_semantic)
    # Detect rounding collisions before replacing an existing output.
    make_timestamps(len(frames), args.start_sec, args.fps)

    if args.ros1_out:
        prepare_output(args.ros1_out, args.overwrite)
        factory = MessageFactory(Stores.ROS1_NOETIC)
        with Rosbag1Writer(args.ros1_out) as writer:
            connections = add_connections(writer, factory, include_semantic)
            write_frames(writer, factory, connections, args, frames)

    if args.ros2_out:
        prepare_output(args.ros2_out, args.overwrite)
        factory = MessageFactory(Stores.ROS2_HUMBLE)
        # rosbags writes the portable rosbag2 SQLite3/metadata-v8 format.
        # Its Writer has no `version` parameter; passing one prevented every
        # ROS 2 conversion before a bag directory could be created.
        with Rosbag2Writer(args.ros2_out) as writer:
            connections = add_connections(writer, factory, include_semantic)
            write_frames(writer, factory, connections, args, frames)


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
