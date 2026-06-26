#!/usr/bin/env python3
"""Convert one hm3dsem_walks scene directory to ROS1 and/or ROS2 bags.

The generator stores frame files but not timestamps or camera-info files. This
script therefore requires an FPS and derives pinhole intrinsics from the HOV-SG
HM3DSem loader convention: 90 degree horizontal FOV, centered principal point.
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
}

TFMESSAGE_TYPE = "tf2_msgs/msg/TFMessage"
TFMESSAGE_DEFINITION = "geometry_msgs/TransformStamped[] transforms\n"


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
        help="Convert saved Habitat camera pose to ROS optical frame, or keep raw Habitat axes",
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
    return args


def prepare_output(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def collect_frames(scene_dir: Path, max_frames: int | None) -> list[dict[str, Path | None]]:
    rgb = {p.stem: p for p in sorted((scene_dir / "rgb").glob("*.png"))}
    depth = {p.stem: p for p in sorted((scene_dir / "depth").glob("*.png"))}
    pose = {p.stem: p for p in sorted((scene_dir / "pose").glob("*.txt"))}
    semantic_dir = scene_dir / "semantic"
    semantic = {p.stem: p for p in sorted(semantic_dir.glob("*.npy"))} if semantic_dir.is_dir() else {}

    common = sorted(set(rgb) & set(depth) & set(pose))
    if not common:
        raise RuntimeError(f"No synchronized rgb/depth/pose frames found in {scene_dir}")

    missing_semantic = semantic_dir.is_dir() and set(common) - set(semantic)
    if missing_semantic:
        raise RuntimeError(f"Semantic directory exists but is missing {len(missing_semantic)} frame(s)")

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


def matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )

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
    return np.array(quat, dtype=np.float64)


def read_pose(path: Path, pose_frame: str) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
    if pose_frame == "optical":
        conversion = np.diag([1.0, -1.0, -1.0, 1.0])
        pose = pose @ conversion
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

    def odom(self, stamp_ns: int, frame_id: str, child_frame_id: str, pose: np.ndarray):
        point, orientation = self.pose_parts(pose)
        pose_msg = self.t["geometry_msgs/msg/Pose"](point, orientation)
        pose_cov = self.t["geometry_msgs/msg/PoseWithCovariance"](pose_msg, np.zeros(36, dtype=np.float64))
        zero = self.t["geometry_msgs/msg/Vector3"](0.0, 0.0, 0.0)
        twist = self.t["geometry_msgs/msg/Twist"](zero, zero)
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
    }
    if include_semantic:
        topics_and_types["semantic"] = (TOPICS["semantic"], "sensor_msgs/msg/Image")
        topics_and_types["semantic_info"] = (TOPICS["semantic_info"], "sensor_msgs/msg/CameraInfo")
    return {
        key: writer.add_connection(topic, typename, typestore=factory.typestore)
        for key, (topic, typename) in topics_and_types.items()
    }


def write_frames(writer, factory: MessageFactory, connections: dict[str, object], args: argparse.Namespace) -> None:
    frames = collect_frames(args.scene_dir, args.max_frames)
    include_semantic = frames[0]["semantic"] is not None

    desc = f"Writing {args.scene_dir.name}"
    for idx, frame in enumerate(tqdm(frames, desc=desc, unit="frame")):
        stamp = timestamp_ns(args.start_sec, args.fps, idx)

        color_bgr = cv2.imread(str(frame["rgb"]), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise RuntimeError(f"Could not read RGB image {frame['rgb']}")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        height, width = color.shape[:2]

        depth = cv2.imread(str(frame["depth"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Could not read depth image {frame['depth']}")
        if depth.dtype != np.uint16:
            raise RuntimeError(f"Expected uint16 depth PNG, got {depth.dtype} at {frame['depth']}")

        pose = read_pose(Path(frame["pose"]), args.pose_frame)

        messages = {
            "color": factory.image(stamp, args.color_frame_id, color, "rgb8"),
            "depth": factory.image(stamp, args.depth_frame_id, depth, "16UC1"),
            "color_info": factory.camera_info(stamp, args.color_frame_id, width, height),
            "depth_info": factory.camera_info(stamp, args.depth_frame_id, width, height),
            "odom": factory.odom(stamp, args.frame_id, args.color_frame_id, pose),
            "tf": factory.tf_message(
                [
                    factory.transform(stamp, args.frame_id, args.color_frame_id, pose),
                    factory.identity_transform(stamp, args.color_frame_id, args.depth_frame_id),
                ]
            ),
        }

        if include_semantic:
            semantic = np.load(frame["semantic"])
            if args.semantic_encoding == "32SC1":
                semantic_image = semantic.astype(np.int32, copy=False)
            else:
                if int(semantic.max()) > np.iinfo(np.uint16).max:
                    raise RuntimeError("Semantic label exceeds uint16 range; use --semantic-encoding 32SC1")
                semantic_image = semantic.astype(np.uint16, copy=False)
            messages["semantic"] = factory.image(stamp, args.semantic_frame_id, semantic_image, args.semantic_encoding)
            messages["semantic_info"] = factory.camera_info(stamp, args.semantic_frame_id, width, height)
            messages["tf"].transforms.append(
                factory.identity_transform(stamp, args.color_frame_id, args.semantic_frame_id)
            )

        for key, msg in messages.items():
            typename = msg.__msgtype__
            if isinstance(writer, Rosbag1Writer):
                data = factory.typestore.serialize_ros1(msg, typename)
            else:
                data = factory.typestore.serialize_cdr(msg, typename)
            writer.write(connections[key], stamp, data)


def convert(args: argparse.Namespace) -> None:
    frames = collect_frames(args.scene_dir, args.max_frames)
    include_semantic = frames[0]["semantic"] is not None

    if args.ros1_out:
        prepare_output(args.ros1_out, args.overwrite)
        factory = MessageFactory(Stores.ROS1_NOETIC)
        with Rosbag1Writer(args.ros1_out) as writer:
            connections = add_connections(writer, factory, include_semantic)
            write_frames(writer, factory, connections, args)

    if args.ros2_out:
        prepare_output(args.ros2_out, args.overwrite)
        factory = MessageFactory(Stores.ROS2_HUMBLE)
        with Rosbag2Writer(args.ros2_out, version=9) as writer:
            connections = add_connections(writer, factory, include_semantic)
            write_frames(writer, factory, connections, args)


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
