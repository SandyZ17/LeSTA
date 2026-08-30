#!/usr/bin/env python3
"""Write decoded LeSTA frames and auxiliary CSV data to a ROS2 bag.

This intentionally writes only standard ROS2 message types, so the result can
be replayed without the ROS1 Velodyne message definitions or ros1_bridge.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2, PointField
from tf2_msgs.msg import TFMessage


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def stamp(ns: int) -> Time:
    msg = Time()
    msg.sec = ns // 1_000_000_000
    msg.nanosec = ns % 1_000_000_000
    return msg


def transform(row) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp(int(row["stamp_ns"]))
    msg.header.frame_id = row["parent_frame"]
    msg.child_frame_id = row["child_frame"]
    msg.transform.translation.x = float(row["translation_x"])
    msg.transform.translation.y = float(row["translation_y"])
    msg.transform.translation.z = float(row["translation_z"])
    msg.transform.rotation.x = float(row["rotation_x"])
    msg.transform.rotation.y = float(row["rotation_y"])
    msg.transform.rotation.z = float(row["rotation_z"])
    msg.transform.rotation.w = float(row["rotation_w"])
    return msg


def pointcloud(path: Path, stamp_ns: int, frame_id: str) -> PointCloud2:
    raw = path.read_bytes()
    marker = raw.find(b"DATA binary")
    if marker < 0:
        raise ValueError(f"expected binary PCD: {path}")
    data_start = raw.find(b"\n", marker) + 1
    points = np.frombuffer(raw[data_start:], dtype="<f4").reshape(-1, 4)
    msg = PointCloud2()
    msg.header.stamp = stamp(stamp_ns)
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
        for name, offset in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = bool(np.isfinite(points).all())
    msg.data = points.astype("<f4", copy=False).tobytes()
    return msg


def imu(row) -> Imu:
    msg = Imu()
    msg.header.stamp = stamp(int(row["stamp_ns"]))
    msg.header.frame_id = "imu_link"
    msg.linear_acceleration.x = float(row["accel_x"])
    msg.linear_acceleration.y = float(row["accel_y"])
    msg.linear_acceleration.z = float(row["accel_z"])
    msg.angular_velocity.x = float(row["gyro_x"])
    msg.angular_velocity.y = float(row["gyro_y"])
    msg.angular_velocity.z = float(row["gyro_z"])
    msg.orientation.x = float(row["orientation_x"])
    msg.orientation.y = float(row["orientation_y"])
    msg.orientation.z = float(row["orientation_z"])
    msg.orientation.w = float(row["orientation_w"])
    return msg


def odometry(row) -> Odometry:
    msg = Odometry()
    msg.header.stamp = stamp(int(row["stamp_ns"]))
    msg.header.frame_id = "odom"
    msg.child_frame_id = "base_link"
    msg.pose.pose.position.x = float(row["pose_x"])
    msg.pose.pose.position.y = float(row["pose_y"])
    msg.pose.pose.position.z = float(row["pose_z"])
    msg.twist.twist.linear.x = float(row["linear_x"])
    msg.twist.twist.linear.y = float(row["linear_y"])
    msg.twist.twist.linear.z = float(row["linear_z"])
    msg.twist.twist.angular.x = float(row["angular_x"])
    msg.twist.twist.angular.y = float(row["angular_y"])
    msg.twist.twist.angular.z = float(row["angular_z"])
    return msg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")

    manifest = rows(args.frame_dir / "manifest.csv")[::args.frame_stride]
    imu_rows = rows(args.frame_dir / "imu.csv")
    odom_rows = rows(args.frame_dir / "odometry_wheel.csv")
    tf_rows = rows(args.frame_dir / "tf.csv")
    tf_static_rows = rows(args.frame_dir / "tf_static.csv")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.output), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topics = {
        "/velodyne/points": "sensor_msgs/msg/PointCloud2",
        "/imu/data": "sensor_msgs/msg/Imu",
        "/odom": "nav_msgs/msg/Odometry",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/clock": "rosgraph_msgs/msg/Clock",
    }
    for name, type_name in topics.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=name, type=type_name, serialization_format="cdr"))

    def write(name, message, ns):
        writer.write(name, serialize_message(message), ns)

    for row in imu_rows:
        ns = int(row["stamp_ns"])
        write("/imu/data", imu(row), ns)
    for row in odom_rows:
        ns = int(row["stamp_ns"])
        write("/odom", odometry(row), ns)
    for row in tf_rows:
        ns = int(row["stamp_ns"])
        write("/tf", TFMessage(transforms=[transform(row)]), ns)
    for row in tf_static_rows:
        ns = int(row["stamp_ns"])
        static_tf = TFMessage(transforms=[transform(row)])
        # Replay the fixed sensor edge on /tf.  Writing it as a normal TF
        # message avoids a rosbag2 QoS mismatch with volatile subscribers.
        write("/tf", static_tf, ns)
    for row in manifest:
        ns = int(row["stamp_ns"])
        write("/velodyne/points", pointcloud(
            args.frame_dir / row["pcd_file"], ns, row["frame_id"]), ns)
        clock = Clock()
        clock.clock = stamp(ns)
        write("/clock", clock, ns)

    print(f"Wrote {len(manifest)} pointcloud frames to {args.output}")
    print(f"Auxiliary messages: imu={len(imu_rows)} odom={len(odom_rows)} "
          f"tf={len(tf_rows)} tf_static={len(tf_static_rows)}")


if __name__ == "__main__":
    main()
