#!/usr/bin/env python3
"""Extract auxiliary ROS 1 topics and align them to decoded LiDAR frames."""

import bisect
import csv
import os
import sys

import numpy as np
import rosbag
from scipy.interpolate import PchipInterpolator


def stamp_ns(message, bag_time):
    header = getattr(message, "header", None)
    return header.stamp.to_nsec() if header and header.stamp else bag_time.to_nsec()


def write_csv(path, header, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def extract(bag_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    imu = []
    odom = []
    cmd_vel = []
    tf = []
    tf_static = []

    with rosbag.Bag(bag_path, "r") as bag:
        for topic, message, bag_time in bag.read_messages():
            timestamp = stamp_ns(message, bag_time)
            if topic == "/imu/data":
                imu.append([
                    timestamp,
                    message.linear_acceleration.x, message.linear_acceleration.y,
                    message.linear_acceleration.z,
                    message.angular_velocity.x, message.angular_velocity.y,
                    message.angular_velocity.z,
                    message.orientation.x, message.orientation.y,
                    message.orientation.z, message.orientation.w,
                ])
            elif topic == "/odometry/wheel":
                odom.append([
                    timestamp,
                    message.pose.pose.position.x, message.pose.pose.position.y,
                    message.pose.pose.position.z,
                    message.twist.twist.linear.x, message.twist.twist.linear.y,
                    message.twist.twist.linear.z,
                    message.twist.twist.angular.x, message.twist.twist.angular.y,
                    message.twist.twist.angular.z,
                ])
            elif topic == "/cmd_vel":
                cmd_vel.append([
                    timestamp, message.linear.x, message.linear.y,
                    message.linear.z, message.angular.x, message.angular.y,
                    message.angular.z,
                ])
            elif topic in ("/tf", "/tf_static"):
                target = tf_static if topic == "/tf_static" else tf
                for transform in message.transforms:
                    transform_stamp = (
                        transform.header.stamp.to_nsec()
                        if transform.header.stamp
                        else bag_time.to_nsec()
                    )
                    target.append([
                        transform_stamp, transform.header.frame_id,
                        transform.child_frame_id,
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    ])

    write_csv(os.path.join(output_dir, "imu.csv"), [
        "stamp_ns", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y",
        "gyro_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    ], imu)
    write_csv(os.path.join(output_dir, "odometry_wheel.csv"), [
        "stamp_ns", "pose_x", "pose_y", "pose_z", "linear_x", "linear_y",
        "linear_z", "angular_x", "angular_y", "angular_z",
    ], odom)
    write_csv(os.path.join(output_dir, "cmd_vel.csv"), [
        "stamp_ns", "linear_x", "linear_y", "linear_z", "angular_x",
        "angular_y", "angular_z",
    ], cmd_vel)
    write_csv(os.path.join(output_dir, "tf.csv"), [
        "stamp_ns", "parent_frame", "child_frame", "translation_x",
        "translation_y", "translation_z", "rotation_x", "rotation_y",
        "rotation_z", "rotation_w",
    ], tf)
    write_csv(os.path.join(output_dir, "tf_static.csv"), [
        "stamp_ns", "parent_frame", "child_frame", "translation_x",
        "translation_y", "translation_z", "rotation_x", "rotation_y",
        "rotation_z", "rotation_w",
    ], tf_static)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise RuntimeError("LiDAR manifest not found: %s" % manifest_path)

    imu.sort(key=lambda row: row[0])
    odom.sort(key=lambda row: row[0])
    imu_times = [row[0] for row in imu]
    odom_times = [row[0] for row in odom]
    odom_array = np.asarray(odom, dtype=float)
    odom_unique_times, odom_unique_indices = np.unique(odom_array[:, 0], return_index=True)
    odom_array = odom_array[odom_unique_indices]
    odom_pchip = PchipInterpolator(odom_unique_times, odom_array[:, 4], extrapolate=False)
    yaw_rate_pchip = PchipInterpolator(odom_unique_times, odom_array[:, 9], extrapolate=False)
    aligned = []
    with open(manifest_path, newline="") as stream:
        for row in csv.DictReader(stream):
            lidar_stamp = int(row["stamp_ns"])
            imu_row = nearest(imu, imu_times, lidar_stamp)
            odom_row = nearest(odom, odom_times, lidar_stamp)
            odom_interp = interpolate_odom(odom_array, odom_pchip, yaw_rate_pchip, lidar_stamp)
            imu_skew_ms = "" if imu_row is None else abs(lidar_stamp - imu_row[0]) / 1e6
            aligned.append([
                row["frame_index"], lidar_stamp, row["pcd_file"],
                nearest_value(imu_row),
                "" if odom_row is None else odom_row[0],
                "" if odom_interp is None else "%.9f" % odom_interp[0],
                "" if odom_interp is None else "%.9f" % odom_interp[1],
                "" if odom_interp is None else "%.9f" % odom_interp[2],
                "" if odom_interp is None else "%.9f" % odom_interp[3],
                "" if odom_interp is None else "%.9f" % odom_interp[4],
                "" if imu_skew_ms == "" else "%.6f" % imu_skew_ms,
                int(odom_interp is not None),
            ])
    write_csv(os.path.join(output_dir, "aligned.csv"), [
        "frame_index", "lidar_stamp_ns", "pcd_file", "imu_stamp_ns",
        "odom_nearest_stamp_ns", "pose_x_interp", "pose_y_interp",
        "pose_z_interp", "linear_x_pchip", "angular_z_pchip",
        "imu_nearest_skew_ms", "odom_interp_valid",
    ], aligned)

    print("Extracted IMU: %d, wheel odometry: %d, TF: %d, TF static: %d, cmd_vel: %d" %
          (len(imu), len(odom), len(tf), len(tf_static), len(cmd_vel)))
    print("Aligned LiDAR frames: %d" % len(aligned))


def nearest(rows, times, target):
    if not rows:
        return None
    index = bisect.bisect_left(times, target)
    candidates = []
    if index < len(rows):
        candidates.append(rows[index])
    if index > 0:
        candidates.append(rows[index - 1])
    return min(candidates, key=lambda row: abs(row[0] - target))


def nearest_value(row):
    return "" if row is None else row[0]


def interpolate_odom(odom_array, speed_pchip, yaw_rate_pchip, target_ns,
                     max_gap_ns=200_000_000):
    """Interpolate odometry only when the bracketing samples are close enough."""
    times = odom_array[:, 0].astype(np.int64)
    index = np.searchsorted(times, target_ns, side="left")
    if index == 0 or index == len(times):
        return None
    gap = times[index] - times[index - 1]
    if gap > max_gap_ns:
        return None
    alpha = (target_ns - times[index - 1]) / float(gap)
    position = odom_array[index - 1, 1:4] + alpha * (
        odom_array[index, 1:4] - odom_array[index - 1, 1:4])
    return (
        position[0], position[1], position[2],
        float(speed_pchip(target_ns)), float(yaw_rate_pchip(target_ns)),
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_auxiliary_data.py INPUT_BAG OUTPUT_DIR")
    extract(sys.argv[1], sys.argv[2])
