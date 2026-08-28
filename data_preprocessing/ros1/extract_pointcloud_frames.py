#!/usr/bin/env python3
"""Save decoded ROS 1 PointCloud2 messages as timestamped binary PCD files."""

import csv
import os
import struct
import threading

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


class FrameExtractor:
    def __init__(self):
        self.output_dir = rospy.get_param("~output_dir")
        self.topic = rospy.get_param("~topic", "/velodyne_points")
        self.stride = max(1, int(rospy.get_param("~stride", 1)))
        self.count = 0
        self.saved = 0
        self.lock = threading.Lock()
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest = open(os.path.join(self.output_dir, "manifest.csv"), "w", newline="")
        self.writer = csv.writer(self.manifest)
        self.writer.writerow(["frame_index", "stamp_ns", "frame_id", "point_count", "pcd_file"])
        rospy.Subscriber(self.topic, PointCloud2, self.callback, queue_size=2)

    @staticmethod
    def write_pcd(path, points):
        with open(path, "wb") as output:
            output.write((
                "# .PCD v0.7 - Point Cloud Data file format\n"
                "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
                "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\n"
                "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n"
                % (len(points), len(points))
            ).encode("ascii"))
            output.write(b"".join(struct.pack("<ffff", *point) for point in points))

    def callback(self, message):
        with self.lock:
            index = self.count
            self.count += 1
            if index % self.stride != 0:
                return

            points = list(point_cloud2.read_points(
                message, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            stamp_ns = message.header.stamp.to_nsec()
            filename = "%012d_%012d.pcd" % (self.saved, stamp_ns)
            self.write_pcd(os.path.join(self.output_dir, filename), points)
            self.writer.writerow([self.saved, stamp_ns, message.header.frame_id, len(points), filename])
            self.manifest.flush()
            self.saved += 1
            rospy.loginfo_throttle(10.0, "Saved %d point-cloud frames", self.saved)


if __name__ == "__main__":
    rospy.init_node("extract_pointcloud_frames")
    FrameExtractor()
    rospy.spin()
