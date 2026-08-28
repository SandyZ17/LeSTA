#!/usr/bin/env bash
set -eo pipefail

bag_path="${1:?usage: extract_frames.sh INPUT_BAG OUTPUT_DIR [STRIDE]}"
output_dir="${2:?usage: extract_frames.sh INPUT_BAG OUTPUT_DIR [STRIDE]}"
stride="${3:-1}"

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
source /opt/ros/noetic/setup.bash

roscore >/tmp/lesta_roscore.log 2>&1 &
roscore_pid=$!
cleanup() {
  kill "$roscore_pid" 2>/dev/null || true
}
trap cleanup EXIT
sleep 2

rosparam set use_sim_time true
roslaunch /workspace/data_preprocessing/ros1/vlp16packet_to_pointcloud.launch \
  packet_topic:=/velodyne_tilted/velodyne_packets >/tmp/lesta_velodyne.log 2>&1 &
decoder_pid=$!
sleep 5

python3 /workspace/data_preprocessing/ros1/extract_pointcloud_frames.py \
  _output_dir:="$output_dir" _topic:=/velodyne_points _stride:="$stride" >/tmp/lesta_extractor.log 2>&1 &
extractor_pid=$!
sleep 2

rosbag play --clock --topics /velodyne_tilted/velodyne_packets --bags "$bag_path" \
  >/tmp/lesta_bag_play.log 2>&1
sleep 2
kill "$extractor_pid" "$decoder_pid" 2>/dev/null || true
