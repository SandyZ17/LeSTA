#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>

struct FrameID {
  std::string robot;
  std::string map;
  std::string sensor;

  static FrameID loadFromConfig(rclcpp::Node &node) {
    FrameID frame;
    frame.robot = node.declare_parameter<std::string>("frame_id.robot_frame", "base_link");
    frame.map = node.declare_parameter<std::string>("frame_id.map_frame", "map");
    frame.sensor = node.declare_parameter<std::string>("frame_id.sensor_frame", "");
    return frame;
  }
};
