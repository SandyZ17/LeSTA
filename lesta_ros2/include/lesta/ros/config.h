#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>

#include "lesta/core/core.h"

namespace lesta_ros::height_mapper {

static lesta::HeightMapper::Config loadConfig(rclcpp::Node &nh) {

  lesta::HeightMapper::Config cfg;
  nh.declare_parameter<std::string>("height_mapper.estimator_type", "StatMean");
  nh.get_parameter("height_mapper.estimator_type", cfg.estimator_type);
  nh.get_parameter_or("height_mapper.frame_id", cfg.frame_id, std::string("map"));
  nh.get_parameter_or("height_mapper.map_length_x", cfg.map_length_x, 10.0);
  nh.get_parameter_or("height_mapper.map_length_y", cfg.map_length_y, 10.0);
  nh.get_parameter_or("height_mapper.grid_resolution", cfg.grid_resolution, 0.1);
  nh.get_parameter_or("height_mapper.min_height_threshold", cfg.min_height, -0.2);
  nh.get_parameter_or("height_mapper.max_height_threshold", cfg.max_height, 1.5);
  return cfg;
}
} // namespace lesta_ros::height_mapper

namespace lesta_ros::global_mapper {

static lesta::GlobalMapper::Config loadConfig(rclcpp::Node &nh) {

  lesta::GlobalMapper::Config cfg;
  cfg.estimator_type = nh.declare_parameter<std::string>("global_mapper.estimator_type", "StatMean");
  cfg.frame_id = nh.declare_parameter<std::string>("global_mapper.frame_id", "map");
  cfg.map_length_x = nh.declare_parameter<double>("global_mapper.map_length_x", 400.0);
  cfg.map_length_y = nh.declare_parameter<double>("global_mapper.map_length_y", 400.0);
  cfg.grid_resolution = nh.declare_parameter<double>("global_mapper.grid_resolution", 0.1);
  cfg.min_height = nh.declare_parameter<double>("global_mapper.min_height_threshold", -0.2);
  cfg.max_height = nh.declare_parameter<double>("global_mapper.max_height_threshold", 1.5);
  cfg.map_save_dir = nh.declare_parameter<std::string>("global_mapper.map_save_dir", "/tmp");
  return cfg;
}
} // namespace lesta_ros::global_mapper

namespace lesta_ros::feature_extractor {

static lesta::FeatureExtractor::Config loadConfig(rclcpp::Node &nh) {

  lesta::FeatureExtractor::Config cfg;
  cfg.pca_radius = nh.declare_parameter<double>("feature_extractor.pca_radius", 0.2);

  return cfg;
}
} // namespace lesta_ros::feature_extractor

namespace lesta_ros::label_generator {

static lesta::LabelGenerator::Config loadConfig(rclcpp::Node &nh) {

  lesta::LabelGenerator::Config cfg;
  cfg.footprint_radius = nh.declare_parameter<double>("label_generator.footprint_radius", 0.5);
  cfg.max_traversable_step = nh.declare_parameter<double>("label_generator.max_traversable_step", 0.1);

  return cfg;
}
} // namespace lesta_ros::label_generator

namespace lesta_ros::traversability_estimator {

static lesta::TraversabilityEstimator::Config loadConfig(rclcpp::Node &nh) {

  lesta::TraversabilityEstimator::Config cfg;

  cfg.model_path = nh.declare_parameter<std::string>("traversability_network.model_path", "");
  cfg.input_dimension = nh.declare_parameter<int>("traversability_network.input_dimension", 4);
  cfg.feature_fields = nh.declare_parameter<std::vector<std::string>>(
      "traversability_network.feature_fields",
      {"step", "slope", "roughness", "curvature", "elevation_variance"});
  cfg.binary_threshold = nh.declare_parameter<float>("traversability_network.binary_threshold", 0.5f);
  return cfg;
}
} // namespace lesta_ros::traversability_estimator

namespace lesta_ros::temporal_traversability_estimator {

static lesta::TemporalTraversabilityEstimator::Config loadConfig(rclcpp::Node &nh) {
  lesta::TemporalTraversabilityEstimator::Config cfg;
  cfg.model_path = nh.get_parameter("temporal_network.model_path").as_string();
  cfg.input_dimension = nh.get_parameter("temporal_network.input_dimension").as_int();
  cfg.sequence_length = nh.get_parameter("temporal_network.sequence_length").as_int();
  cfg.binary_threshold = nh.get_parameter("temporal_network.binary_threshold").as_double();
  return cfg;
}
} // namespace lesta_ros::temporal_traversability_estimator

namespace lesta_ros::traversability_mapper {

static lesta::TraversabilityMapper::Config loadConfig(rclcpp::Node &nh) {

  lesta::TraversabilityMapper::Config cfg;
  cfg.model_path = nh.declare_parameter<std::string>("traversability_mapper.model_path", "");
  cfg.feature_fields = nh.declare_parameter<std::vector<std::string>>(
      "traversability_mapper.feature_fields", {"step", "slope", "roughness", "curvature"});
  cfg.input_dimension = nh.declare_parameter<int>("traversability_mapper.input_dimension", 4);
  cfg.binary_threshold = nh.declare_parameter<float>("traversability_mapper.binary_threshold", 0.5f);

  // mapping parameters
  cfg.use_log_odds_update = nh.declare_parameter<bool>("traversability_mapper.use_log_odds_update", true);
  cfg.learning_rate = nh.declare_parameter<float>("traversability_mapper.learning_rate", 0.7f);
  cfg.prob_min = nh.declare_parameter<float>("traversability_mapper.prob_min", 0.01f);
  cfg.prob_max = nh.declare_parameter<float>("traversability_mapper.prob_max", 0.99f);
  return cfg;
}
} // namespace lesta_ros::traversability_mapper
