/*
 * trav_mapping_node.cpp
 *
 *  Created on: Aug 17, 2023
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include "lesta/ros/trav_mapping_node.h"
#include "lesta/ros/config.h"

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <chrono>
#include <functional>

namespace lesta_ros {

TravMappingNode::TravMappingNode() : Node("trav_mapping_node"), nh_(*this), tf_(*this) {

  // Configs from trav_mapping_params.yaml, feature_extraction_params.yaml
  // ROS node
  TravMappingNode::loadConfig(*this);
  initializePubSubs();
  initializeServices();
  initializeTimers();

  frame_id_ = FrameID::loadFromConfig(*this);

  // Global Mapper
  mapper_ = std::make_unique<lesta::GlobalMapper>(global_mapper::loadConfig(*this));

  // Feature Extractor
  feature_extractor_ = std::make_unique<lesta::FeatureExtractor>(
      feature_extractor::loadConfig(*this));

  // Traversability Mapper
  trav_mapper_ = std::make_unique<lesta::TraversabilityMapper>(
      traversability_mapper::loadConfig(*this));

  std::cout << "\033[1;33m[lesta_ros::TravMappingNode]: "
               "Traversability mapping node initialized. Waiting for scan "
               "inputs...\033[0m\n";
}

void TravMappingNode::loadConfig(rclcpp::Node &nh) {
  cfg_.lidarscan_topic = nh.declare_parameter<std::string>("node.lidarscan_topic", "/velodyne_points");
  cfg_.map_pub_rate = nh.declare_parameter<double>("node.map_publish_rate", 10.0);
  cfg_.scan_filter_range = nh.declare_parameter<double>("node.scan_filter_range", 10.0);
  cfg_.remove_backpoints = nh.declare_parameter<bool>("node.remove_backpoints", true);
  cfg_.debug_mode = nh.declare_parameter<bool>("node.debug_mode", false);
}

void TravMappingNode::initializePubSubs() {
  sub_lidarscan_ =
      nh_.create_subscription<sensor_msgs::msg::PointCloud2>(cfg_.lidarscan_topic, rclcpp::SensorDataQoS(),
        std::bind(&TravMappingNode::lidarScanCallback, this, std::placeholders::_1));
  pub_downsampled_scan_ =
      nh_.create_publisher<sensor_msgs::msg::PointCloud2>("/lesta/mapping/scan_downsampled", 1);
  pub_filtered_scan_ =
      nh_.create_publisher<sensor_msgs::msg::PointCloud2>("/lesta/mapping/scan_filtered", 1);
  pub_rasterized_scan_ =
      nh_.create_publisher<sensor_msgs::msg::PointCloud2>("/lesta/mapping/scan_rasterized", 1);
  pub_travmap_ =
      nh_.create_publisher<sensor_msgs::msg::PointCloud2>("/lesta/mapping/traversability", 1);
  pub_map_region_ =
      nh_.create_publisher<visualization_msgs::msg::Marker>("/lesta/mapping/mapping_region", 1);

  if (cfg_.debug_mode) {
    // TODO: Add debug publishers
  }
}

void TravMappingNode::initializeServices() {
  //
}

void TravMappingNode::initializeTimers() {
  map_publish_timer_ = nh_.create_wall_timer(
      std::chrono::duration<double>(1.0 / cfg_.map_pub_rate),
      std::bind(&TravMappingNode::publishMaps, this));
}

void TravMappingNode::lidarScanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

  if (!lidarscan_received_) {
    lidarscan_received_ = true;
    frame_id_.sensor = msg->header.frame_id;
    std::cout << "\033[1;32m[lesta_ros::TravMappingNode]: "
              << "Lidar scan received! Use LiDAR scans for traversability mapping... "
              << "\033[0m\n";
  }

  // 1. Get transform matrix using tf tree
  geometry_msgs::msg::TransformStamped sensor2base, base2map;
  if (!tf_.lookupTransform(frame_id_.robot, frame_id_.sensor, sensor2base) ||
      !tf_.lookupTransform(frame_id_.map, frame_id_.robot, base2map))
    return;

  // 2. Convert ROS msg to PCL data
  auto scan_raw = std::make_shared<pcl::PointCloud<Laser>>();
  pcl::moveFromROSMsg(*msg, *scan_raw);

  // 3. Preprocess scan data: ready for terrain mapping
  auto scan_preprocessed = preprocessScan(scan_raw, sensor2base, base2map);
  if (!scan_preprocessed)
    return;

  // 4. Terrain mapping
  auto transform_sensor2map = TransformOps::multiplyTransforms(sensor2base, base2map);
  Eigen::Vector3f sensor_origin(transform_sensor2map.transform.translation.x,
                                transform_sensor2map.transform.translation.y,
                                transform_sensor2map.transform.translation.z);
  auto measured_indices = terrainMapping(scan_preprocessed, sensor_origin);

  // 5. Feature extraction
  feature_extractor_->extractFeatures(mapper_->getHeightMap(), measured_indices);

  // 6. Traversability mapping
  trav_mapper_->traversabilityMapping(mapper_->getHeightMap(), measured_indices);

  // 7. Start publishing maps
}
pcl::PointCloud<Laser>::Ptr
TravMappingNode::preprocessScan(const pcl::PointCloud<Laser>::Ptr &scan_raw,
                                const geometry_msgs::msg::TransformStamped &sensor2base,
                                const geometry_msgs::msg::TransformStamped &base2map) {
  // 1. Transform pointcloud to base frame
  auto scan_base = PointCloudOps::applyTransform<Laser>(scan_raw, sensor2base);

  // For visualization: downsample pointcloud
  auto scan_downsampled = PointCloudOps::downsampleVoxel<Laser>(scan_base, 0.4);
  publishDownsampledScan(scan_downsampled);

  // 2. Fast height filtering
  auto scan_preprocessed = std::make_shared<pcl::PointCloud<Laser>>();
  mapper_->fastHeightFilter<Laser>(scan_base, scan_preprocessed);

  // 3. Pass through filter
  scan_preprocessed = PointCloudOps::passThrough<Laser>(scan_preprocessed,
                                                        "x",
                                                        -cfg_.scan_filter_range,
                                                        cfg_.scan_filter_range);
  scan_preprocessed = PointCloudOps::passThrough<Laser>(scan_preprocessed,
                                                        "y",
                                                        -cfg_.scan_filter_range,
                                                        cfg_.scan_filter_range);

  // (Optional) Remove remoter points
  if (cfg_.remove_backpoints)
    scan_preprocessed =
        PointCloudOps::filterAngle2D<Laser>(scan_preprocessed, -135.0, 135.0);

  // 4. Publish filtered scan
  publishFilteredScan(scan_preprocessed);

  // 5. Transform pointcloud to map frame
  scan_preprocessed = PointCloudOps::applyTransform<Laser>(scan_preprocessed, base2map);

  if (scan_preprocessed->empty())
    return nullptr;

  return scan_preprocessed;
}

std::vector<grid_map::Index>
TravMappingNode::terrainMapping(const pcl::PointCloud<Laser>::Ptr &cloud_input,
                                const Eigen::Vector3f &sensor_origin) {
  // 1. Rasterize scan and height mapping
  auto cloud_rasterized = mapper_->heightMapping<Laser>(cloud_input);

  // 2. Publish rasterized scan
  publishRasterizedScan(cloud_rasterized);

  // 3. Raycasting to remove dynamic obstacles
  mapper_->raycasting<Laser>(sensor_origin, cloud_rasterized);

  // 4. Get measured indices
  return getMeasuredIndices(mapper_->getHeightMap(), cloud_rasterized);
}

std::vector<grid_map::Index>
TravMappingNode::getMeasuredIndices(const HeightMap &map,
                                    const pcl::PointCloud<Laser>::Ptr &input_scan) {
  std::vector<grid_map::Index> indices;
  indices.reserve(input_scan->size());
  for (const auto &point : input_scan->points) {
    grid_map::Position position(point.x, point.y);
    grid_map::Index index;
    if (map.getIndex(position, index))
      indices.push_back(index);
  }
  return indices;
}

void TravMappingNode::publishDownsampledScan(const pcl::PointCloud<Laser>::Ptr &scan) {
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*scan, cloud_msg);
  pub_downsampled_scan_->publish(cloud_msg);
}

void TravMappingNode::publishRasterizedScan(const pcl::PointCloud<Laser>::Ptr &scan) {
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*scan, cloud_msg);
  pub_rasterized_scan_->publish(cloud_msg);
}

void TravMappingNode::publishFilteredScan(const pcl::PointCloud<Laser>::Ptr &scan) {
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*scan, cloud_msg);
  pub_filtered_scan_->publish(cloud_msg);
}

void TravMappingNode::publishMaps() {

  publishTravMap(mapper_->getHeightMap(), mapper_->getMeasuredGridIndices());
  publishMapRegion(mapper_->getHeightMap());
}

void TravMappingNode::publishTravMap(const HeightMap &heightmap,
                                     const std::unordered_set<grid_map::Index> &indices) {

  std::vector<std::string> layers = {height_mapping::layers::Height::ELEVATION,
                                     height_mapping::layers::Height::ELEVATION_VARIANCE,
                                     lesta::layers::Feature::STEP,
                                     lesta::layers::Feature::SLOPE,
                                     lesta::layers::Feature::ROUGHNESS,
                                     lesta::layers::Feature::CURVATURE,
                                     lesta::layers::Traversability::PROBABILITY,
                                     lesta::layers::Traversability::BINARY,
                                     lesta::layers::Traversability::LOG_ODDS,
                                     lesta::layers::Traversability::LOG_ODDS_PROBABILITY,
                                     lesta::layers::Traversability::LOG_ODDS_BINARY};
  sensor_msgs::msg::PointCloud2 msg;
  toPointCloud2(heightmap, layers, indices, msg);
  pub_travmap_->publish(msg);
}

void TravMappingNode::publishMapRegion(const HeightMap &map) {
  visualization_msgs::msg::Marker marker;
  toMapRegion(map, marker);
  pub_map_region_->publish(marker);
}

void TravMappingNode::toPointCloud2(
    const HeightMap &map,
    const std::vector<std::string> &layers,
    const std::unordered_set<grid_map::Index> &measuredIndices,
    sensor_msgs::msg::PointCloud2 &cloud) {
  // Setup cloud header
  cloud.header.frame_id = map.getFrameId();
  const auto timestamp = map.getTimestamp();
  cloud.header.stamp.sec = static_cast<int32_t>(timestamp / 1000000000ULL);
  cloud.header.stamp.nanosec = static_cast<uint32_t>(timestamp % 1000000000ULL);
  cloud.is_dense = false;

  // Setup field names and cloud structure
  std::vector<std::string> fieldNames;
  fieldNames.reserve(layers.size());

  // Setup field names
  fieldNames.insert(fieldNames.end(), {"x", "y", "z"});
  for (const auto &layer : layers) {
    if (layer == "color") {
      fieldNames.push_back("rgb");
    } else {
      fieldNames.push_back(layer);
    }
  }

  // Setup point field structure
  cloud.fields.clear();
  cloud.fields.reserve(fieldNames.size());
  int offset = 0;

  for (const auto &name : fieldNames) {
    sensor_msgs::msg::PointField field;
    field.name = name;
    field.count = 1;
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.offset = offset;
    cloud.fields.push_back(field);
    offset += sizeof(float);
  }

  // Initialize cloud size
  const size_t num_points = measuredIndices.size();
  cloud.height = 1;
  cloud.width = num_points;
  cloud.point_step = offset;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(cloud.height * cloud.row_step);

  // Setup point field iterators
  std::unordered_map<std::string, sensor_msgs::PointCloud2Iterator<float>> iterators;
  for (const auto &name : fieldNames) {
    iterators.emplace(name, sensor_msgs::PointCloud2Iterator<float>(cloud, name));
  }

  // Fill point cloud data
  size_t validPoints = 0;
  for (const auto &index : measuredIndices) {
    grid_map::Position3 position;
    if (!map.getPosition3(height_mapping::layers::Height::ELEVATION, index, position)) {
      continue;
    }

    // Update each field
    for (auto &[fieldName, iterator] : iterators) {
      if (fieldName == "x")
        *iterator = static_cast<float>(position.x());
      else if (fieldName == "y")
        *iterator = static_cast<float>(position.y());
      else if (fieldName == "z")
        *iterator = static_cast<float>(position.z());
      else if (fieldName == "rgb")
        *iterator = static_cast<float>(map.at("color", index));
      else
        *iterator = static_cast<float>(map.at(fieldName, index));
      ++iterator;
    }
    ++validPoints;
  }

  // Adjust final cloud size
  cloud.width = validPoints;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(cloud.height * cloud.row_step);
}

void TravMappingNode::toMapRegion(const HeightMap &map,
                                  visualization_msgs::msg::Marker &marker) {

  marker.ns = "trav_map";
  marker.lifetime = rclcpp::Duration(0, 0);
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;

  marker.scale.x = 0.1;
  marker.color.a = 1.0;
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;

  marker.header.frame_id = map.getFrameId();
  marker.header.stamp = nh_.get_clock()->now();

  float length_x_half = (map.getLength().x() - 0.5 * map.getResolution()) / 2.0;
  float length_y_half = (map.getLength().y() - 0.5 * map.getResolution()) / 2.0;

  marker.points.resize(5);
  marker.points[0].x = map.getPosition().x() + length_x_half;
  marker.points[0].y = map.getPosition().y() + length_x_half;
  marker.points[0].z = 0;

  marker.points[1].x = map.getPosition().x() + length_x_half;
  marker.points[1].y = map.getPosition().y() - length_x_half;
  marker.points[1].z = 0;

  marker.points[2].x = map.getPosition().x() - length_x_half;
  marker.points[2].y = map.getPosition().y() - length_x_half;
  marker.points[2].z = 0;

  marker.points[3].x = map.getPosition().x() - length_x_half;
  marker.points[3].y = map.getPosition().y() + length_x_half;
  marker.points[3].z = 0;

  marker.points[4] = marker.points[0];
}
} // namespace lesta_ros

int main(int argc, char **argv) {

  rclcpp::init(argc, argv);
  auto node = std::make_shared<lesta_ros::TravMappingNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
