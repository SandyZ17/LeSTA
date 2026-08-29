/*
 * trav_prediction_node.cpp
 *
 *  Created on: Aug 17, 2023
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include "lesta/ros/trav_prediction_node.h"
#include "lesta/ros/config.h"

#include <grid_map_ros/GridMapRosConverter.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>

namespace lesta_ros {

TravPredictionNode::TravPredictionNode()
    : Node("trav_prediction_node"), nh_(*this), tf_(*this) {

  // Configs from trav_prediction_node.yaml, feature_extraction_params.yaml
  // ROS node
  TravPredictionNode::loadConfig(*this);
  initializePubSubs();
  initializeServices();
  initializeTimers();

  frame_id_ = FrameID::loadFromConfig(*this);

  // Height Mapper
  mapper_ = std::make_unique<lesta::HeightMapper>(height_mapper::loadConfig(*this));

  // Feature Extractor
  feature_extractor_ = std::make_unique<lesta::FeatureExtractor>(
      feature_extractor::loadConfig(*this));

  if (cfg_.temporal_enabled) {
    temporal_estimator_ = std::make_unique<lesta::TemporalTraversabilityEstimator>(
        temporal_traversability_estimator::loadConfig(*this));
  } else {
    trav_estimator_ = std::make_unique<lesta::TraversabilityEstimator>(
        traversability_estimator::loadConfig(*this));
  }

  std::cout << "\033[1;33m[lesta_ros::TravPredictionNode]: "
               "Traversability prediction node initialized. Waiting for scan "
               "inputs...\033[0m\n";
}

void TravPredictionNode::loadConfig(rclcpp::Node &nh) {

  cfg_.lidarscan_topic = nh.declare_parameter<std::string>("node.lidarscan_topic", "/velodyne_points");
  cfg_.pose_update_rate = nh.declare_parameter<double>("node.pose_update_rate", 10.0);
  cfg_.map_pub_rate = nh.declare_parameter<double>("node.map_publish_rate", 10.0);
  cfg_.remove_backpoints = nh.declare_parameter<bool>("node.remove_backpoints", true);
  cfg_.debug_mode = nh.declare_parameter<bool>("node.debug_mode", false);
  cfg_.temporal_enabled = nh.declare_parameter<bool>("temporal_enabled", false);
  nh.declare_parameter<std::string>("temporal_network.model_path", "");
  nh.declare_parameter<int>("temporal_network.input_dimension", 21);
  nh.declare_parameter<int>("temporal_network.sequence_length", 10);
  nh.declare_parameter<double>("temporal_network.binary_threshold", 0.5);
  cfg_.odometry_topic = nh.declare_parameter<std::string>("node.odometry_topic", "/odom");
  cfg_.imu_topic = nh.declare_parameter<std::string>("node.imu_topic", "/imu/data");
}

void TravPredictionNode::initializePubSubs() {

  sub_lidarscan_ = nh_.create_subscription<sensor_msgs::msg::PointCloud2>(
      cfg_.lidarscan_topic, rclcpp::SensorDataQoS(),
      std::bind(&TravPredictionNode::lidarScanCallback, this, std::placeholders::_1));
  if (cfg_.temporal_enabled) {
    sub_imu_ = nh_.create_subscription<sensor_msgs::msg::Imu>(
        cfg_.imu_topic, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { latest_imu_ = std::move(msg); });
    sub_odometry_ = nh_.create_subscription<nav_msgs::msg::Odometry>(
        cfg_.odometry_topic, rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { latest_odometry_ = std::move(msg); });
  }
  pub_downsampled_scan_ = nh_.create_publisher<sensor_msgs::msg::PointCloud2>("/lesta/prediction/scan_downsampled", 1);
  pub_filtered_scan_ = nh_.create_publisher<sensor_msgs::msg::PointCloud2>("/lesta/prediction/scan_filtered", 1);
  pub_rasterized_scan_ = nh_.create_publisher<sensor_msgs::msg::PointCloud2>("/lesta/prediction/scan_rasterized", 1);
  pub_heightmap_ = nh_.create_publisher<grid_map_msgs::msg::GridMap>("/lesta/prediction/heightmap", 1);
  pub_travmap_ = nh_.create_publisher<grid_map_msgs::msg::GridMap>("/lesta/prediction/traversability", 1);
}

void TravPredictionNode::initializeServices() {
  // TODO: Implement this
}

void TravPredictionNode::initializeTimers() {

  pose_update_timer_ = nh_.create_wall_timer(
      std::chrono::duration<double>(1.0 / cfg_.pose_update_rate),
      std::bind(&TravPredictionNode::updateMapOrigin, this));
  map_publish_timer_ = nh_.create_wall_timer(
      std::chrono::duration<double>(1.0 / cfg_.map_pub_rate),
      std::bind(&TravPredictionNode::publishMaps, this));
}

void TravPredictionNode::lidarScanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

  if (!lidarscan_received_) {
    lidarscan_received_ = true;
    frame_id_.sensor = msg->header.frame_id;
    std::cout << "\033[1;32m[lesta_ros::TravPredictionNode]: Pointcloud Received! "
              << "Use LiDAR scans for traversability prediction... \033[0m\n";
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

  // 6. Traversability estimation
  if (!cfg_.temporal_enabled) {
    trav_estimator_->estimateTraversability(mapper_->getHeightMap(), measured_indices);
  } else if (latest_imu_ && latest_odometry_) {
    const rclcpp::Time stamp(msg->header.stamp);
    const double dt = previous_lidar_stamp_.nanoseconds() == 0
                          ? 0.0 : (stamp - previous_lidar_stamp_).seconds();
    const double speed = latest_odometry_->twist.twist.linear.x;
    const double yaw_rate = latest_odometry_->twist.twist.angular.z;
    const double longitudinal_accel = dt > 1e-6 ?
        (speed - previous_linear_x_) / dt : 0.0;
    const auto &accel = latest_imu_->linear_acceleration;
    const auto &gyro = latest_imu_->angular_velocity;
    Eigen::VectorXf motion(13);
    motion << speed, yaw_rate, accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z,
        static_cast<float>(dt), static_cast<float>(longitudinal_accel),
        static_cast<float>(speed * yaw_rate),
        static_cast<float>(std::sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z)),
        static_cast<float>(std::sqrt(gyro.x * gyro.x + gyro.y * gyro.y + gyro.z * gyro.z));
    temporal_estimator_->estimate(mapper_->getHeightMap(), measured_indices, motion);
    previous_lidar_stamp_ = stamp;
    previous_linear_x_ = speed;
  }

  // 7. Publish maps
}

pcl::PointCloud<Laser>::Ptr
TravPredictionNode::preprocessScan(const pcl::PointCloud<Laser>::Ptr &scan_raw,
                                   const geometry_msgs::msg::TransformStamped &sensor2base,
                                   const geometry_msgs::msg::TransformStamped &base2map) {

  // 1. Transform pointcloud to base frame
  auto scan_base = PointCloudOps::applyTransform<Laser>(scan_raw, sensor2base);

  // 2. Publish downsampled scan
  auto scan_downsampled = PointCloudOps::downsampleVoxel<Laser>(scan_base, 0.4);
  publishDownsampledScan(scan_downsampled);

  // 2. Fast height filtering
  auto scan_preprocessed = std::make_shared<pcl::PointCloud<Laser>>();
  mapper_->fastHeightFilter<Laser>(scan_base, scan_preprocessed);

  // 3. Pass through filter
  auto range = mapper_->getHeightMap().getLength() / 2.0;
  scan_preprocessed =
      PointCloudOps::passThrough<Laser>(scan_preprocessed, "x", -range.x(), range.x());
  scan_preprocessed =
      PointCloudOps::passThrough<Laser>(scan_preprocessed, "y", -range.y(), range.y());

  // (Optional) Remove remoter points
  if (cfg_.remove_backpoints)
    scan_preprocessed =
        PointCloudOps::filterAngle2D<Laser>(scan_preprocessed, -135.0, 135.0);

  // 4. Transform pointcloud to map frame
  scan_preprocessed = PointCloudOps::applyTransform<Laser>(scan_preprocessed, base2map);

  if (scan_preprocessed->empty())
    return nullptr;

  // 5. Publish filtered scan
  publishFilteredScan(scan_preprocessed);

  return scan_preprocessed;
}

std::vector<grid_map::Index>
TravPredictionNode::terrainMapping(const pcl::PointCloud<Laser>::Ptr &cloud_input,
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
TravPredictionNode::getMeasuredIndices(const HeightMap &map,
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

void TravPredictionNode::publishDownsampledScan(const pcl::PointCloud<Laser>::Ptr &scan) {
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*scan, cloud_msg);
  pub_downsampled_scan_->publish(cloud_msg);
}

void TravPredictionNode::publishFilteredScan(const pcl::PointCloud<Laser>::Ptr &scan) {
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*scan, cloud_msg);
  pub_filtered_scan_->publish(cloud_msg);
}

void TravPredictionNode::publishRasterizedScan(const pcl::PointCloud<Laser>::Ptr &scan) {
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*scan, cloud_msg);
  pub_rasterized_scan_->publish(cloud_msg);
}

void TravPredictionNode::publishHeightmap(const HeightMap &map) {
  std::vector<std::string> layers = map.getBasicLayers();

  auto msg = grid_map::GridMapRosConverter::toMessage(map, layers);
  pub_heightmap_->publish(*msg);
}

void TravPredictionNode::publishTravMap(const HeightMap &map) {
  std::vector<std::string> layers = map.getBasicLayers();
  const char *optional_layers[] = {
      height_mapping::layers::Height::ELEVATION_VARIANCE,
      lesta::layers::Feature::STEP, lesta::layers::Feature::SLOPE,
      lesta::layers::Feature::ROUGHNESS, lesta::layers::Feature::CURVATURE,
      lesta::layers::Feature::NORMAL_X, lesta::layers::Feature::NORMAL_Y,
      lesta::layers::Feature::NORMAL_Z, lesta::layers::Feature::HEIGHT_RANGE,
      lesta::layers::Feature::ELEVATION_IQR, lesta::layers::Feature::ROUGHNESS_P95,
      lesta::layers::Feature::OCCUPIED_CELL_RATIO,
      lesta::layers::Traversability::BINARY,
      lesta::layers::Traversability::PROBABILITY};
  for (const auto *layer : optional_layers) {
    if (map.exists(layer))
      layers.emplace_back(layer);
  }

  auto msg = grid_map::GridMapRosConverter::toMessage(map, layers);
  pub_travmap_->publish(*msg);
}

void TravPredictionNode::updateMapOrigin() {

  // 1. Get transform matrix using tf tree
  geometry_msgs::msg::TransformStamped base2map;
  if (!tf_.lookupTransform(frame_id_.map, frame_id_.robot, base2map))
    return;

  // 2. Update map origin
  auto robot_position = grid_map::Position(base2map.transform.translation.x,
                                           base2map.transform.translation.y);
  mapper_->moveMapOrigin(robot_position);
}

void TravPredictionNode::publishMaps() {

  publishHeightmap(mapper_->getHeightMap());
  publishTravMap(mapper_->getHeightMap());
}
} // namespace lesta_ros

int main(int argc, char **argv) {

  rclcpp::init(argc, argv);
  auto node = std::make_shared<lesta_ros::TravPredictionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
