/*
 * trav_prediction_node.h
 *
 *  Created on: Aug 17, 2023
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>

#include "common/ros/common.h"
#include "lesta/core/core.h"

namespace lesta_ros {

class TravPredictionNode : public rclcpp::Node {
public:
  struct Config {
    std::string lidarscan_topic;
    double pose_update_rate;
    double map_pub_rate;
    bool remove_backpoints;
    bool debug_mode;
  } cfg_;

  TravPredictionNode();
  ~TravPredictionNode() = default;
  void loadConfig(rclcpp::Node &nh);

private:
  // init functions
  void initializeTimers();
  void initializePubSubs();
  void initializeServices();

  // main logic
  void lidarScanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  pcl::PointCloud<Laser>::Ptr
  preprocessScan(const pcl::PointCloud<Laser>::Ptr &scan_raw,
                 const geometry_msgs::msg::TransformStamped &sensor2base,
                 const geometry_msgs::msg::TransformStamped &base2map);
  std::vector<grid_map::Index>
  terrainMapping(const pcl::PointCloud<Laser>::Ptr &inputcloud,
                 const Eigen::Vector3f &sensorOrigin3D);

  // helper functions
  std::vector<grid_map::Index>
  getMeasuredIndices(const HeightMap &map, const pcl::PointCloud<Laser>::Ptr &scan);
  void publishDownsampledScan(const pcl::PointCloud<Laser>::Ptr &scan);
  void publishFilteredScan(const pcl::PointCloud<Laser>::Ptr &scan);
  void publishRasterizedScan(const pcl::PointCloud<Laser>::Ptr &scan);
  void publishHeightmap(const HeightMap &map);
  void publishTravMap(const HeightMap &map);

  // Timer functions
  void updateMapOrigin();
  void publishMaps();

  rclcpp::Node &nh_;

  // Subscribers & Publishers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidarscan_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_downsampled_scan_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_filtered_scan_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_rasterized_scan_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pub_heightmap_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pub_travmap_;

  // Timers
  rclcpp::TimerBase::SharedPtr pose_update_timer_;
  rclcpp::TimerBase::SharedPtr map_publish_timer_;

  // Core objects
  std::unique_ptr<lesta::HeightMapper> mapper_;
  std::unique_ptr<lesta::FeatureExtractor> feature_extractor_;
  std::unique_ptr<lesta::TraversabilityEstimator> trav_estimator_;
  TransformOps tf_;
  FrameID frame_id_;

  // State variables
  bool lidarscan_received_{false};
};
} // namespace lesta_ros
