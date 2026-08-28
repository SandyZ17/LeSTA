/*
 * trav_mapping_node.h
 *
 *  Created on: Aug 17, 2023
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "common/ros/common.h"
#include "lesta/core/core.h"

namespace lesta_ros {

class TravMappingNode : public rclcpp::Node {
public:
  struct Config {
    std::string lidarscan_topic;
    double map_pub_rate;
    double scan_filter_range;
    bool remove_backpoints;
    bool debug_mode;
  } cfg_;

  TravMappingNode();
  ~TravMappingNode() = default;
  void loadConfig(rclcpp::Node &nh);

private:
  // init functions
  void initializeTimers();
  void initializePubSubs();
  void initializeServices();

  // callback functions: main logic
  void lidarScanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void publishMaps();

  pcl::PointCloud<Laser>::Ptr
  preprocessScan(const pcl::PointCloud<Laser>::Ptr &scan_raw,
                 const geometry_msgs::msg::TransformStamped &sensor2base,
                 const geometry_msgs::msg::TransformStamped &base2map);
  std::vector<grid_map::Index>
  terrainMapping(const pcl::PointCloud<Laser>::Ptr &cloud_input,
                 const Eigen::Vector3f &sensor_origin);

  // Helper functions
  std::vector<grid_map::Index>
  getMeasuredIndices(const HeightMap &map, const pcl::PointCloud<Laser>::Ptr &input_scan);
  void publishDownsampledScan(const pcl::PointCloud<Laser>::Ptr &scan);
  void publishFilteredScan(const pcl::PointCloud<Laser>::Ptr &scan);
  void publishRasterizedScan(const pcl::PointCloud<Laser>::Ptr &scan);
  void publishTravMap(const HeightMap &heightmap,
                      const std::unordered_set<grid_map::Index> &measuredIndices);
  void publishMapRegion(const HeightMap &map);
  void toPointCloud2(const HeightMap &map,
                     const std::vector<std::string> &layers,
                     const std::unordered_set<grid_map::Index> &measuredIndices,
                     sensor_msgs::msg::PointCloud2 &cloud);
  void toMapRegion(const HeightMap &map, visualization_msgs::msg::Marker &marker);

  // Timer callback

  rclcpp::Node &nh_;

  // Subscribers & Publishers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidarscan_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_downsampled_scan_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_filtered_scan_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_rasterized_scan_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_travmap_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_map_region_;

  // Timers
  rclcpp::TimerBase::SharedPtr map_publish_timer_;

  // Core objects
  std::unique_ptr<lesta::GlobalMapper> mapper_;
  std::unique_ptr<lesta::FeatureExtractor> feature_extractor_;
  std::unique_ptr<lesta::TraversabilityMapper> trav_mapper_;
  TransformOps tf_;
  FrameID frame_id_;

  // State variables
  bool lidarscan_received_{false};
};
} // namespace lesta_ros
