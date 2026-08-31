cmake_minimum_required(VERSION 3.8)
project(lesta_ros2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(height_mapping_core REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Torch REQUIRED PATHS /opt/torch/share/cmake/Torch)
find_library(SYSTEM_CURL_LIBRARY curl PATHS /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH REQUIRED)

  rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SaveTrainingData.srv"
  DEPENDENCIES std_msgs
)

include_directories(include ${PCL_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

set(LESTA_CORE_SOURCES src/core/HeightMapper.cpp src/core/FeatureExtractor.cpp)
add_executable(label_generation_node src/ros/label_generation_node.cpp ${LESTA_CORE_SOURCES} src/core/GlobalMapper.cpp src/core/LabelGenerator.cpp)
add_executable(trav_prediction_node src/ros/trav_prediction_node.cpp ${LESTA_CORE_SOURCES} src/core/TraversabilityNetwork.cpp src/core/TraversabilityEstimator.cpp src/core/TemporalTraversabilityEstimator.cpp)
add_executable(trav_mapping_node src/ros/trav_mapping_node.cpp ${LESTA_CORE_SOURCES} src/core/GlobalMapper.cpp src/core/TraversabilityNetwork.cpp src/core/TraversabilityEstimator.cpp src/core/TraversabilityMapper.cpp)

foreach(target label_generation_node trav_prediction_node trav_mapping_node)
  ament_target_dependencies(${target}
    rclcpp sensor_msgs nav_msgs geometry_msgs visualization_msgs tf2_ros tf2_eigen
    tf2_geometry_msgs pcl_conversions grid_map_core grid_map_msgs grid_map_ros
    height_mapping_core)
  target_link_libraries(${target}
    ${PCL_LIBRARIES} ${TORCH_LIBRARIES} ${SYSTEM_CURL_LIBRARY}
    grid_map_core::grid_map_core)
  add_dependencies(${target} ${PROJECT_NAME}__rosidl_generator_cpp)
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp)
  install(TARGETS ${target} DESTINATION lib/${PROJECT_NAME})
endforeach()

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(label_generation_node "${cpp_typesupport_target}")

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING
  PATTERN "*.launch.py"
  PATTERN "*.rviz")

ament_export_dependencies(rosidl_default_runtime)
ament_package()
