cmake_minimum_required(VERSION 3.8)
project(height_mapping_core)
add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(PCL REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(grid_map_core REQUIRED)

###########
## Build ##
###########
set(CMAKE_BUILD_TYPE Release)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${grid_map_core_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_library(${PROJECT_NAME}
  src/HeightMap.cpp
  src/KalmanEstimator.cpp
  src/MovingAverageEstimator.cpp
  src/StatMeanEstimator.cpp
  src/FastHeightFilter.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  grid_map_core
)

target_link_libraries(${PROJECT_NAME}
  grid_map_core::grid_map_core
  ${PCL_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_dependencies(grid_map_core)
ament_export_libraries(${PROJECT_NAME})
ament_package()
