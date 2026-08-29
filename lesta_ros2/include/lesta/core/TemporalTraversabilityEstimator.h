#pragma once

#include "lesta/core/TraversabilityNetwork.h"
#include "lesta/types/layer_definitions.h"
#include <height_mapping_core/height_mapping_core.h>
#include <deque>
#include <map>

namespace lesta {

class TemporalTraversabilityEstimator {
public:
  struct Config {
    std::string model_path;
    int input_dimension{21};
    int sequence_length{10};
    float binary_threshold{0.5f};
  };

  explicit TemporalTraversabilityEstimator(const Config &config);

  void reset();
  void estimate(HeightMap &map, const std::vector<grid_map::Index> &indices,
                const Eigen::VectorXf &motion_features);

private:
  using CellKey = std::pair<int, int>;
  using Window = std::deque<Eigen::VectorXf>;

  static CellKey key(const grid_map::Index &index) { return {index.x(), index.y()}; }
  Eigen::VectorXf terrainFeatures(const HeightMap &map, const grid_map::Index &index) const;
  void ensureLayers(HeightMap &map) const;

  Config config_;
  TraversabilityNetwork network_;
  std::map<CellKey, Window> histories_;
};

}  // namespace lesta
