#include "lesta/core/TemporalTraversabilityEstimator.h"

#include <cmath>
#include <stdexcept>

namespace lesta {

TemporalTraversabilityEstimator::TemporalTraversabilityEstimator(const Config &config)
    : config_(config) {
  if (config_.input_dimension != 21 || config_.sequence_length <= 0)
    throw std::invalid_argument("Temporal estimator expects 21 features and a positive window");
  if (!network_.loadCheckpoint(config_.model_path))
    throw std::runtime_error("Failed to load temporal TorchScript model: " + config_.model_path);
  network_.setInputDimension(config_.input_dimension);
}

void TemporalTraversabilityEstimator::reset() { histories_.clear(); }

Eigen::VectorXf TemporalTraversabilityEstimator::terrainFeatures(
    const HeightMap &map, const grid_map::Index &index) const {
  const char *feature_layers[] = {
      layers::Feature::STEP, layers::Feature::SLOPE, layers::Feature::ROUGHNESS,
      layers::Feature::CURVATURE, layers::Feature::HEIGHT_RANGE,
      layers::Feature::ELEVATION_IQR, layers::Feature::ROUGHNESS_P95,
      layers::Feature::OCCUPIED_CELL_RATIO};
  Eigen::VectorXf features(8);
  for (int i = 0; i < 8; ++i) {
    if (map.isEmptyAt(feature_layers[i], index))
      return {};
    features(i) = map.at(feature_layers[i], index);
  }
  // The original ROS2 feature layer stores slope in degrees; the temporal
  // dataset uses radians, so convert only at this model boundary.
  features(1) = features(1) * static_cast<float>(M_PI / 180.0);
  return features;
}

void TemporalTraversabilityEstimator::ensureLayers(HeightMap &map) const {
  map.addLayer(layers::Traversability::PROBABILITY);
  map.addLayer(layers::Traversability::BINARY);
}

void TemporalTraversabilityEstimator::estimate(
    HeightMap &map, const std::vector<grid_map::Index> &indices,
    const Eigen::VectorXf &motion_features) {
  if (motion_features.size() != 13)
    throw std::invalid_argument("Temporal motion feature vector must have 13 values");
  ensureLayers(map);

  std::vector<std::vector<Eigen::VectorXf>> ready_windows;
  std::vector<grid_map::Index> ready_indices;
  for (const auto &index : indices) {
    const auto terrain = terrainFeatures(map, index);
    if (terrain.size() == 0)
      continue;
    Eigen::VectorXf frame(21);
    frame << motion_features, terrain;
    auto &history = histories_[key(index)];
    history.push_back(std::move(frame));
    while (history.size() > static_cast<size_t>(config_.sequence_length))
      history.pop_front();
    if (history.size() == static_cast<size_t>(config_.sequence_length)) {
      ready_windows.emplace_back(history.begin(), history.end());
      ready_indices.push_back(index);
    }
  }

  const auto logits = network_.inferenceTemporal(ready_windows);
  for (size_t i = 0; i < logits.size(); ++i) {
    const float probability = logits[i] >= 0.0f
                                  ? 1.0f / (1.0f + std::exp(-logits[i]))
                                  : std::exp(logits[i]) / (1.0f + std::exp(logits[i]));
    map.at(layers::Traversability::PROBABILITY, ready_indices[i]) = probability;
    map.at(layers::Traversability::BINARY, ready_indices[i]) =
        probability >= config_.binary_threshold ? 1.0f : 0.0f;
  }
}

}  // namespace lesta
