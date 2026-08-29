/*
 * TraversabilityNetwork.cpp
 *
 *  Created on: Feb 15, 2025
 *      Author: Ikhyeon Cho
 *	 Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include "lesta/core/TraversabilityNetwork.h"
#include <torch/torch.h>
#include <iostream>

namespace lesta {

bool TraversabilityNetwork::loadCheckpoint(const std::string &model_path) {
  try {
    // Load the TorchScript model
    torch::Device device = torch::kCPU;
    // torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    model_ = torch::jit::load(model_path, device);
    model_.eval();
    model_loaded_ = true;

    // Extract just the filename from the full path
    std::string model_name = model_path;
    size_t last_slash_pos = model_path.find_last_of("/\\");
    if (last_slash_pos != std::string::npos) {
      model_name = model_path.substr(last_slash_pos + 1);
    }

    // Print model information
    std::cout << "\033[1;32m[lesta::TraversabilityNetwork]: Model loaded to "
              << (device.is_cuda() ? "GPU" : "CPU") << "\033[0m" << std::endl;
    std::cout << "\033[1;32m[lesta::TraversabilityNetwork]: Model checkpoint: "
              << model_name << "\033[0m" << std::endl;
    return true;
  } catch (const c10::Error &e) {
    // std::cerr << "\033[1;31m[lesta::TraversabilityNetwork]: Error loading the model: "
    // << e.what() << "\033[0m" << std::endl;
    return false;
  }
}

std::vector<float>
TraversabilityNetwork::inference(const std::vector<Eigen::VectorXf> &features) {
  if (!model_loaded_) {
    throw std::runtime_error("Model not loaded");
  }

  std::vector<float> predictions;
  predictions.reserve(features.size());

  try {
    // Set to evaluation mode
    torch::NoGradGuard no_grad;

    for (const auto &feature : features) {
      // Convert Eigen vector to torch tensor
      std::vector<float> feature_vec(feature.data(), feature.data() + feature.size());
      auto options = torch::TensorOptions().dtype(torch::kFloat32);
      torch::Tensor input_tensor =
          torch::from_blob(feature_vec.data(), {1, input_dimension_}, options).clone();

      // Perform inference
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(input_tensor);

      // Forward pass
      torch::Tensor output = model_.forward(inputs).toTensor();

      // Get prediction value
      float prediction = output.item<float>();
      predictions.push_back(prediction);
    }
  } catch (const c10::Error &e) {
    std::cerr << "Error during inference: " << e.what() << std::endl;
    throw;
  }

  return predictions;
}

std::vector<float> TraversabilityNetwork::inferenceTemporal(
    const std::vector<std::vector<Eigen::VectorXf>> &feature_windows) {
  if (!model_loaded_) {
    throw std::runtime_error("Model not loaded");
  }
  if (feature_windows.empty()) {
    return {};
  }

  const auto sequence_length = feature_windows.front().size();
  if (sequence_length == 0) {
    throw std::invalid_argument("Temporal feature window must not be empty");
  }

  std::vector<float> flattened;
  flattened.reserve(feature_windows.size() * sequence_length * input_dimension_);
  for (const auto &window : feature_windows) {
    if (window.size() != sequence_length) {
      throw std::invalid_argument("Temporal windows must have equal lengths");
    }
    for (const auto &feature : window) {
      if (feature.size() != input_dimension_) {
        throw std::invalid_argument("Temporal feature dimension mismatch");
      }
      flattened.insert(flattened.end(), feature.data(), feature.data() + feature.size());
    }
  }

  try {
    torch::NoGradGuard no_grad;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto input = torch::from_blob(
        flattened.data(),
        {static_cast<long>(feature_windows.size()),
         static_cast<long>(sequence_length),
         static_cast<long>(input_dimension_)},
        options).clone();
    auto output = model_.forward({input}).toTensor().flatten().contiguous();
    if (output.numel() != static_cast<long>(feature_windows.size())) {
      throw std::runtime_error("Temporal model output size does not match batch size");
    }
    const auto output_cpu = output.to(torch::kCPU);
    const auto *data = output_cpu.data_ptr<float>();
    return std::vector<float>(data, data + output_cpu.numel());
  } catch (const c10::Error &e) {
    std::cerr << "Error during temporal inference: " << e.what() << std::endl;
    throw;
  }
}

} // namespace lesta
