#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "tlie/kv_cache.hpp"
#include "tlie/model_config.hpp"
#include "tlie/result.hpp"
#include "tlie/weight_store.hpp"

namespace tlie {

struct ModelTrace {
  std::map<std::string, std::vector<float>> tensors;
};

class TinyLlamaCpu {
 public:
  static Result<TinyLlamaCpu> Create(ModelConfig config, WeightStore weights,
                                     std::size_t max_sequence_length);

  TinyLlamaCpu(const TinyLlamaCpu&) = delete;
  TinyLlamaCpu& operator=(const TinyLlamaCpu&) = delete;
  TinyLlamaCpu(TinyLlamaCpu&&) noexcept = default;
  TinyLlamaCpu& operator=(TinyLlamaCpu&&) noexcept = default;

  [[nodiscard]] Result<std::vector<float>> Forward(int token, std::size_t position,
                                                   ModelTrace* trace = nullptr);
  void Reset() noexcept;
  [[nodiscard]] const ModelConfig& config() const { return config_; }

 private:
  TinyLlamaCpu(ModelConfig config, WeightStore weights, KvCache cache);

  [[nodiscard]] Result<void> Linear(const TensorView& weight, std::span<const float> input,
                                    std::span<float> output) const;

  ModelConfig config_;
  WeightStore weights_;
  KvCache cache_;
  std::size_t next_position_{0};
};

}  // namespace tlie
