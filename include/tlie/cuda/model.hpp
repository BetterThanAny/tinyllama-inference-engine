#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "tlie/cuda/weight_store.hpp"
#include "tlie/model_config.hpp"
#include "tlie/result.hpp"

namespace tlie::cuda {

struct CudaStepTimings {
  float compute_ms{0.0F};
  float transfer_ms{0.0F};
  float wall_ms{0.0F};
};

class TinyLlamaCuda {
 public:
  static Result<TinyLlamaCuda> Create(ModelConfig config, CudaWeightStore weights,
                                      std::size_t max_sequence_length,
                                      bool allow_rope_extrapolation = false,
                                      std::size_t max_batch_size = 1,
                                      std::size_t kv_slot_count = 1);

  ~TinyLlamaCuda();
  TinyLlamaCuda(const TinyLlamaCuda&) = delete;
  TinyLlamaCuda& operator=(const TinyLlamaCuda&) = delete;
  TinyLlamaCuda(TinyLlamaCuda&&) noexcept;
  TinyLlamaCuda& operator=(TinyLlamaCuda&&) noexcept;

  [[nodiscard]] Result<std::vector<float>> Forward(int token, std::size_t position,
                                                   CudaStepTimings* timings = nullptr);
  [[nodiscard]] Result<std::vector<std::vector<float>>> ForwardBatch(
      std::span<const int> tokens, std::span<const std::size_t> positions,
      std::span<const std::size_t> kv_slots, CudaStepTimings* timings = nullptr);
  [[nodiscard]] Result<void> Reset();
  [[nodiscard]] Result<void> ResetSlot(std::size_t kv_slot);
  [[nodiscard]] std::size_t max_sequence_length() const;
  [[nodiscard]] std::size_t max_batch_size() const;
  [[nodiscard]] std::size_t kv_slot_count() const;
  [[nodiscard]] std::size_t device_allocation_count() const;
  [[nodiscard]] std::size_t kv_cache_bytes() const;
  [[nodiscard]] const ModelConfig& config() const;

 private:
  struct Impl;
  explicit TinyLlamaCuda(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace tlie::cuda
