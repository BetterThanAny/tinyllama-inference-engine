#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tlie/model_config.hpp"
#include "tlie/result.hpp"
#include "tlie/sha256.hpp"
#include "tlie/weight_store.hpp"

namespace tlie::cuda {

enum class CudaWeightDType { kFloat16, kInt8PerOutputChannel };

struct CudaTensorView {
  std::string name;
  std::vector<std::size_t> shape;
  CudaWeightDType dtype{CudaWeightDType::kFloat16};
  const __half* values{nullptr};
  const std::int8_t* quantized_values{nullptr};
  const float* scales{nullptr};
  std::size_t elements{0};
  Sha256Digest checksum{};
};

class CudaWeightStore {
 public:
  CudaWeightStore() = default;
  ~CudaWeightStore();
  CudaWeightStore(const CudaWeightStore&) = delete;
  CudaWeightStore& operator=(const CudaWeightStore&) = delete;
  CudaWeightStore(CudaWeightStore&& other) noexcept;
  CudaWeightStore& operator=(CudaWeightStore&& other) noexcept;

  static Result<CudaWeightStore> Load(const std::filesystem::path& path,
                                      const WeightLoadOptions& options);
  static Result<CudaWeightStore> LoadInt8(const std::filesystem::path& path,
                                          const WeightLoadOptions& options);

  [[nodiscard]] Result<CudaTensorView> Get(std::string_view name) const;
  [[nodiscard]] Result<void> Validate(const std::vector<TensorSpec>& expected) const;
  [[nodiscard]] std::size_t tensor_count() const { return tensors_.size(); }
  [[nodiscard]] std::size_t allocation_count() const { return allocations_.size(); }

 private:
  void Reset() noexcept;

  Sha256Digest source_checksum_{};
  std::vector<void*> allocations_;
  std::unordered_map<std::string, CudaTensorView> tensors_;
};

}  // namespace tlie::cuda
