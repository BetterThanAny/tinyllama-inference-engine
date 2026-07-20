#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tlie/model_config.hpp"
#include "tlie/result.hpp"
#include "tlie/sha256.hpp"

namespace tlie {

enum class DType : std::uint8_t { kFloat32 = 1 };

struct TensorView {
  std::string name;
  DType dtype;
  std::vector<std::size_t> shape;
  std::span<const float> values;
  Sha256Digest checksum{};
};

struct WeightLoadOptions {
  std::uint64_t max_file_bytes{std::numeric_limits<std::uint64_t>::max()};
  bool verify_tensor_checksums{true};
  std::string expected_file_sha256;
  std::string expected_source_sha256;
};

class WeightStore {
 public:
  WeightStore() = default;
  ~WeightStore();
  WeightStore(const WeightStore&) = delete;
  WeightStore& operator=(const WeightStore&) = delete;
  WeightStore(WeightStore&& other) noexcept;
  WeightStore& operator=(WeightStore&& other) noexcept;

  static Result<WeightStore> Load(const std::filesystem::path& path,
                                  const WeightLoadOptions& options = {});

  [[nodiscard]] Result<TensorView> Get(std::string_view name) const;
  [[nodiscard]] Result<void> Validate(const std::vector<TensorSpec>& expected) const;
  [[nodiscard]] const Sha256Digest& source_checksum() const { return source_checksum_; }
  [[nodiscard]] std::size_t tensor_count() const { return tensors_.size(); }

 private:
  void Reset() noexcept;

  int file_descriptor_{-1};
  const std::byte* mapping_{nullptr};
  std::size_t mapping_size_{0};
  Sha256Digest source_checksum_{};
  std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace tlie
