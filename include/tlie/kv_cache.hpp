#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "tlie/result.hpp"

namespace tlie {

class KvCache {
 public:
  static Result<KvCache> Create(std::size_t layers, std::size_t max_sequence_length,
                                std::size_t kv_dimension,
                                std::size_t max_bytes = std::numeric_limits<std::size_t>::max());

  [[nodiscard]] Result<void> Store(std::size_t layer, std::size_t position,
                                   std::span<const float> key, std::span<const float> value);
  [[nodiscard]] Result<std::span<const float>> Keys(std::size_t layer,
                                                    std::size_t sequence_length) const;
  [[nodiscard]] Result<std::span<const float>> Values(std::size_t layer,
                                                      std::size_t sequence_length) const;
  void Clear() noexcept;

  [[nodiscard]] std::size_t layers() const { return layers_; }
  [[nodiscard]] std::size_t max_sequence_length() const { return max_sequence_length_; }
  [[nodiscard]] std::size_t kv_dimension() const { return kv_dimension_; }

 private:
  KvCache(std::size_t layers, std::size_t max_sequence_length, std::size_t kv_dimension,
          std::vector<float> keys, std::vector<float> values);

  std::size_t layers_{0};
  std::size_t max_sequence_length_{0};
  std::size_t kv_dimension_{0};
  std::vector<float> keys_;
  std::vector<float> values_;
};

}  // namespace tlie
