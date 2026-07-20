#include "tlie/kv_cache.hpp"

#include <algorithm>
#include <limits>

namespace tlie {
namespace {

bool CheckedProduct(const std::size_t left, const std::size_t right, std::size_t* result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

}  // namespace

KvCache::KvCache(const std::size_t layers, const std::size_t max_sequence_length,
                 const std::size_t kv_dimension, std::vector<float> keys, std::vector<float> values)
    : layers_(layers),
      max_sequence_length_(max_sequence_length),
      kv_dimension_(kv_dimension),
      keys_(std::move(keys)),
      values_(std::move(values)) {}

Result<KvCache> KvCache::Create(const std::size_t layers, const std::size_t max_sequence_length,
                                const std::size_t kv_dimension, const std::size_t max_bytes) {
  if (layers == 0 || max_sequence_length == 0 || kv_dimension == 0) {
    return Result<KvCache>::Failure(
        {ErrorCode::kInvalidShape, "KV Cache dimensions must be non-zero"});
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  if (!CheckedProduct(layers, max_sequence_length, &elements) ||
      !CheckedProduct(elements, kv_dimension, &elements) ||
      !CheckedProduct(elements, sizeof(float) * 2, &bytes)) {
    return Result<KvCache>::Failure({ErrorCode::kOutOfMemory, "KV Cache size overflows"});
  }
  if (bytes > max_bytes) {
    return Result<KvCache>::Failure(
        {ErrorCode::kOutOfMemory, "KV Cache exceeds configured memory budget"});
  }
  try {
    return Result<KvCache>::Success(KvCache(layers, max_sequence_length, kv_dimension,
                                            std::vector<float>(elements, 0.0F),
                                            std::vector<float>(elements, 0.0F)));
  } catch (const std::bad_alloc&) {
    return Result<KvCache>::Failure({ErrorCode::kOutOfMemory, "KV Cache allocation failed"});
  }
}

Result<void> KvCache::Store(const std::size_t layer, const std::size_t position,
                            const std::span<const float> key, const std::span<const float> value) {
  if (layer >= layers_ || position >= max_sequence_length_) {
    return Result<void>::Failure({ErrorCode::kOutOfBounds, "KV Cache write is out of bounds"});
  }
  if (key.size() != kv_dimension_ || value.size() != kv_dimension_) {
    return Result<void>::Failure({ErrorCode::kInvalidShape, "KV Cache vector shape is invalid"});
  }
  const std::size_t offset = (layer * max_sequence_length_ + position) * kv_dimension_;
  std::copy(key.begin(), key.end(), keys_.begin() + static_cast<std::ptrdiff_t>(offset));
  std::copy(value.begin(), value.end(), values_.begin() + static_cast<std::ptrdiff_t>(offset));
  return Result<void>::Success();
}

Result<std::span<const float>> KvCache::Keys(const std::size_t layer,
                                             const std::size_t sequence_length) const {
  if (layer >= layers_ || sequence_length == 0 || sequence_length > max_sequence_length_) {
    return Result<std::span<const float>>::Failure(
        {ErrorCode::kOutOfBounds, "KV Cache key read is out of bounds"});
  }
  const std::size_t offset = layer * max_sequence_length_ * kv_dimension_;
  return Result<std::span<const float>>::Success(
      std::span<const float>(keys_.data() + offset, sequence_length * kv_dimension_));
}

Result<std::span<const float>> KvCache::Values(const std::size_t layer,
                                               const std::size_t sequence_length) const {
  if (layer >= layers_ || sequence_length == 0 || sequence_length > max_sequence_length_) {
    return Result<std::span<const float>>::Failure(
        {ErrorCode::kOutOfBounds, "KV Cache value read is out of bounds"});
  }
  const std::size_t offset = layer * max_sequence_length_ * kv_dimension_;
  return Result<std::span<const float>>::Success(
      std::span<const float>(values_.data() + offset, sequence_length * kv_dimension_));
}

void KvCache::Clear() noexcept {
  std::fill(keys_.begin(), keys_.end(), 0.0F);
  std::fill(values_.begin(), values_.end(), 0.0F);
}

}  // namespace tlie
