#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "tlie/result.hpp"

namespace tlie {

Result<void> RmsNorm(std::span<const float> input, std::span<const float> weight, float epsilon,
                     std::span<float> output);
Result<void> ApplyRope(std::span<float> query, std::span<float> key, std::size_t query_heads,
                       std::size_t key_value_heads, std::size_t head_dim, std::size_t position,
                       float theta);
Result<void> SoftmaxInPlace(std::span<float> values);
Result<void> AttentionReference(std::span<const float> query, std::span<const float> keys,
                                std::span<const float> values, std::size_t sequence_length,
                                std::size_t query_heads, std::size_t key_value_heads,
                                std::size_t head_dim, std::span<float> output);
Result<void> SiluMultiply(std::span<const float> gate, std::span<const float> up,
                          std::span<float> output);
Result<int> GreedySample(std::span<const float> logits);
bool AllFinite(std::span<const float> values);

}  // namespace tlie
