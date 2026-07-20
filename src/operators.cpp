#include "tlie/operators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace tlie {
namespace {

Result<void> ShapeError(const std::string& message) {
  return Result<void>::Failure({ErrorCode::kInvalidShape, message});
}

}  // namespace

bool AllFinite(const std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](const float value) { return std::isfinite(value); });
}

Result<void> RmsNorm(const std::span<const float> input, const std::span<const float> weight,
                     const float epsilon, const std::span<float> output) {
  if (input.empty() || input.size() != weight.size() || input.size() != output.size()) {
    return ShapeError("RMSNorm input, weight, and output shapes must match and be non-empty");
  }
  if (!(epsilon > 0.0F) || !AllFinite(input) || !AllFinite(weight)) {
    return Result<void>::Failure(
        {ErrorCode::kNumerical, "RMSNorm requires finite inputs and positive epsilon"});
  }
  double square_sum = 0.0;
  for (const float value : input) {
    square_sum += static_cast<double>(value) * static_cast<double>(value);
  }
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(square_sum / static_cast<double>(input.size())) + epsilon);
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index] * scale * weight[index];
  }
  if (!AllFinite(output)) {
    return Result<void>::Failure({ErrorCode::kNumerical, "RMSNorm produced NaN or Inf"});
  }
  return Result<void>::Success();
}

Result<void> ApplyRope(const std::span<float> query, const std::span<float> key,
                       const std::size_t query_heads, const std::size_t key_value_heads,
                       const std::size_t head_dim, const std::size_t position, const float theta) {
  if (head_dim == 0 || head_dim % 2 != 0 || query.size() != query_heads * head_dim ||
      key.size() != key_value_heads * head_dim) {
    return ShapeError("RoPE requires matching heads and an even, non-zero head dimension");
  }
  if (!(theta > 0.0F) || !AllFinite(query) || !AllFinite(key)) {
    return Result<void>::Failure(
        {ErrorCode::kNumerical, "RoPE requires finite inputs and positive theta"});
  }
  const std::size_t half = head_dim / 2;
  const auto rotate = [&](const std::span<float> tensor, const std::size_t heads) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t base = head * head_dim;
      for (std::size_t dimension = 0; dimension < half; ++dimension) {
        const float frequency =
            std::pow(theta, -2.0F * static_cast<float>(dimension) / static_cast<float>(head_dim));
        const float angle = static_cast<float>(position) * frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = tensor[base + dimension];
        const float second = tensor[base + dimension + half];
        tensor[base + dimension] = first * cosine - second * sine;
        tensor[base + dimension + half] = second * cosine + first * sine;
      }
    }
  };
  rotate(query, query_heads);
  rotate(key, key_value_heads);
  return Result<void>::Success();
}

Result<void> SoftmaxInPlace(const std::span<float> values) {
  if (values.empty()) {
    return ShapeError("softmax input must not be empty");
  }
  if (!AllFinite(values)) {
    return Result<void>::Failure({ErrorCode::kNumerical, "softmax input contains NaN or Inf"});
  }
  const float maximum = *std::max_element(values.begin(), values.end());
  double sum = 0.0;
  for (float& value : values) {
    value = std::exp(value - maximum);
    sum += value;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    return Result<void>::Failure({ErrorCode::kNumerical, "softmax normalization is invalid"});
  }
  for (float& value : values) {
    value = static_cast<float>(static_cast<double>(value) / sum);
  }
  return Result<void>::Success();
}

Result<void> AttentionReference(const std::span<const float> query,
                                const std::span<const float> keys,
                                const std::span<const float> values,
                                const std::size_t sequence_length, const std::size_t query_heads,
                                const std::size_t key_value_heads, const std::size_t head_dim,
                                const std::span<float> output) {
  if (sequence_length == 0 || query_heads == 0 || key_value_heads == 0 || head_dim == 0 ||
      query_heads % key_value_heads != 0 || query.size() != query_heads * head_dim ||
      keys.size() != sequence_length * key_value_heads * head_dim || keys.size() != values.size() ||
      output.size() != query.size()) {
    return ShapeError("attention tensor shapes or grouped-query head counts are invalid");
  }
  if (!AllFinite(query) || !AllFinite(keys) || !AllFinite(values)) {
    return Result<void>::Failure({ErrorCode::kNumerical, "attention input contains NaN or Inf"});
  }
  const std::size_t groups = query_heads / key_value_heads;
  const std::size_t kv_stride = key_value_heads * head_dim;
  const float inverse_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> scores(sequence_length);
  std::fill(output.begin(), output.end(), 0.0F);
  for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
    const std::size_t kv_head = query_head / groups;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      float dot = 0.0F;
      const std::size_t query_base = query_head * head_dim;
      const std::size_t key_base = position * kv_stride + kv_head * head_dim;
      for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
        dot += query[query_base + dimension] * keys[key_base + dimension];
      }
      scores[position] = dot * inverse_scale;
    }
    auto softmax = SoftmaxInPlace(scores);
    if (!softmax.ok()) {
      return softmax;
    }
    const std::size_t output_base = query_head * head_dim;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t value_base = position * kv_stride + kv_head * head_dim;
      for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
        output[output_base + dimension] += scores[position] * values[value_base + dimension];
      }
    }
  }
  return Result<void>::Success();
}

Result<void> SiluMultiply(const std::span<const float> gate, const std::span<const float> up,
                          const std::span<float> output) {
  if (gate.empty() || gate.size() != up.size() || gate.size() != output.size()) {
    return ShapeError("SiLU multiply input and output shapes must match and be non-empty");
  }
  if (!AllFinite(gate) || !AllFinite(up)) {
    return Result<void>::Failure(
        {ErrorCode::kNumerical, "SiLU multiply input contains NaN or Inf"});
  }
  for (std::size_t index = 0; index < gate.size(); ++index) {
    output[index] = (gate[index] / (1.0F + std::exp(-gate[index]))) * up[index];
  }
  if (!AllFinite(output)) {
    return Result<void>::Failure({ErrorCode::kNumerical, "SiLU multiply produced NaN or Inf"});
  }
  return Result<void>::Success();
}

Result<int> GreedySample(const std::span<const float> logits) {
  if (logits.empty()) {
    return Result<int>::Failure({ErrorCode::kInvalidShape, "cannot sample empty logits"});
  }
  if (!AllFinite(logits)) {
    return Result<int>::Failure({ErrorCode::kNumerical, "logits contain NaN or Inf"});
  }
  const auto maximum = std::max_element(logits.begin(), logits.end());
  return Result<int>::Success(static_cast<int>(std::distance(logits.begin(), maximum)));
}

}  // namespace tlie
