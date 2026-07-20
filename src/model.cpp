#include "tlie/model.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#if defined(TLIE_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#include "tlie/operators.hpp"

namespace tlie {
namespace {

Result<TensorView> RequiredTensor(const WeightStore& weights, const std::string& name) {
  return weights.Get(name);
}

void AddInPlace(const std::span<float> destination, const std::span<const float> source) {
  for (std::size_t index = 0; index < destination.size(); ++index) {
    destination[index] += source[index];
  }
}

}  // namespace

TinyLlamaCpu::TinyLlamaCpu(ModelConfig config, WeightStore weights, KvCache cache)
    : config_(std::move(config)), weights_(std::move(weights)), cache_(std::move(cache)) {}

Result<TinyLlamaCpu> TinyLlamaCpu::Create(ModelConfig config, WeightStore weights,
                                          const std::size_t max_sequence_length) {
  auto config_validation = config.ValidateFixedTinyLlama();
  if (!config_validation.ok()) {
    return Result<TinyLlamaCpu>::Failure(config_validation.error());
  }
  if (max_sequence_length == 0 || max_sequence_length > config.max_position_embeddings) {
    return Result<TinyLlamaCpu>::Failure(
        {ErrorCode::kOutOfBounds, "requested sequence length exceeds the model limit"});
  }
  auto weight_validation = weights.Validate(config.ExpectedTensors());
  if (!weight_validation.ok()) {
    return Result<TinyLlamaCpu>::Failure(weight_validation.error());
  }
  auto cache = KvCache::Create(config.num_hidden_layers, max_sequence_length, config.kv_dim());
  if (!cache.ok()) {
    return Result<TinyLlamaCpu>::Failure(cache.error());
  }
  return Result<TinyLlamaCpu>::Success(
      TinyLlamaCpu(std::move(config), std::move(weights), std::move(cache).value()));
}

Result<void> TinyLlamaCpu::Linear(const TensorView& weight, const std::span<const float> input,
                                  const std::span<float> output) const {
  if (weight.shape.size() != 2 || weight.shape[1] != input.size() ||
      weight.shape[0] != output.size()) {
    return Result<void>::Failure(
        {ErrorCode::kInvalidShape, "linear weight, input, and output shapes are inconsistent"});
  }
#if defined(TLIE_USE_ACCELERATE)
  cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(weight.shape[0]),
              static_cast<int>(weight.shape[1]), 1.0F, weight.values.data(),
              static_cast<int>(weight.shape[1]), input.data(), 1, 0.0F, output.data(), 1);
#else
  for (std::size_t row = 0; row < weight.shape[0]; ++row) {
    float sum = 0.0F;
    const std::size_t base = row * weight.shape[1];
    for (std::size_t column = 0; column < weight.shape[1]; ++column) {
      sum += weight.values[base + column] * input[column];
    }
    output[row] = sum;
  }
#endif
  if (!AllFinite(output)) {
    return Result<void>::Failure({ErrorCode::kNumerical, "linear projection produced NaN or Inf"});
  }
  return Result<void>::Success();
}

Result<std::vector<float>> TinyLlamaCpu::Forward(const int token, const std::size_t position,
                                                 ModelTrace* trace) {
  if (token < 0 || static_cast<std::size_t>(token) >= config_.vocab_size) {
    return Result<std::vector<float>>::Failure(
        {ErrorCode::kOutOfBounds, "input token is outside the model vocabulary"});
  }
  if (position != next_position_ || position >= cache_.max_sequence_length()) {
    return Result<std::vector<float>>::Failure(
        {ErrorCode::kOutOfBounds, "model positions must be contiguous and within KV Cache bounds"});
  }
  const auto embedding = RequiredTensor(weights_, "model.embed_tokens.weight");
  if (!embedding.ok()) {
    return Result<std::vector<float>>::Failure(embedding.error());
  }
  std::vector<float> hidden(config_.hidden_size);
  const std::size_t embedding_offset = static_cast<std::size_t>(token) * config_.hidden_size;
  std::copy_n(embedding.value().values.begin() + static_cast<std::ptrdiff_t>(embedding_offset),
              config_.hidden_size, hidden.begin());
  if (trace != nullptr) {
    trace->tensors["embedding"] = hidden;
  }

  std::vector<float> normalized(config_.hidden_size);
  std::vector<float> query(config_.hidden_size);
  std::vector<float> key(config_.kv_dim());
  std::vector<float> value(config_.kv_dim());
  std::vector<float> attention(config_.hidden_size);
  std::vector<float> projected(config_.hidden_size);
  std::vector<float> gate(config_.intermediate_size);
  std::vector<float> up(config_.intermediate_size);
  std::vector<float> activated(config_.intermediate_size);

  for (std::size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    const auto input_norm = RequiredTensor(weights_, prefix + "input_layernorm.weight");
    const auto q_weight = RequiredTensor(weights_, prefix + "self_attn.q_proj.weight");
    const auto k_weight = RequiredTensor(weights_, prefix + "self_attn.k_proj.weight");
    const auto v_weight = RequiredTensor(weights_, prefix + "self_attn.v_proj.weight");
    const auto o_weight = RequiredTensor(weights_, prefix + "self_attn.o_proj.weight");
    const auto post_norm = RequiredTensor(weights_, prefix + "post_attention_layernorm.weight");
    const auto gate_weight = RequiredTensor(weights_, prefix + "mlp.gate_proj.weight");
    const auto up_weight = RequiredTensor(weights_, prefix + "mlp.up_proj.weight");
    const auto down_weight = RequiredTensor(weights_, prefix + "mlp.down_proj.weight");
    const Result<TensorView>* required[] = {&input_norm,  &q_weight,  &k_weight,
                                            &v_weight,    &o_weight,  &post_norm,
                                            &gate_weight, &up_weight, &down_weight};
    for (const auto* tensor : required) {
      if (!tensor->ok()) {
        return Result<std::vector<float>>::Failure(tensor->error());
      }
    }

    auto operation = RmsNorm(hidden, input_norm.value().values, config_.rms_norm_eps, normalized);
    if (!operation.ok()) {
      return Result<std::vector<float>>::Failure(operation.error());
    }
    const std::string trace_prefix = "layer" + std::to_string(layer) + ".";
    if (trace != nullptr) {
      trace->tensors[trace_prefix + "input_norm"] = normalized;
    }
    operation = Linear(q_weight.value(), normalized, query);
    if (operation.ok()) {
      operation = Linear(k_weight.value(), normalized, key);
    }
    if (operation.ok()) {
      operation = Linear(v_weight.value(), normalized, value);
    }
    if (!operation.ok()) {
      return Result<std::vector<float>>::Failure(operation.error());
    }
    operation = ApplyRope(query, key, config_.num_attention_heads, config_.num_key_value_heads,
                          config_.head_dim(), position, config_.rope_theta);
    if (!operation.ok()) {
      return Result<std::vector<float>>::Failure(operation.error());
    }
    operation = cache_.Store(layer, position, key, value);
    if (!operation.ok()) {
      return Result<std::vector<float>>::Failure(operation.error());
    }
    const auto keys = cache_.Keys(layer, position + 1);
    const auto values = cache_.Values(layer, position + 1);
    if (!keys.ok()) {
      return Result<std::vector<float>>::Failure(keys.error());
    }
    if (!values.ok()) {
      return Result<std::vector<float>>::Failure(values.error());
    }
    operation = AttentionReference(query, keys.value(), values.value(), position + 1,
                                   config_.num_attention_heads, config_.num_key_value_heads,
                                   config_.head_dim(), attention);
    if (operation.ok()) {
      operation = Linear(o_weight.value(), attention, projected);
    }
    if (!operation.ok()) {
      return Result<std::vector<float>>::Failure(operation.error());
    }
    if (trace != nullptr) {
      trace->tensors[trace_prefix + "attention_output"] = projected;
    }
    AddInPlace(hidden, projected);
    operation = RmsNorm(hidden, post_norm.value().values, config_.rms_norm_eps, normalized);
    if (!operation.ok()) {
      return Result<std::vector<float>>::Failure(operation.error());
    }
    if (trace != nullptr) {
      trace->tensors[trace_prefix + "post_attention_norm"] = normalized;
    }
    operation = Linear(gate_weight.value(), normalized, gate);
    if (operation.ok()) {
      operation = Linear(up_weight.value(), normalized, up);
    }
    if (operation.ok()) {
      operation = SiluMultiply(gate, up, activated);
    }
    if (operation.ok()) {
      operation = Linear(down_weight.value(), activated, projected);
    }
    if (!operation.ok()) {
      return Result<std::vector<float>>::Failure(operation.error());
    }
    if (trace != nullptr) {
      trace->tensors[trace_prefix + "mlp_output"] = projected;
    }
    AddInPlace(hidden, projected);
    if (trace != nullptr) {
      trace->tensors[trace_prefix + "output"] = hidden;
    }
  }

  const auto final_norm = RequiredTensor(weights_, "model.norm.weight");
  const auto lm_head = RequiredTensor(weights_, "lm_head.weight");
  if (!final_norm.ok()) {
    return Result<std::vector<float>>::Failure(final_norm.error());
  }
  if (!lm_head.ok()) {
    return Result<std::vector<float>>::Failure(lm_head.error());
  }
  auto operation = RmsNorm(hidden, final_norm.value().values, config_.rms_norm_eps, normalized);
  if (!operation.ok()) {
    return Result<std::vector<float>>::Failure(operation.error());
  }
  if (trace != nullptr) {
    trace->tensors["final_norm"] = normalized;
  }
  std::vector<float> logits(config_.vocab_size);
  operation = Linear(lm_head.value(), normalized, logits);
  if (!operation.ok()) {
    return Result<std::vector<float>>::Failure(operation.error());
  }
  if (trace != nullptr) {
    trace->tensors["logits"] = logits;
  }
  ++next_position_;
  return Result<std::vector<float>>::Success(std::move(logits));
}

void TinyLlamaCpu::Reset() noexcept {
  cache_.Clear();
  next_position_ = 0;
}

}  // namespace tlie
