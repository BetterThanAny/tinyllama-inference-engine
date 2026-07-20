#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <utility>

#include "tlie/cuda/kernels.hpp"
#include "tlie/cuda/model.hpp"

namespace tlie::cuda {
namespace {

Result<void> CudaFailure(const std::string& operation, const cudaError_t error) {
  const ErrorCode code =
      error == cudaErrorMemoryAllocation ? ErrorCode::kOutOfMemory : ErrorCode::kCuda;
  return Result<void>::Failure(
      {code, operation + " failed: " + std::string(cudaGetErrorString(error))});
}

Result<void> CublasFailure(const std::string& operation, const cublasStatus_t status) {
  return Result<void>::Failure(
      {ErrorCode::kCuda, operation + " failed with cuBLAS status " + std::to_string(status)});
}

struct LayerWeights {
  CudaTensorView input_norm;
  CudaTensorView query;
  CudaTensorView key;
  CudaTensorView value;
  CudaTensorView attention_output;
  CudaTensorView post_attention_norm;
  CudaTensorView gate;
  CudaTensorView up;
  CudaTensorView down;
};

}  // namespace

struct TinyLlamaCuda::Impl {
  ModelConfig model_config;
  CudaWeightStore weight_store;
  std::size_t maximum_sequence_length{0};
  std::size_t maximum_batch_size{1};
  std::size_t slot_count{1};
  std::vector<std::size_t> next_positions;
  cudaStream_t stream{nullptr};
  cublasHandle_t cublas{nullptr};
  cudaEvent_t compute_start{nullptr};
  cudaEvent_t compute_stop{nullptr};
  cudaEvent_t transfer_stop{nullptr};
  std::vector<void*> allocations;
  std::vector<LayerWeights> layers;
  CudaTensorView embeddings;
  CudaTensorView final_norm_weight;
  CudaTensorView lm_head;
  __half* hidden{nullptr};
  __half* normalized{nullptr};
  __half* query{nullptr};
  __half* key{nullptr};
  __half* value{nullptr};
  __half* attention{nullptr};
  __half* projected{nullptr};
  __half* gate{nullptr};
  __half* up{nullptr};
  __half* activated{nullptr};
  __half* logits{nullptr};
  __half* key_cache{nullptr};
  __half* value_cache{nullptr};
  std::vector<__half> host_logits;

  Impl(ModelConfig config, CudaWeightStore weights, const std::size_t max_sequence_length,
       const std::size_t max_batch, const std::size_t kv_slots)
      : model_config(std::move(config)),
        weight_store(std::move(weights)),
        maximum_sequence_length(max_sequence_length),
        maximum_batch_size(max_batch),
        slot_count(kv_slots),
        next_positions(kv_slots, 0) {}

  ~Impl() {
    for (void* allocation : allocations) {
      if (allocation != nullptr) {
        cudaFree(allocation);
      }
    }
    if (compute_start != nullptr) {
      cudaEventDestroy(compute_start);
    }
    if (compute_stop != nullptr) {
      cudaEventDestroy(compute_stop);
    }
    if (transfer_stop != nullptr) {
      cudaEventDestroy(transfer_stop);
    }
    if (cublas != nullptr) {
      cublasDestroy(cublas);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  Result<void> Allocate(const std::size_t elements, __half** output) {
    if (elements == 0 || output == nullptr ||
        elements > std::numeric_limits<std::size_t>::max() / sizeof(__half)) {
      return Result<void>::Failure(
          {ErrorCode::kOutOfMemory, "CUDA workspace allocation size is invalid"});
    }
    void* allocation = nullptr;
    const cudaError_t status = cudaMalloc(&allocation, elements * sizeof(__half));
    if (status != cudaSuccess) {
      return CudaFailure("cudaMalloc workspace", status);
    }
    allocations.push_back(allocation);
    *output = static_cast<__half*>(allocation);
    return Result<void>::Success();
  }

  Result<CudaTensorView> Tensor(const std::string& name) const {
    const auto tensor = weight_store.Get(name);
    if (!tensor.ok()) {
      return Result<CudaTensorView>::Failure(tensor.error());
    }
    return tensor;
  }

  Result<void> Linear(const CudaTensorView& weight, const __half* input, __half* output,
                      const std::size_t output_rows, const std::size_t input_columns,
                      const std::size_t batch) const {
    if (weight.shape != std::vector<std::size_t>{output_rows, input_columns}) {
      return Result<void>::Failure(
          {ErrorCode::kInvalidShape, "CUDA linear weight shape is inconsistent"});
    }
    if (weight.dtype == CudaWeightDType::kFloat16) {
      return GemmFp16RowMajor(cublas, weight.values, input, output, output_rows, input_columns,
                              batch, stream);
    }
    for (std::size_t row = 0; row < batch; ++row) {
      auto result = LaunchInt8WeightOnlyGemv(
          weight.quantized_values, weight.scales, input + row * input_columns,
          output + row * output_rows, output_rows, input_columns, stream);
      if (!result.ok()) {
        return result;
      }
    }
    return Result<void>::Success();
  }

  Result<void> Initialize() {
    cudaError_t cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      return CudaFailure("cudaStreamCreateWithFlags", cuda_status);
    }
    cublasStatus_t cublas_status = cublasCreate(&cublas);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return CublasFailure("cublasCreate", cublas_status);
    }
    cublas_status = cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return CublasFailure("cublasSetMathMode", cublas_status);
    }
    cuda_status = cudaEventCreate(&compute_start);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaEventCreate(&compute_stop);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaEventCreate(&transfer_stop);
    }
    if (cuda_status != cudaSuccess) {
      return CudaFailure("cudaEventCreate", cuda_status);
    }

    const auto embedding = Tensor("model.embed_tokens.weight");
    const auto final_norm = Tensor("model.norm.weight");
    const auto head = Tensor("lm_head.weight");
    if (!embedding.ok()) {
      return Result<void>::Failure(embedding.error());
    }
    if (!final_norm.ok()) {
      return Result<void>::Failure(final_norm.error());
    }
    if (!head.ok()) {
      return Result<void>::Failure(head.error());
    }
    embeddings = embedding.value();
    final_norm_weight = final_norm.value();
    lm_head = head.value();
    if (final_norm_weight.dtype != CudaWeightDType::kFloat16 ||
        final_norm_weight.values == nullptr) {
      return Result<void>::Failure(
          {ErrorCode::kInvalidDtype, "CUDA final RMSNorm weight must remain FP16"});
    }

    layers.reserve(model_config.num_hidden_layers);
    for (std::size_t layer = 0; layer < model_config.num_hidden_layers; ++layer) {
      const std::string prefix = "model.layers." + std::to_string(layer) + ".";
      const auto input_norm = Tensor(prefix + "input_layernorm.weight");
      const auto query_weight = Tensor(prefix + "self_attn.q_proj.weight");
      const auto key_weight = Tensor(prefix + "self_attn.k_proj.weight");
      const auto value_weight = Tensor(prefix + "self_attn.v_proj.weight");
      const auto output_weight = Tensor(prefix + "self_attn.o_proj.weight");
      const auto post_norm = Tensor(prefix + "post_attention_layernorm.weight");
      const auto gate_weight = Tensor(prefix + "mlp.gate_proj.weight");
      const auto up_weight = Tensor(prefix + "mlp.up_proj.weight");
      const auto down_weight = Tensor(prefix + "mlp.down_proj.weight");
      const Result<CudaTensorView>* required[] = {&input_norm,   &query_weight,  &key_weight,
                                                  &value_weight, &output_weight, &post_norm,
                                                  &gate_weight,  &up_weight,     &down_weight};
      for (const auto* tensor : required) {
        if (!tensor->ok()) {
          return Result<void>::Failure(tensor->error());
        }
      }
      if (input_norm.value().dtype != CudaWeightDType::kFloat16 ||
          post_norm.value().dtype != CudaWeightDType::kFloat16 ||
          input_norm.value().values == nullptr || post_norm.value().values == nullptr) {
        return Result<void>::Failure(
            {ErrorCode::kInvalidDtype, "CUDA RMSNorm weights must remain FP16"});
      }
      layers.push_back({input_norm.value(), query_weight.value(), key_weight.value(),
                        value_weight.value(), output_weight.value(), post_norm.value(),
                        gate_weight.value(), up_weight.value(), down_weight.value()});
    }

    auto allocated = Allocate(maximum_batch_size * model_config.hidden_size, &hidden);
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.hidden_size, &normalized);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.hidden_size, &query);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.kv_dim(), &key);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.kv_dim(), &value);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.hidden_size, &attention);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.hidden_size, &projected);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.intermediate_size, &gate);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.intermediate_size, &up);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.intermediate_size, &activated);
    }
    if (allocated.ok()) {
      allocated = Allocate(maximum_batch_size * model_config.vocab_size, &logits);
    }
    std::size_t cache_elements = model_config.num_hidden_layers * maximum_sequence_length;
    if (cache_elements > std::numeric_limits<std::size_t>::max() / model_config.kv_dim()) {
      return Result<void>::Failure({ErrorCode::kOutOfMemory, "CUDA KV Cache size overflows"});
    }
    cache_elements *= model_config.kv_dim();
    if (cache_elements > std::numeric_limits<std::size_t>::max() / slot_count) {
      return Result<void>::Failure({ErrorCode::kOutOfMemory, "CUDA KV slot size overflows"});
    }
    cache_elements *= slot_count;
    if (allocated.ok()) {
      allocated = Allocate(cache_elements, &key_cache);
    }
    if (allocated.ok()) {
      allocated = Allocate(cache_elements, &value_cache);
    }
    if (!allocated.ok()) {
      return allocated;
    }
    host_logits.resize(maximum_batch_size * model_config.vocab_size);
    return Reset();
  }

  Result<void> Reset() {
    const std::size_t cache_elements = slot_count * model_config.num_hidden_layers *
                                       maximum_sequence_length * model_config.kv_dim();
    cudaError_t status = cudaMemsetAsync(key_cache, 0, cache_elements * sizeof(__half), stream);
    if (status == cudaSuccess) {
      status = cudaMemsetAsync(value_cache, 0, cache_elements * sizeof(__half), stream);
    }
    if (status != cudaSuccess) {
      return CudaFailure("CUDA KV Cache reset", status);
    }
    auto synchronized = Synchronize(stream);
    if (synchronized.ok()) {
      std::fill(next_positions.begin(), next_positions.end(), 0);
    }
    return synchronized;
  }

  Result<void> ResetSlot(const std::size_t slot) {
    if (slot >= slot_count) {
      return Result<void>::Failure({ErrorCode::kOutOfBounds, "CUDA KV slot is out of bounds"});
    }
    const std::size_t slot_elements =
        model_config.num_hidden_layers * maximum_sequence_length * model_config.kv_dim();
    cudaError_t status = cudaMemsetAsync(key_cache + slot * slot_elements, 0,
                                         slot_elements * sizeof(__half), stream);
    if (status == cudaSuccess) {
      status = cudaMemsetAsync(value_cache + slot * slot_elements, 0,
                               slot_elements * sizeof(__half), stream);
    }
    if (status != cudaSuccess) {
      return CudaFailure("CUDA KV slot reset", status);
    }
    auto synchronized = Synchronize(stream);
    if (synchronized.ok()) {
      next_positions[slot] = 0;
    }
    return synchronized;
  }
};

TinyLlamaCuda::TinyLlamaCuda(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TinyLlamaCuda::~TinyLlamaCuda() = default;
TinyLlamaCuda::TinyLlamaCuda(TinyLlamaCuda&&) noexcept = default;
TinyLlamaCuda& TinyLlamaCuda::operator=(TinyLlamaCuda&&) noexcept = default;

Result<TinyLlamaCuda> TinyLlamaCuda::Create(ModelConfig config, CudaWeightStore weights,
                                            const std::size_t max_sequence_length,
                                            const bool allow_rope_extrapolation,
                                            const std::size_t max_batch_size,
                                            const std::size_t kv_slot_count) {
  auto config_validation = config.ValidateFixedTinyLlama();
  if (!config_validation.ok()) {
    return Result<TinyLlamaCuda>::Failure(config_validation.error());
  }
  auto weight_validation = weights.Validate(config.ExpectedTensors());
  if (!weight_validation.ok()) {
    return Result<TinyLlamaCuda>::Failure(weight_validation.error());
  }
  // M2 benchmarks admit a 4096-token input followed by at most 256 decode tokens. The model's
  // trained context is still 2048; callers must explicitly acknowledge RoPE extrapolation.
  constexpr std::size_t kMaximumM2SequenceLength = 4096 + 256;
  if (max_sequence_length == 0 || max_sequence_length > kMaximumM2SequenceLength ||
      max_batch_size == 0 || max_batch_size > 64 || kv_slot_count == 0 || kv_slot_count > 256 ||
      max_batch_size > kv_slot_count ||
      (max_sequence_length > config.max_position_embeddings && !allow_rope_extrapolation)) {
    return Result<TinyLlamaCuda>::Failure(
        {ErrorCode::kOutOfBounds,
         "CUDA sequence length exceeds the model limit without explicit RoPE extrapolation"});
  }
  try {
    auto impl = std::make_unique<Impl>(std::move(config), std::move(weights), max_sequence_length,
                                       max_batch_size, kv_slot_count);
    auto initialized = impl->Initialize();
    if (!initialized.ok()) {
      return Result<TinyLlamaCuda>::Failure(initialized.error());
    }
    return Result<TinyLlamaCuda>::Success(TinyLlamaCuda(std::move(impl)));
  } catch (const std::bad_alloc&) {
    return Result<TinyLlamaCuda>::Failure(
        {ErrorCode::kOutOfMemory, "host allocation failed while creating CUDA model"});
  }
}

Result<std::vector<float>> TinyLlamaCuda::Forward(const int token, const std::size_t position,
                                                  CudaStepTimings* timings) {
  if (impl_ == nullptr) {
    return Result<std::vector<float>>::Failure({ErrorCode::kCuda, "CUDA model is not initialized"});
  }
  if (token < 0 || static_cast<std::size_t>(token) >= impl_->model_config.vocab_size) {
    return Result<std::vector<float>>::Failure(
        {ErrorCode::kOutOfBounds, "CUDA input token is outside the vocabulary"});
  }
  if (position != impl_->next_positions[0] || position >= impl_->maximum_sequence_length) {
    return Result<std::vector<float>>::Failure(
        {ErrorCode::kOutOfBounds, "CUDA positions must be contiguous and inside the KV Cache"});
  }
  const auto wall_start = std::chrono::steady_clock::now();
  cudaError_t cuda_status = cudaEventRecord(impl_->compute_start, impl_->stream);
  if (cuda_status != cudaSuccess) {
    return Result<std::vector<float>>::Failure(
        CudaFailure("cudaEventRecord compute start", cuda_status).error());
  }
  const std::size_t hidden = impl_->model_config.hidden_size;
  const std::size_t kv_dimension = impl_->model_config.kv_dim();
  if (impl_->embeddings.dtype == CudaWeightDType::kFloat16) {
    cuda_status = cudaMemcpyAsync(
        impl_->hidden, impl_->embeddings.values + static_cast<std::size_t>(token) * hidden,
        hidden * sizeof(__half), cudaMemcpyDeviceToDevice, impl_->stream);
    if (cuda_status != cudaSuccess) {
      return Result<std::vector<float>>::Failure(
          CudaFailure("CUDA embedding copy", cuda_status).error());
    }
  } else {
    auto dequantized =
        LaunchInt8EmbeddingRow(impl_->embeddings.quantized_values, impl_->embeddings.scales,
                               static_cast<std::size_t>(token), impl_->model_config.vocab_size,
                               hidden, impl_->hidden, impl_->stream);
    if (!dequantized.ok()) {
      return Result<std::vector<float>>::Failure(dequantized.error());
    }
  }

  for (std::size_t layer = 0; layer < impl_->model_config.num_hidden_layers; ++layer) {
    const LayerWeights& weights = impl_->layers[layer];
    auto operation =
        LaunchRmsNorm(impl_->hidden, weights.input_norm.values, impl_->model_config.rms_norm_eps,
                      impl_->normalized, 1, hidden, impl_->stream);
    if (operation.ok()) {
      operation = impl_->Linear(weights.query, impl_->normalized, impl_->query, hidden, hidden, 1);
    }
    if (operation.ok()) {
      operation =
          impl_->Linear(weights.key, impl_->normalized, impl_->key, kv_dimension, hidden, 1);
    }
    if (operation.ok()) {
      operation =
          impl_->Linear(weights.value, impl_->normalized, impl_->value, kv_dimension, hidden, 1);
    }
    if (operation.ok()) {
      operation =
          LaunchRope(impl_->query, impl_->key, impl_->model_config.num_attention_heads,
                     impl_->model_config.num_key_value_heads, impl_->model_config.head_dim(),
                     position, impl_->model_config.rope_theta, impl_->stream);
    }
    const std::size_t cache_layer_offset = layer * impl_->maximum_sequence_length * kv_dimension;
    if (operation.ok()) {
      operation = LaunchKvUpdate(impl_->key, impl_->value, impl_->key_cache + cache_layer_offset,
                                 impl_->value_cache + cache_layer_offset, position,
                                 impl_->maximum_sequence_length, kv_dimension, impl_->stream);
    }
    if (operation.ok()) {
      operation = LaunchAttentionDecode(impl_->query, impl_->key_cache + cache_layer_offset,
                                        impl_->value_cache + cache_layer_offset, impl_->attention,
                                        position + 1, impl_->model_config.num_attention_heads,
                                        impl_->model_config.num_key_value_heads,
                                        impl_->model_config.head_dim(), impl_->stream);
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.attention_output, impl_->attention, impl_->projected,
                                hidden, hidden, 1);
    }
    if (operation.ok()) {
      operation = LaunchAddInPlace(impl_->hidden, impl_->projected, hidden, impl_->stream);
    }
    if (operation.ok()) {
      operation = LaunchRmsNorm(impl_->hidden, weights.post_attention_norm.values,
                                impl_->model_config.rms_norm_eps, impl_->normalized, 1, hidden,
                                impl_->stream);
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.gate, impl_->normalized, impl_->gate,
                                impl_->model_config.intermediate_size, hidden, 1);
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.up, impl_->normalized, impl_->up,
                                impl_->model_config.intermediate_size, hidden, 1);
    }
    if (operation.ok()) {
      operation = LaunchSiluMultiply(impl_->gate, impl_->up, impl_->activated,
                                     impl_->model_config.intermediate_size, impl_->stream);
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.down, impl_->activated, impl_->projected, hidden,
                                impl_->model_config.intermediate_size, 1);
    }
    if (operation.ok()) {
      operation = LaunchAddInPlace(impl_->hidden, impl_->projected, hidden, impl_->stream);
    }
    if (!operation.ok()) {
      return Result<std::vector<float>>::Failure(operation.error());
    }
  }

  auto operation =
      LaunchRmsNorm(impl_->hidden, impl_->final_norm_weight.values,
                    impl_->model_config.rms_norm_eps, impl_->normalized, 1, hidden, impl_->stream);
  if (operation.ok()) {
    operation = impl_->Linear(impl_->lm_head, impl_->normalized, impl_->logits,
                              impl_->model_config.vocab_size, hidden, 1);
  }
  if (!operation.ok()) {
    return Result<std::vector<float>>::Failure(operation.error());
  }
  cuda_status = cudaEventRecord(impl_->compute_stop, impl_->stream);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpyAsync(impl_->host_logits.data(), impl_->logits,
                                  impl_->host_logits.size() * sizeof(__half),
                                  cudaMemcpyDeviceToHost, impl_->stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaEventRecord(impl_->transfer_stop, impl_->stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaStreamSynchronize(impl_->stream);
  }
  if (cuda_status != cudaSuccess) {
    return Result<std::vector<float>>::Failure(
        CudaFailure("CUDA logits transfer", cuda_status).error());
  }

  std::vector<float> host_logits(impl_->host_logits.size());
  for (std::size_t index = 0; index < host_logits.size(); ++index) {
    host_logits[index] = __half2float(impl_->host_logits[index]);
    if (!std::isfinite(host_logits[index])) {
      return Result<std::vector<float>>::Failure(
          {ErrorCode::kNumerical, "CUDA logits contain NaN or Inf"});
    }
  }
  if (timings != nullptr) {
    cuda_status =
        cudaEventElapsedTime(&timings->compute_ms, impl_->compute_start, impl_->compute_stop);
    if (cuda_status == cudaSuccess) {
      cuda_status =
          cudaEventElapsedTime(&timings->transfer_ms, impl_->compute_stop, impl_->transfer_stop);
    }
    if (cuda_status != cudaSuccess) {
      return Result<std::vector<float>>::Failure(
          CudaFailure("CUDA event timing", cuda_status).error());
    }
    timings->wall_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - wall_start)
            .count();
  }
  ++impl_->next_positions[0];
  return Result<std::vector<float>>::Success(std::move(host_logits));
}

Result<std::vector<std::vector<float>>> TinyLlamaCuda::ForwardBatch(
    const std::span<const int> tokens, const std::span<const std::size_t> positions,
    const std::span<const std::size_t> kv_slots, CudaStepTimings* timings) {
  if (impl_ == nullptr) {
    return Result<std::vector<std::vector<float>>>::Failure(
        {ErrorCode::kCuda, "CUDA model is not initialized"});
  }
  const std::size_t batch = tokens.size();
  if (batch == 0 || batch > impl_->maximum_batch_size || positions.size() != batch ||
      kv_slots.size() != batch) {
    return Result<std::vector<std::vector<float>>>::Failure(
        {ErrorCode::kInvalidArgument, "CUDA batch dimensions are invalid"});
  }
  for (std::size_t row = 0; row < batch; ++row) {
    if (tokens[row] < 0 ||
        static_cast<std::size_t>(tokens[row]) >= impl_->model_config.vocab_size) {
      return Result<std::vector<std::vector<float>>>::Failure(
          {ErrorCode::kOutOfBounds, "CUDA batch token is outside the vocabulary"});
    }
    if (kv_slots[row] >= impl_->slot_count || positions[row] >= impl_->maximum_sequence_length ||
        positions[row] != impl_->next_positions[kv_slots[row]]) {
      return Result<std::vector<std::vector<float>>>::Failure(
          {ErrorCode::kOutOfBounds, "CUDA batch position is not contiguous in its KV slot"});
    }
    for (std::size_t previous = 0; previous < row; ++previous) {
      if (kv_slots[previous] == kv_slots[row]) {
        return Result<std::vector<std::vector<float>>>::Failure(
            {ErrorCode::kInvalidArgument, "CUDA batch cannot update one KV slot twice"});
      }
    }
  }

  const auto wall_start = std::chrono::steady_clock::now();
  cudaError_t cuda_status = cudaEventRecord(impl_->compute_start, impl_->stream);
  if (cuda_status != cudaSuccess) {
    return Result<std::vector<std::vector<float>>>::Failure(
        CudaFailure("cudaEventRecord batch compute start", cuda_status).error());
  }
  const std::size_t hidden = impl_->model_config.hidden_size;
  const std::size_t kv_dimension = impl_->model_config.kv_dim();
  for (std::size_t row = 0; row < batch; ++row) {
    if (impl_->embeddings.dtype == CudaWeightDType::kFloat16) {
      cuda_status =
          cudaMemcpyAsync(impl_->hidden + row * hidden,
                          impl_->embeddings.values + static_cast<std::size_t>(tokens[row]) * hidden,
                          hidden * sizeof(__half), cudaMemcpyDeviceToDevice, impl_->stream);
      if (cuda_status != cudaSuccess) {
        return Result<std::vector<std::vector<float>>>::Failure(
            CudaFailure("CUDA batch embedding copy", cuda_status).error());
      }
    } else {
      auto dequantized = LaunchInt8EmbeddingRow(
          impl_->embeddings.quantized_values, impl_->embeddings.scales,
          static_cast<std::size_t>(tokens[row]), impl_->model_config.vocab_size, hidden,
          impl_->hidden + row * hidden, impl_->stream);
      if (!dequantized.ok()) {
        return Result<std::vector<std::vector<float>>>::Failure(dequantized.error());
      }
    }
  }

  for (std::size_t layer = 0; layer < impl_->model_config.num_hidden_layers; ++layer) {
    const LayerWeights& weights = impl_->layers[layer];
    auto operation =
        LaunchRmsNorm(impl_->hidden, weights.input_norm.values, impl_->model_config.rms_norm_eps,
                      impl_->normalized, batch, hidden, impl_->stream);
    if (operation.ok()) {
      operation =
          impl_->Linear(weights.query, impl_->normalized, impl_->query, hidden, hidden, batch);
    }
    if (operation.ok()) {
      operation =
          impl_->Linear(weights.key, impl_->normalized, impl_->key, kv_dimension, hidden, batch);
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.value, impl_->normalized, impl_->value, kv_dimension,
                                hidden, batch);
    }
    for (std::size_t row = 0; operation.ok() && row < batch; ++row) {
      operation =
          LaunchRope(impl_->query + row * hidden, impl_->key + row * kv_dimension,
                     impl_->model_config.num_attention_heads,
                     impl_->model_config.num_key_value_heads, impl_->model_config.head_dim(),
                     positions[row], impl_->model_config.rope_theta, impl_->stream);
      const std::size_t cache_offset =
          (kv_slots[row] * impl_->model_config.num_hidden_layers + layer) *
          impl_->maximum_sequence_length * kv_dimension;
      if (operation.ok()) {
        operation = LaunchKvUpdate(
            impl_->key + row * kv_dimension, impl_->value + row * kv_dimension,
            impl_->key_cache + cache_offset, impl_->value_cache + cache_offset, positions[row],
            impl_->maximum_sequence_length, kv_dimension, impl_->stream);
      }
      if (operation.ok()) {
        operation = LaunchAttentionDecode(
            impl_->query + row * hidden, impl_->key_cache + cache_offset,
            impl_->value_cache + cache_offset, impl_->attention + row * hidden, positions[row] + 1,
            impl_->model_config.num_attention_heads, impl_->model_config.num_key_value_heads,
            impl_->model_config.head_dim(), impl_->stream);
      }
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.attention_output, impl_->attention, impl_->projected,
                                hidden, hidden, batch);
    }
    if (operation.ok()) {
      operation = LaunchAddInPlace(impl_->hidden, impl_->projected, batch * hidden, impl_->stream);
    }
    if (operation.ok()) {
      operation = LaunchRmsNorm(impl_->hidden, weights.post_attention_norm.values,
                                impl_->model_config.rms_norm_eps, impl_->normalized, batch, hidden,
                                impl_->stream);
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.gate, impl_->normalized, impl_->gate,
                                impl_->model_config.intermediate_size, hidden, batch);
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.up, impl_->normalized, impl_->up,
                                impl_->model_config.intermediate_size, hidden, batch);
    }
    if (operation.ok()) {
      operation = LaunchSiluMultiply(impl_->gate, impl_->up, impl_->activated,
                                     batch * impl_->model_config.intermediate_size, impl_->stream);
    }
    if (operation.ok()) {
      operation = impl_->Linear(weights.down, impl_->activated, impl_->projected, hidden,
                                impl_->model_config.intermediate_size, batch);
    }
    if (operation.ok()) {
      operation = LaunchAddInPlace(impl_->hidden, impl_->projected, batch * hidden, impl_->stream);
    }
    if (!operation.ok()) {
      return Result<std::vector<std::vector<float>>>::Failure(operation.error());
    }
  }

  auto operation = LaunchRmsNorm(impl_->hidden, impl_->final_norm_weight.values,
                                 impl_->model_config.rms_norm_eps, impl_->normalized, batch, hidden,
                                 impl_->stream);
  if (operation.ok()) {
    operation = impl_->Linear(impl_->lm_head, impl_->normalized, impl_->logits,
                              impl_->model_config.vocab_size, hidden, batch);
  }
  if (!operation.ok()) {
    return Result<std::vector<std::vector<float>>>::Failure(operation.error());
  }
  cuda_status = cudaEventRecord(impl_->compute_stop, impl_->stream);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemcpyAsync(impl_->host_logits.data(), impl_->logits,
                                  batch * impl_->model_config.vocab_size * sizeof(__half),
                                  cudaMemcpyDeviceToHost, impl_->stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaEventRecord(impl_->transfer_stop, impl_->stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaStreamSynchronize(impl_->stream);
  }
  if (cuda_status != cudaSuccess) {
    return Result<std::vector<std::vector<float>>>::Failure(
        CudaFailure("CUDA batch logits transfer", cuda_status).error());
  }

  std::vector<std::vector<float>> output(batch, std::vector<float>(impl_->model_config.vocab_size));
  for (std::size_t row = 0; row < batch; ++row) {
    for (std::size_t index = 0; index < impl_->model_config.vocab_size; ++index) {
      output[row][index] =
          __half2float(impl_->host_logits[row * impl_->model_config.vocab_size + index]);
      if (!std::isfinite(output[row][index])) {
        return Result<std::vector<std::vector<float>>>::Failure(
            {ErrorCode::kNumerical, "CUDA batch logits contain NaN or Inf"});
      }
    }
  }
  if (timings != nullptr) {
    cuda_status =
        cudaEventElapsedTime(&timings->compute_ms, impl_->compute_start, impl_->compute_stop);
    if (cuda_status == cudaSuccess) {
      cuda_status =
          cudaEventElapsedTime(&timings->transfer_ms, impl_->compute_stop, impl_->transfer_stop);
    }
    if (cuda_status != cudaSuccess) {
      return Result<std::vector<std::vector<float>>>::Failure(
          CudaFailure("CUDA batch event timing", cuda_status).error());
    }
    timings->wall_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - wall_start)
            .count();
  }
  for (const std::size_t slot : kv_slots) {
    ++impl_->next_positions[slot];
  }
  return Result<std::vector<std::vector<float>>>::Success(std::move(output));
}

Result<void> TinyLlamaCuda::Reset() {
  if (impl_ == nullptr) {
    return Result<void>::Failure({ErrorCode::kCuda, "CUDA model is not initialized"});
  }
  return impl_->Reset();
}

Result<void> TinyLlamaCuda::ResetSlot(const std::size_t kv_slot) {
  if (impl_ == nullptr) {
    return Result<void>::Failure({ErrorCode::kCuda, "CUDA model is not initialized"});
  }
  return impl_->ResetSlot(kv_slot);
}

std::size_t TinyLlamaCuda::max_sequence_length() const {
  return impl_ == nullptr ? 0 : impl_->maximum_sequence_length;
}

std::size_t TinyLlamaCuda::max_batch_size() const {
  return impl_ == nullptr ? 0 : impl_->maximum_batch_size;
}

std::size_t TinyLlamaCuda::kv_slot_count() const {
  return impl_ == nullptr ? 0 : impl_->slot_count;
}

std::size_t TinyLlamaCuda::device_allocation_count() const {
  return impl_ == nullptr ? 0 : impl_->weight_store.allocation_count() + impl_->allocations.size();
}

std::size_t TinyLlamaCuda::kv_cache_bytes() const {
  if (impl_ == nullptr) {
    return 0;
  }
  return 2 * impl_->slot_count * impl_->model_config.num_hidden_layers *
         impl_->maximum_sequence_length * impl_->model_config.kv_dim() * sizeof(__half);
}

const ModelConfig& TinyLlamaCuda::config() const {
  if (impl_ == nullptr) {
    std::terminate();
  }
  return impl_->model_config;
}

}  // namespace tlie::cuda
