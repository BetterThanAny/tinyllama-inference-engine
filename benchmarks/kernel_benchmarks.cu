#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "tlie/cuda/kernels.hpp"

namespace {

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t elements) : elements_(elements) {
    if (cudaMalloc(&pointer_, elements * sizeof(T)) != cudaSuccess) {
      pointer_ = nullptr;
    }
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] T* get() const { return static_cast<T*>(pointer_); }
  [[nodiscard]] bool valid() const { return pointer_ != nullptr; }
  [[nodiscard]] std::size_t size() const { return elements_; }

 private:
  void* pointer_{nullptr};
  std::size_t elements_{0};
};

struct EventPair {
  cudaEvent_t start{nullptr};
  cudaEvent_t stop{nullptr};

  EventPair() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  ~EventPair() {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
  }
};

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << nlohmann::json({{"error", message}}).dump() << '\n';
  std::exit(EXIT_FAILURE);
}

void RequireCuda(const cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    Fail(operation + ": " + cudaGetErrorString(status));
  }
}

void Require(const tlie::Result<void>& result) {
  if (!result.ok()) {
    Fail(result.error().message);
  }
}

void Fill(DeviceBuffer<__half>& buffer, const float scale = 1.0F) {
  if (!buffer.valid()) {
    Fail("CUDA benchmark allocation failed");
  }
  std::vector<__half> values(buffer.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    const float value = std::sin(static_cast<float>(index % 1009) * 0.013F) * scale;
    values[index] = __float2half_rn(value);
  }
  RequireCuda(cudaMemcpy(buffer.get(), values.data(), values.size() * sizeof(__half),
                         cudaMemcpyHostToDevice),
              "benchmark input upload");
}

void FillInt8(DeviceBuffer<std::int8_t>& buffer) {
  if (!buffer.valid()) {
    Fail("CUDA benchmark allocation failed");
  }
  std::vector<std::int8_t> values(buffer.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<std::int8_t>(static_cast<int>(index % 127U) - 63);
  }
  RequireCuda(cudaMemcpy(buffer.get(), values.data(), values.size() * sizeof(std::int8_t),
                         cudaMemcpyHostToDevice),
              "benchmark INT8 input upload");
}

void FillFloat(DeviceBuffer<float>& buffer, const float value) {
  if (!buffer.valid()) {
    Fail("CUDA benchmark allocation failed");
  }
  const std::vector<float> values(buffer.size(), value);
  RequireCuda(cudaMemcpy(buffer.get(), values.data(), values.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "benchmark float input upload");
}

int ParseCount(const char* text, const std::string& name, const bool allow_zero) {
  int value = 0;
  const std::string input(text);
  const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc{} || end != input.data() + input.size() || value < (allow_zero ? 0 : 1)) {
    Fail(name + (allow_zero ? " must be a non-negative integer" : " must be a positive integer"));
  }
  return value;
}

nlohmann::json Measure(const std::string& name, const int warmup, const int samples,
                       cudaStream_t stream, const std::function<tlie::Result<void>()>& operation,
                       const nlohmann::json& shape) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    Require(operation());
  }
  Require(tlie::cuda::Synchronize(stream));
  EventPair events;
  if (events.start == nullptr || events.stop == nullptr) {
    Fail("unable to create CUDA benchmark events");
  }
  std::vector<float> milliseconds;
  milliseconds.reserve(static_cast<std::size_t>(samples));
  for (int sample = 0; sample < samples; ++sample) {
    RequireCuda(cudaEventRecord(events.start, stream), "record benchmark start");
    Require(operation());
    RequireCuda(cudaEventRecord(events.stop, stream), "record benchmark stop");
    RequireCuda(cudaEventSynchronize(events.stop), "synchronize benchmark stop");
    float elapsed = 0.0F;
    RequireCuda(cudaEventElapsedTime(&elapsed, events.start, events.stop),
                "calculate benchmark duration");
    milliseconds.push_back(elapsed);
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  const auto percentile = [&](const double fraction) {
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(milliseconds.size())) - 1.0);
    return milliseconds[std::min(index, milliseconds.size() - 1)];
  };
  return {{"name", name},
          {"shape", shape},
          {"warmup", warmup},
          {"samples", samples},
          {"timing", "cuda_event"},
          {"median_ms", percentile(0.5)},
          {"p95_ms", percentile(0.95)}};
}

}  // namespace

int main(const int argc, char** argv) {
  int warmup = 10;
  int samples = 50;
  if (argc != 1) {
    if (argc != 5 || std::string(argv[1]) != "--warmup" || std::string(argv[3]) != "--samples") {
      Fail("usage: kernel_benchmarks [--warmup N --samples N]");
    }
    warmup = ParseCount(argv[2], "warmup", true);
    samples = ParseCount(argv[4], "samples", false);
  }
  cudaStream_t stream = nullptr;
  RequireCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
  cublasHandle_t cublas = nullptr;
  if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
    Fail("create cuBLAS handle failed");
  }

  nlohmann::json results = nlohmann::json::array();
  DeviceBuffer<__half> hidden(2048);
  DeviceBuffer<__half> norm_weight(2048);
  DeviceBuffer<__half> normalized(2048);
  Fill(hidden);
  Fill(norm_weight, 0.5F);
  results.push_back(Measure("rms_norm", warmup, samples, stream,
                            [&] {
                              return tlie::cuda::LaunchRmsNorm(hidden.get(), norm_weight.get(),
                                                               1.0e-5F, normalized.get(), 1, 2048,
                                                               stream);
                            },
                            {{"rows", 1}, {"columns", 2048}}));

  DeviceBuffer<__half> query(32 * 64);
  DeviceBuffer<__half> key(4 * 64);
  Fill(query);
  Fill(key);
  results.push_back(Measure(
      "rope", warmup, samples, stream,
      [&] {
        return tlie::cuda::LaunchRope(query.get(), key.get(), 32, 4, 64, 4095, 10000.0F, stream);
      },
      {{"query_heads", 32}, {"key_value_heads", 4}, {"head_dim", 64}, {"position", 4095}}));

  DeviceBuffer<__half> softmax(4096);
  Fill(softmax);
  results.push_back(
      Measure("softmax", warmup, samples, stream,
              [&] { return tlie::cuda::LaunchSoftmax(softmax.get(), 1, 4096, stream); },
              {{"rows", 1}, {"columns", 4096}}));

  DeviceBuffer<__half> cache_key(4096 * 4 * 64);
  DeviceBuffer<__half> cache_value(4096 * 4 * 64);
  Fill(cache_key);
  Fill(cache_value);
  results.push_back(
      Measure("kv_update", warmup, samples, stream,
              [&] {
                return tlie::cuda::LaunchKvUpdate(key.get(), key.get(), cache_key.get(),
                                                  cache_value.get(), 4095, 4096, 4 * 64, stream);
              },
              {{"position", 4095}, {"max_sequence_length", 4096}, {"kv_dimension", 256}}));

  DeviceBuffer<__half> attention_output(32 * 64);
  for (const std::size_t context : {128U, 512U, 2048U, 4096U}) {
    results.push_back(Measure(
        "attention_decode", warmup, samples, stream,
        [&] {
          return tlie::cuda::LaunchAttentionDecode(query.get(), cache_key.get(), cache_value.get(),
                                                   attention_output.get(), context, 32, 4, 64,
                                                   stream);
        },
        {{"context", context}, {"query_heads", 32}, {"key_value_heads", 4}, {"head_dim", 64}}));
  }

  DeviceBuffer<__half> gate(5632);
  DeviceBuffer<__half> up(5632);
  DeviceBuffer<__half> activated(5632);
  Fill(gate);
  Fill(up);
  results.push_back(Measure("silu_multiply", warmup, samples, stream,
                            [&] {
                              return tlie::cuda::LaunchSiluMultiply(gate.get(), up.get(),
                                                                    activated.get(), 5632, stream);
                            },
                            {{"elements", 5632}}));

  DeviceBuffer<__half> residual_destination(2048);
  DeviceBuffer<__half> residual_source(2048);
  Fill(residual_destination, 0.01F);
  Fill(residual_source, 0.001F);
  results.push_back(Measure("residual_add", warmup, samples, stream,
                            [&] {
                              return tlie::cuda::LaunchAddInPlace(
                                  residual_destination.get(), residual_source.get(), 2048, stream);
                            },
                            {{"elements", 2048}}));

  constexpr std::size_t output_rows = 5632;
  constexpr std::size_t input_columns = 2048;
  constexpr std::size_t maximum_batch = 8;
  DeviceBuffer<__half> matrix(output_rows * input_columns);
  DeviceBuffer<__half> gemm_input(maximum_batch * input_columns);
  DeviceBuffer<__half> gemm_output(maximum_batch * output_rows);
  Fill(matrix, 0.01F);
  Fill(gemm_input, 0.1F);
  for (const std::size_t batch : {1U, 8U}) {
    results.push_back(Measure(
        batch == 1 ? "gemv" : "gemm", warmup, samples, stream,
        [&] {
          return tlie::cuda::GemmFp16RowMajor(cublas, matrix.get(), gemm_input.get(),
                                              gemm_output.get(), output_rows, input_columns, batch,
                                              stream);
        },
        {{"output_rows", output_rows}, {"input_columns", input_columns}, {"batch", batch}}));
  }

  DeviceBuffer<std::int8_t> int8_matrix(output_rows * input_columns);
  DeviceBuffer<float> int8_scales(output_rows);
  FillInt8(int8_matrix);
  FillFloat(int8_scales, 0.01F);
  results.push_back(
      Measure("int8_weight_only_gemv", warmup, samples, stream,
              [&] {
                return tlie::cuda::LaunchInt8WeightOnlyGemv(int8_matrix.get(), int8_scales.get(),
                                                            gemm_input.get(), gemm_output.get(),
                                                            output_rows, input_columns, stream);
              },
              {{"output_rows", output_rows}, {"input_columns", input_columns}, {"batch", 1}}));

  constexpr std::size_t vocabulary_size = 32000;
  DeviceBuffer<std::int8_t> int8_embedding(vocabulary_size * input_columns);
  DeviceBuffer<float> embedding_scales(vocabulary_size);
  DeviceBuffer<__half> embedding_output(input_columns);
  FillInt8(int8_embedding);
  FillFloat(embedding_scales, 0.01F);
  results.push_back(Measure(
      "int8_embedding_row", warmup, samples, stream,
      [&] {
        return tlie::cuda::LaunchInt8EmbeddingRow(int8_embedding.get(), embedding_scales.get(),
                                                  3681, vocabulary_size, input_columns,
                                                  embedding_output.get(), stream);
      },
      {{"vocabulary", vocabulary_size}, {"columns", input_columns}, {"token", 3681}}));

  int device = 0;
  RequireCuda(cudaGetDevice(&device), "get CUDA device");
  cudaDeviceProp properties{};
  RequireCuda(cudaGetDeviceProperties(&properties, device), "get CUDA properties");
  std::cout << nlohmann::json({{"schema_version", 1},
                               {"gpu", properties.name},
                               {"timing", "cuda_event"},
                               {"benchmarks", results}})
                   .dump()
            << '\n';
  if (cublasDestroy(cublas) != CUBLAS_STATUS_SUCCESS) {
    Fail("destroy cuBLAS handle failed");
  }
  RequireCuda(cudaStreamDestroy(stream), "destroy stream");
  return EXIT_SUCCESS;
}
