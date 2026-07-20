#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "../test_support.hpp"
#include "tlie/cuda/kernels.hpp"
#include "tlie/operators.hpp"

namespace {

using tlie::test::Context;

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

std::vector<__half> ToHalf(const std::span<const float> values) {
  std::vector<__half> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(),
                 [](const float value) { return __float2half_rn(value); });
  return result;
}

std::vector<float> ToFloat(const std::span<const __half> values) {
  std::vector<float> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(),
                 [](const __half value) { return __half2float(value); });
  return result;
}

template <typename T, std::size_t Extent>
bool Upload(DeviceBuffer<T>& destination, const std::span<const T, Extent> source) {
  return destination.valid() && destination.size() == source.size() &&
         cudaMemcpy(destination.get(), source.data(), source.size_bytes(),
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

template <typename T, std::size_t Extent>
bool Upload(DeviceBuffer<T>& destination, const std::span<T, Extent> source) {
  return Upload(destination, std::span<const T>(source));
}

template <typename T>
bool Download(const DeviceBuffer<T>& source, const std::span<T> destination) {
  return source.valid() && source.size() == destination.size() &&
         cudaMemcpy(destination.data(), source.get(), destination.size_bytes(),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

void CheckVector(Context& context, const std::span<const float> actual,
                 const std::span<const float> expected, const float tolerance) {
  TLIE_CHECK(context, actual.size() == expected.size());
  if (actual.size() != expected.size()) {
    return;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    TLIE_CHECK_NEAR(context, actual[index], expected[index], tolerance);
  }
}

bool Ready(Context& context, const std::initializer_list<bool> conditions) {
  const bool ready =
      std::all_of(conditions.begin(), conditions.end(), [](const bool value) { return value; });
  TLIE_CHECK(context, ready);
  return ready;
}

void TestRmsNorm(Context& context, cudaStream_t stream) {
  constexpr std::size_t rows = 2;
  constexpr std::size_t columns = 8;
  const std::vector<float> input = {-1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 2.5F,
                                    0.2F,  0.4F,  0.6F, 0.8F, 1.0F, 1.2F, 1.4F, 1.6F};
  const std::vector<float> weight = {0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 1.75F, 2.0F, 2.25F};
  const auto half_input = ToHalf(input);
  const auto half_weight = ToHalf(weight);
  DeviceBuffer<__half> device_input(input.size());
  DeviceBuffer<__half> device_weight(weight.size());
  DeviceBuffer<__half> device_output(input.size());
  if (!Ready(context, {Upload(device_input, std::span(half_input)),
                       Upload(device_weight, std::span(half_weight)), device_output.valid()})) {
    return;
  }
  const auto launched = tlie::cuda::LaunchRmsNorm(device_input.get(), device_weight.get(), 1.0e-5F,
                                                  device_output.get(), rows, columns, stream);
  TLIE_CHECK(context, launched.ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  std::vector<__half> half_output(input.size());
  TLIE_CHECK(context, Download(device_output, std::span(half_output)));
  const auto actual = ToFloat(half_output);
  std::vector<float> expected(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    TLIE_CHECK(context, tlie::RmsNorm(std::span(input).subspan(row * columns, columns), weight,
                                      1.0e-5F, std::span(expected).subspan(row * columns, columns))
                            .ok());
  }
  CheckVector(context, actual, expected, 2.0e-3F);
}

void TestRope(Context& context, cudaStream_t stream) {
  constexpr std::size_t query_heads = 4;
  constexpr std::size_t key_value_heads = 2;
  constexpr std::size_t head_dim = 8;
  std::vector<float> query(query_heads * head_dim);
  std::vector<float> key(key_value_heads * head_dim);
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = static_cast<float>(index) / 17.0F - 0.7F;
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] = static_cast<float>(index) / 11.0F - 0.4F;
  }
  auto expected_query = query;
  auto expected_key = key;
  TLIE_CHECK(context, tlie::ApplyRope(expected_query, expected_key, query_heads, key_value_heads,
                                      head_dim, 4095, 10000.0F)
                          .ok());
  auto half_query = ToHalf(query);
  auto half_key = ToHalf(key);
  DeviceBuffer<__half> device_query(query.size());
  DeviceBuffer<__half> device_key(key.size());
  if (!Ready(context, {Upload(device_query, std::span(half_query)),
                       Upload(device_key, std::span(half_key))})) {
    return;
  }
  TLIE_CHECK(context, tlie::cuda::LaunchRope(device_query.get(), device_key.get(), query_heads,
                                             key_value_heads, head_dim, 4095, 10000.0F, stream)
                          .ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  TLIE_CHECK(context, Download(device_query, std::span(half_query)));
  TLIE_CHECK(context, Download(device_key, std::span(half_key)));
  CheckVector(context, ToFloat(half_query), expected_query, 3.0e-3F);
  CheckVector(context, ToFloat(half_key), expected_key, 3.0e-3F);
}

void TestSoftmax(Context& context, cudaStream_t stream) {
  constexpr std::size_t rows = 2;
  constexpr std::size_t columns = 7;
  const std::vector<float> input = {-20.0F, -2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 20.0F,
                                    100.0F, 99.0F, 98.0F, 3.0F, 2.0F, 1.0F, -100.0F};
  std::vector<float> expected = input;
  for (std::size_t row = 0; row < rows; ++row) {
    TLIE_CHECK(context,
               tlie::SoftmaxInPlace(std::span(expected).subspan(row * columns, columns)).ok());
  }
  auto half_values = ToHalf(input);
  DeviceBuffer<__half> device_values(input.size());
  if (!Ready(context, {Upload(device_values, std::span(half_values))})) {
    return;
  }
  TLIE_CHECK(context, tlie::cuda::LaunchSoftmax(device_values.get(), rows, columns, stream).ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  TLIE_CHECK(context, Download(device_values, std::span(half_values)));
  const auto actual = ToFloat(half_values);
  CheckVector(context, actual, expected, 1.0e-3F);
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (const float value : std::span(actual).subspan(row * columns, columns)) {
      sum += value;
    }
    TLIE_CHECK_NEAR(context, sum, 1.0F, 1.0e-3F);
  }
}

void TestKvCache(Context& context, cudaStream_t stream) {
  constexpr std::size_t sequence = 5;
  constexpr std::size_t dimension = 4;
  const auto half_key = ToHalf(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F});
  const auto half_value = ToHalf(std::vector<float>{5.0F, 6.0F, 7.0F, 8.0F});
  DeviceBuffer<__half> device_key(dimension);
  DeviceBuffer<__half> device_value(dimension);
  DeviceBuffer<__half> key_cache(sequence * dimension);
  DeviceBuffer<__half> value_cache(sequence * dimension);
  if (!Ready(context,
             {Upload(device_key, std::span(half_key)), Upload(device_value, std::span(half_value)),
              key_cache.valid(), value_cache.valid()})) {
    return;
  }
  TLIE_CHECK(context,
             cudaMemset(key_cache.get(), 0, sequence * dimension * sizeof(__half)) == cudaSuccess);
  TLIE_CHECK(context, cudaMemset(value_cache.get(), 0, sequence * dimension * sizeof(__half)) ==
                          cudaSuccess);
  TLIE_CHECK(context, tlie::cuda::LaunchKvUpdate(device_key.get(), device_value.get(),
                                                 key_cache.get(), value_cache.get(), sequence - 1,
                                                 sequence, dimension, stream)
                          .ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  std::vector<__half> cache(sequence * dimension);
  TLIE_CHECK(context, Download(key_cache, std::span(cache)));
  const auto cache_float = ToFloat(cache);
  for (std::size_t index = 0; index < dimension; ++index) {
    TLIE_CHECK_NEAR(context, cache_float[(sequence - 1) * dimension + index],
                    static_cast<float>(index + 1), 0.0F);
  }
  const auto out_of_bounds =
      tlie::cuda::LaunchKvUpdate(device_key.get(), device_value.get(), key_cache.get(),
                                 value_cache.get(), sequence, sequence, dimension, stream);
  TLIE_CHECK(context, !out_of_bounds.ok());
  TLIE_CHECK(context, out_of_bounds.error().code == tlie::ErrorCode::kInvalidArgument);
}

void TestAttentionAt(Context& context, cudaStream_t stream, const std::size_t sequence_length) {
  constexpr std::size_t query_heads = 4;
  constexpr std::size_t key_value_heads = 2;
  constexpr std::size_t head_dim = 8;
  std::vector<float> query(query_heads * head_dim);
  std::vector<float> keys(sequence_length * key_value_heads * head_dim);
  std::vector<float> values(keys.size());
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = std::sin(static_cast<float>(index) * 0.17F);
  }
  for (std::size_t index = 0; index < keys.size(); ++index) {
    keys[index] = std::cos(static_cast<float>(index) * 0.013F);
    values[index] = std::sin(static_cast<float>(index) * 0.007F);
  }
  std::vector<float> expected(query.size());
  TLIE_CHECK(context, tlie::AttentionReference(query, keys, values, sequence_length, query_heads,
                                               key_value_heads, head_dim, expected)
                          .ok());
  const auto half_query = ToHalf(query);
  const auto half_keys = ToHalf(keys);
  const auto half_values = ToHalf(values);
  DeviceBuffer<__half> device_query(query.size());
  DeviceBuffer<__half> device_keys(keys.size());
  DeviceBuffer<__half> device_values(values.size());
  DeviceBuffer<__half> device_output(query.size());
  if (!Ready(context, {Upload(device_query, std::span(half_query)),
                       Upload(device_keys, std::span(half_keys)),
                       Upload(device_values, std::span(half_values)), device_output.valid()})) {
    return;
  }
  TLIE_CHECK(context, tlie::cuda::LaunchAttentionDecode(device_query.get(), device_keys.get(),
                                                        device_values.get(), device_output.get(),
                                                        sequence_length, query_heads,
                                                        key_value_heads, head_dim, stream)
                          .ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  std::vector<__half> half_output(query.size());
  TLIE_CHECK(context, Download(device_output, std::span(half_output)));
  CheckVector(context, ToFloat(half_output), expected, 6.0e-3F);
}

void TestPointwise(Context& context, cudaStream_t stream) {
  const std::vector<float> gate = {-3.0F, -1.0F, 0.0F, 1.0F, 3.0F};
  const std::vector<float> up = {0.5F, 1.0F, 1.5F, 2.0F, 2.5F};
  std::vector<float> expected(gate.size());
  TLIE_CHECK(context, tlie::SiluMultiply(gate, up, expected).ok());
  const auto half_gate = ToHalf(gate);
  auto half_up = ToHalf(up);
  DeviceBuffer<__half> device_gate(gate.size());
  DeviceBuffer<__half> device_up(up.size());
  DeviceBuffer<__half> device_output(gate.size());
  if (!Ready(context, {Upload(device_gate, std::span(half_gate)),
                       Upload(device_up, std::span(half_up)), device_output.valid()})) {
    return;
  }
  TLIE_CHECK(context, tlie::cuda::LaunchSiluMultiply(device_gate.get(), device_up.get(),
                                                     device_output.get(), gate.size(), stream)
                          .ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  std::vector<__half> half_output(gate.size());
  TLIE_CHECK(context, Download(device_output, std::span(half_output)));
  CheckVector(context, ToFloat(half_output), expected, 2.0e-3F);

  TLIE_CHECK(
      context,
      tlie::cuda::LaunchAddInPlace(device_up.get(), device_gate.get(), gate.size(), stream).ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  TLIE_CHECK(context, Download(device_up, std::span(half_up)));
  auto sum = ToFloat(half_up);
  for (std::size_t index = 0; index < sum.size(); ++index) {
    TLIE_CHECK_NEAR(context, sum[index], up[index] + gate[index], 2.0e-3F);
  }
}

void TestGemm(Context& context, cudaStream_t stream) {
  constexpr std::size_t output_rows = 3;
  constexpr std::size_t input_columns = 4;
  constexpr std::size_t batch = 2;
  const std::vector<float> weight = {1.0F, 2.0F, 3.0F,  4.0F,  -1.0F, 0.5F,
                                     2.0F, 1.0F, 0.25F, -0.5F, 1.5F,  2.0F};
  const std::vector<float> input = {1.0F, 0.5F, -1.0F, 2.0F, -2.0F, 1.0F, 0.25F, 0.5F};
  std::vector<float> expected(batch * output_rows, 0.0F);
  for (std::size_t sample = 0; sample < batch; ++sample) {
    for (std::size_t row = 0; row < output_rows; ++row) {
      for (std::size_t column = 0; column < input_columns; ++column) {
        expected[sample * output_rows + row] +=
            weight[row * input_columns + column] * input[sample * input_columns + column];
      }
    }
  }
  const auto half_weight = ToHalf(weight);
  const auto half_input = ToHalf(input);
  DeviceBuffer<__half> device_weight(weight.size());
  DeviceBuffer<__half> device_input(input.size());
  DeviceBuffer<__half> device_output(expected.size());
  cublasHandle_t handle = nullptr;
  const bool handle_created = cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS;
  if (!Ready(context, {Upload(device_weight, std::span(half_weight)),
                       Upload(device_input, std::span(half_input)), device_output.valid(),
                       handle_created})) {
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
    return;
  }
  TLIE_CHECK(context, tlie::cuda::GemmFp16RowMajor(handle, device_weight.get(), device_input.get(),
                                                   device_output.get(), output_rows, input_columns,
                                                   batch, stream)
                          .ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  std::vector<__half> half_output(expected.size());
  TLIE_CHECK(context, Download(device_output, std::span(half_output)));
  CheckVector(context, ToFloat(half_output), expected, 3.0e-3F);
  TLIE_CHECK(context, cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}

void TestInt8WeightOnly(Context& context, cudaStream_t stream) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t columns = 4;
  const std::vector<std::int8_t> weight = {127, -64, 32, 0, -5, 10, 20, -40, 1, 2, 3, 4};
  const std::vector<float> scales = {0.02F, 0.1F, 0.5F};
  const std::vector<float> input = {1.0F, -0.5F, 2.0F, 0.25F};
  const auto half_input = ToHalf(input);
  DeviceBuffer<std::int8_t> device_weight(weight.size());
  DeviceBuffer<float> device_scales(scales.size());
  DeviceBuffer<__half> device_input(input.size());
  DeviceBuffer<__half> device_output(rows);
  if (!Ready(context,
             {Upload(device_weight, std::span(weight)), Upload(device_scales, std::span(scales)),
              Upload(device_input, std::span(half_input)), device_output.valid()})) {
    return;
  }
  TLIE_CHECK(context, tlie::cuda::LaunchInt8WeightOnlyGemv(device_weight.get(), device_scales.get(),
                                                           device_input.get(), device_output.get(),
                                                           rows, columns, stream)
                          .ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  std::vector<__half> half_output(rows);
  TLIE_CHECK(context, Download(device_output, std::span(half_output)));
  const auto actual = ToFloat(half_output);
  for (std::size_t row = 0; row < rows; ++row) {
    float expected = 0.0F;
    for (std::size_t column = 0; column < columns; ++column) {
      expected += static_cast<float>(weight[row * columns + column]) * scales[row] * input[column];
    }
    TLIE_CHECK_NEAR(context, actual[row], expected, 2.0e-2F);
  }

  std::vector<__half> embedding(columns);
  DeviceBuffer<__half> embedding_output(columns);
  TLIE_CHECK(context,
             tlie::cuda::LaunchInt8EmbeddingRow(device_weight.get(), device_scales.get(), 1, rows,
                                                columns, embedding_output.get(), stream)
                 .ok());
  TLIE_CHECK(context, tlie::cuda::Synchronize(stream).ok());
  TLIE_CHECK(context, Download(embedding_output, std::span(embedding)));
  const auto embedding_float = ToFloat(embedding);
  for (std::size_t column = 0; column < columns; ++column) {
    TLIE_CHECK_NEAR(context, embedding_float[column],
                    static_cast<float>(weight[columns + column]) * scales[1], 1.0e-3F);
  }
  const auto invalid =
      tlie::cuda::LaunchInt8EmbeddingRow(device_weight.get(), device_scales.get(), rows, rows,
                                         columns, embedding_output.get(), stream);
  TLIE_CHECK(context, !invalid.ok());
  TLIE_CHECK(context, invalid.error().code == tlie::ErrorCode::kInvalidArgument);
}

void TestFiniteAndFailures(Context& context, cudaStream_t stream) {
  auto values = ToHalf(std::vector<float>{0.0F, 1.0F, -2.0F});
  DeviceBuffer<__half> device_values(values.size());
  if (!Ready(context, {Upload(device_values, std::span(values))})) {
    return;
  }
  auto finite = tlie::cuda::CheckFinite(device_values.get(), values.size(), stream);
  TLIE_CHECK(context, finite.ok());
  TLIE_CHECK(context, finite.ok() && finite.value());
  values[1] = __float2half(std::numeric_limits<float>::quiet_NaN());
  TLIE_CHECK(context, Upload(device_values, std::span(values)));
  finite = tlie::cuda::CheckFinite(device_values.get(), values.size(), stream);
  TLIE_CHECK(context, finite.ok());
  TLIE_CHECK(context, finite.ok() && !finite.value());
  const auto invalid = tlie::cuda::LaunchSoftmax(nullptr, 1, 1, stream);
  TLIE_CHECK(context, !invalid.ok());
  TLIE_CHECK(context, invalid.error().code == tlie::ErrorCode::kInvalidArgument);
}

}  // namespace

int main() {
  Context context;
  int device = -1;
  const auto get_device_status = cudaGetDevice(&device);
  if (get_device_status != cudaSuccess) {
    std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(get_device_status) << " ("
              << static_cast<int>(get_device_status) << ")\n";
  }
  TLIE_CHECK(context, get_device_status == cudaSuccess);
  cudaDeviceProp properties{};
  cudaError_t get_properties_status = cudaErrorInvalidDevice;
  if (device >= 0) {
    get_properties_status = cudaGetDeviceProperties(&properties, device);
    if (get_properties_status != cudaSuccess) {
      std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(get_properties_status)
                << " (" << static_cast<int>(get_properties_status) << ")\n";
    }
  }
  TLIE_CHECK(context, device >= 0 && get_properties_status == cudaSuccess);
  if (device < 0) {
    return context.Finish("tlie_cuda_kernel_tests");
  }
  std::cout << "CUDA device: " << properties.name << " (SM " << properties.major << '.'
            << properties.minor << ")\n";
  cudaStream_t stream = nullptr;
  TLIE_CHECK(context, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  if (stream == nullptr) {
    return context.Finish("tlie_cuda_kernel_tests");
  }
  TestRmsNorm(context, stream);
  TestRope(context, stream);
  TestSoftmax(context, stream);
  TestKvCache(context, stream);
  TestAttentionAt(context, stream, 128);
  TestAttentionAt(context, stream, 4096);
  TestPointwise(context, stream);
  TestGemm(context, stream);
  TestInt8WeightOnly(context, stream);
  TestFiniteAndFailures(context, stream);
  TLIE_CHECK(context, cudaStreamDestroy(stream) == cudaSuccess);
  return context.Finish("tlie_cuda_kernel_tests");
}
