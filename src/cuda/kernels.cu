#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "tlie/cuda/kernels.hpp"

namespace tlie::cuda {
namespace {

constexpr int kThreads = 256;

Result<void> Invalid(const std::string& message) {
  return Result<void>::Failure({ErrorCode::kInvalidArgument, message});
}

Result<void> CudaFailure(const std::string& operation, const cudaError_t error) {
  return Result<void>::Failure(
      {ErrorCode::kCuda, operation + " failed: " + cudaGetErrorString(error)});
}

Result<void> CublasFailure(const std::string& operation, const cublasStatus_t status) {
  return Result<void>::Failure(
      {ErrorCode::kCuda, operation + " failed with cuBLAS status " + std::to_string(status)});
}

Result<void> CheckLaunch(const std::string& operation) {
  const cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return CudaFailure(operation, error);
  }
  return Result<void>::Success();
}

__device__ float BlockReduceSum(float value, float* shared) {
  const unsigned thread = threadIdx.x;
  shared[thread] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1U) {
    if (thread < stride) {
      shared[thread] += shared[thread + stride];
    }
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__device__ float BlockReduceMax(float value, float* shared) {
  const unsigned thread = threadIdx.x;
  shared[thread] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1U) {
    if (thread < stride) {
      shared[thread] = fmaxf(shared[thread], shared[thread + stride]);
    }
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__global__ void RmsNormKernel(const __half* input, const __half* weight, const float epsilon,
                              __half* output, const std::size_t columns) {
  extern __shared__ float reduction[];
  const std::size_t row = blockIdx.x;
  const std::size_t base = row * columns;
  float square_sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = __half2float(input[base + column]);
    square_sum += value * value;
  }
  const float total = BlockReduceSum(square_sum, reduction);
  const float inverse_rms = rsqrtf(total / static_cast<float>(columns) + epsilon);
  for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const float normalized =
        __half2float(input[base + column]) * inverse_rms * __half2float(weight[column]);
    output[base + column] = __float2half_rn(normalized);
  }
}

__global__ void RopeKernel(__half* query, __half* key, const std::size_t query_heads,
                           const std::size_t key_value_heads, const std::size_t head_dim,
                           const std::size_t position, const float theta) {
  const std::size_t head = blockIdx.x;
  const std::size_t dimension = threadIdx.x;
  const std::size_t half_dimension = head_dim / 2;
  if (dimension >= half_dimension) {
    return;
  }
  const float frequency =
      powf(theta, -2.0F * static_cast<float>(dimension) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(position) * frequency;
  float sine = 0.0F;
  float cosine = 0.0F;
  sincosf(angle, &sine, &cosine);
  if (head < query_heads) {
    const std::size_t base = head * head_dim;
    const float first = __half2float(query[base + dimension]);
    const float second = __half2float(query[base + dimension + half_dimension]);
    query[base + dimension] = __float2half_rn(first * cosine - second * sine);
    query[base + dimension + half_dimension] = __float2half_rn(second * cosine + first * sine);
  }
  if (head < key_value_heads) {
    const std::size_t base = head * head_dim;
    const float first = __half2float(key[base + dimension]);
    const float second = __half2float(key[base + dimension + half_dimension]);
    key[base + dimension] = __float2half_rn(first * cosine - second * sine);
    key[base + dimension + half_dimension] = __float2half_rn(second * cosine + first * sine);
  }
}

__global__ void SoftmaxKernel(__half* values, const std::size_t columns) {
  extern __shared__ float reduction[];
  const std::size_t base = blockIdx.x * columns;
  float local_max = -CUDART_INF_F;
  for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
    local_max = fmaxf(local_max, __half2float(values[base + column]));
  }
  const float maximum = BlockReduceMax(local_max, reduction);
  float local_sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const float exponent = expf(__half2float(values[base + column]) - maximum);
    values[base + column] = __float2half_rn(exponent);
    local_sum += exponent;
  }
  const float sum = BlockReduceSum(local_sum, reduction);
  for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
    values[base + column] = __float2half_rn(__half2float(values[base + column]) / sum);
  }
}

__global__ void KvUpdateKernel(const __half* key, const __half* value, __half* key_cache,
                               __half* value_cache, const std::size_t position,
                               const std::size_t kv_dimension) {
  for (std::size_t index = threadIdx.x; index < kv_dimension; index += blockDim.x) {
    const std::size_t destination = position * kv_dimension + index;
    key_cache[destination] = key[index];
    value_cache[destination] = value[index];
  }
}

__global__ void AttentionDecodeKernel(const __half* query, const __half* key_cache,
                                      const __half* value_cache, __half* output,
                                      const std::size_t sequence_length,
                                      const std::size_t query_heads,
                                      const std::size_t key_value_heads,
                                      const std::size_t head_dim) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduction = scores + sequence_length;
  const std::size_t query_head = blockIdx.x;
  if (query_head >= query_heads) {
    return;
  }
  const std::size_t groups = query_heads / key_value_heads;
  const std::size_t kv_head = query_head / groups;
  const std::size_t kv_stride = key_value_heads * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float local_max = -CUDART_INF_F;
  for (std::size_t position = threadIdx.x; position < sequence_length; position += blockDim.x) {
    float dot = 0.0F;
    const std::size_t query_base = query_head * head_dim;
    const std::size_t key_base = position * kv_stride + kv_head * head_dim;
    for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
      dot += __half2float(query[query_base + dimension]) *
             __half2float(key_cache[key_base + dimension]);
    }
    scores[position] = dot * scale;
    local_max = fmaxf(local_max, scores[position]);
  }
  const float maximum = BlockReduceMax(local_max, reduction);
  float local_sum = 0.0F;
  for (std::size_t position = threadIdx.x; position < sequence_length; position += blockDim.x) {
    scores[position] = expf(scores[position] - maximum);
    local_sum += scores[position];
  }
  const float sum = BlockReduceSum(local_sum, reduction);
  for (std::size_t position = threadIdx.x; position < sequence_length; position += blockDim.x) {
    scores[position] /= sum;
  }
  __syncthreads();
  if (threadIdx.x < head_dim) {
    const std::size_t dimension = threadIdx.x;
    float weighted = 0.0F;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t value_base = position * kv_stride + kv_head * head_dim;
      weighted += scores[position] * __half2float(value_cache[value_base + dimension]);
    }
    output[query_head * head_dim + dimension] = __float2half_rn(weighted);
  }
}

__global__ void SiluMultiplyKernel(const __half* gate, const __half* up, __half* output,
                                   const std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    const float gate_value = __half2float(gate[index]);
    const float activated = gate_value / (1.0F + expf(-gate_value));
    output[index] = __float2half_rn(activated * __half2float(up[index]));
  }
}

__global__ void AddInPlaceKernel(__half* destination, const __half* source,
                                 const std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    destination[index] =
        __float2half_rn(__half2float(destination[index]) + __half2float(source[index]));
  }
}

__global__ void Int8WeightOnlyGemvKernel(const std::int8_t* weight, const float* scales,
                                         const __half* input, __half* output,
                                         const std::size_t input_columns) {
  extern __shared__ float reduction[];
  const std::size_t row = blockIdx.x;
  const std::size_t base = row * input_columns;
  float partial = 0.0F;
  for (std::size_t column = threadIdx.x; column < input_columns; column += blockDim.x) {
    partial += static_cast<float>(weight[base + column]) * __half2float(input[column]);
  }
  const float sum = BlockReduceSum(partial, reduction);
  if (threadIdx.x == 0) {
    output[row] = __float2half_rn(sum * scales[row]);
  }
}

__global__ void Int8EmbeddingRowKernel(const std::int8_t* weight, const float* scales,
                                       const std::size_t row, const std::size_t columns,
                                       __half* output) {
  const float scale = scales[row];
  const std::size_t base = row * columns;
  for (std::size_t column = blockIdx.x * blockDim.x + threadIdx.x; column < columns;
       column += blockDim.x * gridDim.x) {
    output[column] = __float2half_rn(static_cast<float>(weight[base + column]) * scale);
  }
}

__global__ void AllFiniteKernel(const __half* values, const std::size_t elements, int* finite) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    if (!isfinite(__half2float(values[index]))) {
      atomicExch(finite, 0);
    }
  }
}

}  // namespace

Result<void> LaunchRmsNorm(const __half* input, const __half* weight, const float epsilon,
                           __half* output, const std::size_t rows, const std::size_t columns,
                           cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows == 0 || columns == 0 ||
      !(epsilon > 0.0F)) {
    return Invalid("RMSNorm CUDA arguments are invalid");
  }
  RmsNormKernel<<<static_cast<unsigned>(rows), kThreads, kThreads * sizeof(float), stream>>>(
      input, weight, epsilon, output, columns);
  return CheckLaunch("RMSNorm kernel launch");
}

Result<void> LaunchRope(__half* query, __half* key, const std::size_t query_heads,
                        const std::size_t key_value_heads, const std::size_t head_dim,
                        const std::size_t position, const float theta, cudaStream_t stream) {
  if (query == nullptr || key == nullptr || query_heads == 0 || key_value_heads == 0 ||
      head_dim == 0 || head_dim % 2 != 0 || head_dim / 2 > 1024 || !(theta > 0.0F)) {
    return Invalid("RoPE CUDA arguments are invalid");
  }
  const auto blocks = static_cast<unsigned>(std::max(query_heads, key_value_heads));
  RopeKernel<<<blocks, static_cast<unsigned>(head_dim / 2), 0, stream>>>(
      query, key, query_heads, key_value_heads, head_dim, position, theta);
  return CheckLaunch("RoPE kernel launch");
}

Result<void> LaunchSoftmax(__half* values, const std::size_t rows, const std::size_t columns,
                           cudaStream_t stream) {
  if (values == nullptr || rows == 0 || columns == 0) {
    return Invalid("Softmax CUDA arguments are invalid");
  }
  SoftmaxKernel<<<static_cast<unsigned>(rows), kThreads, kThreads * sizeof(float), stream>>>(
      values, columns);
  return CheckLaunch("Softmax kernel launch");
}

Result<void> LaunchKvUpdate(const __half* key, const __half* value, __half* key_cache,
                            __half* value_cache, const std::size_t position,
                            const std::size_t max_sequence_length, const std::size_t kv_dimension,
                            cudaStream_t stream) {
  if (key == nullptr || value == nullptr || key_cache == nullptr || value_cache == nullptr ||
      kv_dimension == 0 || position >= max_sequence_length) {
    return Invalid("KV Cache CUDA update is out of bounds or has invalid pointers");
  }
  KvUpdateKernel<<<1, kThreads, 0, stream>>>(key, value, key_cache, value_cache, position,
                                             kv_dimension);
  return CheckLaunch("KV Cache update kernel launch");
}

Result<void> LaunchAttentionDecode(const __half* query, const __half* key_cache,
                                   const __half* value_cache, __half* output,
                                   const std::size_t sequence_length, const std::size_t query_heads,
                                   const std::size_t key_value_heads, const std::size_t head_dim,
                                   cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
      sequence_length == 0 || query_heads == 0 || key_value_heads == 0 || head_dim == 0 ||
      head_dim > kThreads || query_heads % key_value_heads != 0) {
    return Invalid("Attention CUDA arguments are invalid");
  }
  const std::size_t shared_bytes = (sequence_length + kThreads) * sizeof(float);
  AttentionDecodeKernel<<<static_cast<unsigned>(query_heads), kThreads, shared_bytes, stream>>>(
      query, key_cache, value_cache, output, sequence_length, query_heads, key_value_heads,
      head_dim);
  return CheckLaunch("attention decode kernel launch");
}

Result<void> LaunchSiluMultiply(const __half* gate, const __half* up, __half* output,
                                const std::size_t elements, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || elements == 0) {
    return Invalid("SiLU CUDA arguments are invalid");
  }
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>((elements + kThreads - 1) / kThreads, 4096));
  SiluMultiplyKernel<<<blocks, kThreads, 0, stream>>>(gate, up, output, elements);
  return CheckLaunch("SiLU multiply kernel launch");
}

Result<void> LaunchAddInPlace(__half* destination, const __half* source, const std::size_t elements,
                              cudaStream_t stream) {
  if (destination == nullptr || source == nullptr || elements == 0) {
    return Invalid("residual add CUDA arguments are invalid");
  }
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>((elements + kThreads - 1) / kThreads, 4096));
  AddInPlaceKernel<<<blocks, kThreads, 0, stream>>>(destination, source, elements);
  return CheckLaunch("residual add kernel launch");
}

Result<void> GemmFp16RowMajor(cublasHandle_t handle, const __half* weight, const __half* input,
                              __half* output, const std::size_t output_rows,
                              const std::size_t input_columns, const std::size_t batch,
                              cudaStream_t stream) {
  if (handle == nullptr || weight == nullptr || input == nullptr || output == nullptr ||
      output_rows == 0 || input_columns == 0 || batch == 0 ||
      output_rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      input_columns > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      batch > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Invalid("FP16 GEMM arguments are invalid");
  }
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasFailure("cublasSetStream", status);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(output_rows), static_cast<int>(batch),
      static_cast<int>(input_columns), &alpha, weight, CUDA_R_16F, static_cast<int>(input_columns),
      input, CUDA_R_16F, static_cast<int>(input_columns), &beta, output, CUDA_R_16F,
      static_cast<int>(output_rows), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasFailure("cublasGemmEx", status);
  }
  return Result<void>::Success();
}

Result<void> LaunchInt8WeightOnlyGemv(const std::int8_t* weight, const float* scales,
                                      const __half* input, __half* output,
                                      const std::size_t output_rows,
                                      const std::size_t input_columns, cudaStream_t stream) {
  if (weight == nullptr || scales == nullptr || input == nullptr || output == nullptr ||
      output_rows == 0 || input_columns == 0 ||
      output_rows > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("INT8 weight-only GEMV arguments are invalid");
  }
  Int8WeightOnlyGemvKernel<<<static_cast<unsigned>(output_rows), kThreads, kThreads * sizeof(float),
                             stream>>>(weight, scales, input, output, input_columns);
  return CheckLaunch("INT8 weight-only GEMV kernel launch");
}

Result<void> LaunchInt8EmbeddingRow(const std::int8_t* weight, const float* scales,
                                    const std::size_t row, const std::size_t rows,
                                    const std::size_t columns, __half* output,
                                    cudaStream_t stream) {
  if (weight == nullptr || scales == nullptr || output == nullptr || rows == 0 || columns == 0 ||
      row >= rows) {
    return Invalid("INT8 embedding row arguments are invalid");
  }
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>((columns + kThreads - 1) / kThreads, 4096));
  Int8EmbeddingRowKernel<<<blocks, kThreads, 0, stream>>>(weight, scales, row, columns, output);
  return CheckLaunch("INT8 embedding row kernel launch");
}

Result<bool> CheckFinite(const __half* values, const std::size_t elements, cudaStream_t stream) {
  if (values == nullptr || elements == 0) {
    return Result<bool>::Failure(
        {ErrorCode::kInvalidArgument, "finite check CUDA arguments are invalid"});
  }
  int* device_finite = nullptr;
  cudaError_t error = cudaMalloc(&device_finite, sizeof(int));
  if (error != cudaSuccess) {
    return Result<bool>::Failure({ErrorCode::kCuda, "finite-check allocation failed: " +
                                                        std::string(cudaGetErrorString(error))});
  }
  const int initial = 1;
  error = cudaMemcpyAsync(device_finite, &initial, sizeof(initial), cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) {
    const auto blocks =
        static_cast<unsigned>(std::min<std::size_t>((elements + kThreads - 1) / kThreads, 4096));
    AllFiniteKernel<<<blocks, kThreads, 0, stream>>>(values, elements, device_finite);
    error = cudaPeekAtLastError();
  }
  int host_finite = 0;
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(&host_finite, device_finite, sizeof(host_finite),
                            cudaMemcpyDeviceToHost, stream);
  }
  if (error == cudaSuccess) {
    error = cudaStreamSynchronize(stream);
  }
  const cudaError_t free_error = cudaFree(device_finite);
  if (error != cudaSuccess) {
    return Result<bool>::Failure(
        {ErrorCode::kCuda, "finite check failed: " + std::string(cudaGetErrorString(error))});
  }
  if (free_error != cudaSuccess) {
    return Result<bool>::Failure(
        {ErrorCode::kCuda,
         "finite-check free failed: " + std::string(cudaGetErrorString(free_error))});
  }
  return Result<bool>::Success(host_finite != 0);
}

Result<void> Synchronize(cudaStream_t stream) {
  const cudaError_t error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    return CudaFailure("cudaStreamSynchronize", error);
  }
  return Result<void>::Success();
}

}  // namespace tlie::cuda
