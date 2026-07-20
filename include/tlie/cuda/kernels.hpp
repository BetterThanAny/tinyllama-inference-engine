#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "tlie/result.hpp"

namespace tlie::cuda {

Result<void> LaunchRmsNorm(const __half* input, const __half* weight, float epsilon, __half* output,
                           std::size_t rows, std::size_t columns, cudaStream_t stream);
Result<void> LaunchRope(__half* query, __half* key, std::size_t query_heads,
                        std::size_t key_value_heads, std::size_t head_dim, std::size_t position,
                        float theta, cudaStream_t stream);
Result<void> LaunchSoftmax(__half* values, std::size_t rows, std::size_t columns,
                           cudaStream_t stream);
Result<void> LaunchKvUpdate(const __half* key, const __half* value, __half* key_cache,
                            __half* value_cache, std::size_t position,
                            std::size_t max_sequence_length, std::size_t kv_dimension,
                            cudaStream_t stream);
Result<void> LaunchAttentionDecode(const __half* query, const __half* key_cache,
                                   const __half* value_cache, __half* output,
                                   std::size_t sequence_length, std::size_t query_heads,
                                   std::size_t key_value_heads, std::size_t head_dim,
                                   cudaStream_t stream);
Result<void> LaunchSiluMultiply(const __half* gate, const __half* up, __half* output,
                                std::size_t elements, cudaStream_t stream);
Result<void> LaunchAddInPlace(__half* destination, const __half* source, std::size_t elements,
                              cudaStream_t stream);
Result<void> GemmFp16RowMajor(cublasHandle_t handle, const __half* weight, const __half* input,
                              __half* output, std::size_t output_rows, std::size_t input_columns,
                              std::size_t batch, cudaStream_t stream);
Result<void> LaunchInt8WeightOnlyGemv(const std::int8_t* weight, const float* scales,
                                      const __half* input, __half* output, std::size_t output_rows,
                                      std::size_t input_columns, cudaStream_t stream);
Result<void> LaunchInt8EmbeddingRow(const std::int8_t* weight, const float* scales, std::size_t row,
                                    std::size_t rows, std::size_t columns, __half* output,
                                    cudaStream_t stream);
Result<bool> CheckFinite(const __half* values, std::size_t elements, cudaStream_t stream);
Result<void> Synchronize(cudaStream_t stream);

}  // namespace tlie::cuda
