#include "tlie/error.hpp"

namespace tlie {

const char* ErrorCodeName(const ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kIo:
      return "io_error";
    case ErrorCode::kInvalidFormat:
      return "invalid_format";
    case ErrorCode::kChecksumMismatch:
      return "checksum_mismatch";
    case ErrorCode::kInvalidConfig:
      return "invalid_config";
    case ErrorCode::kInvalidShape:
      return "invalid_shape";
    case ErrorCode::kInvalidDtype:
      return "invalid_dtype";
    case ErrorCode::kOutOfMemory:
      return "out_of_memory";
    case ErrorCode::kOutOfBounds:
      return "out_of_bounds";
    case ErrorCode::kTokenizer:
      return "tokenizer_error";
    case ErrorCode::kNumerical:
      return "numerical_error";
    case ErrorCode::kInvalidArgument:
      return "invalid_argument";
    case ErrorCode::kCuda:
      return "cuda_error";
  }
  return "unknown_error";
}

}  // namespace tlie
