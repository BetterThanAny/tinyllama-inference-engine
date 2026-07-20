#pragma once

#include <string>

namespace tlie {

enum class ErrorCode {
  kIo,
  kInvalidFormat,
  kChecksumMismatch,
  kInvalidConfig,
  kInvalidShape,
  kInvalidDtype,
  kOutOfMemory,
  kOutOfBounds,
  kTokenizer,
  kNumerical,
  kInvalidArgument,
  kCuda,
};

struct Error {
  ErrorCode code;
  std::string message;
};

const char* ErrorCodeName(ErrorCode code) noexcept;

}  // namespace tlie
