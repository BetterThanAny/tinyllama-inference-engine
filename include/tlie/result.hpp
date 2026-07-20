#pragma once

#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

#include "tlie/error.hpp"

namespace tlie {

template <typename T>
class Result {
 public:
  static Result Success(T value) { return Result(std::move(value)); }
  static Result Failure(Error error) { return Result(std::move(error)); }

  [[nodiscard]] bool ok() const noexcept { return std::holds_alternative<T>(value_); }
  [[nodiscard]] const T& value() const& {
    if (!ok()) {
      throw std::logic_error("attempted to read a failed Result");
    }
    return std::get<T>(value_);
  }
  [[nodiscard]] T& value() & {
    if (!ok()) {
      throw std::logic_error("attempted to read a failed Result");
    }
    return std::get<T>(value_);
  }
  [[nodiscard]] T&& value() && {
    if (!ok()) {
      throw std::logic_error("attempted to read a failed Result");
    }
    return std::get<T>(std::move(value_));
  }
  [[nodiscard]] const Error& error() const {
    if (ok()) {
      throw std::logic_error("attempted to read error from a successful Result");
    }
    return std::get<Error>(value_);
  }

 private:
  explicit Result(T value) : value_(std::move(value)) {}
  explicit Result(Error error) : value_(std::move(error)) {}

  std::variant<T, Error> value_;
};

template <>
class Result<void> {
 public:
  static Result Success() { return Result(); }
  static Result Failure(Error error) { return Result(std::move(error)); }

  [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }
  [[nodiscard]] const Error& error() const {
    if (ok()) {
      throw std::logic_error("attempted to read error from a successful Result");
    }
    return *error_;
  }

 private:
  Result() = default;
  explicit Result(Error error) : error_(std::move(error)) {}

  std::optional<Error> error_;
};

}  // namespace tlie
