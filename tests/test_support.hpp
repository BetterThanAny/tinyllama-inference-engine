#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <string_view>

namespace tlie::test {

class Context {
 public:
  void Check(const bool condition, const std::string_view expression, const std::string_view file,
             const int line) {
    ++checks_;
    if (!condition) {
      ++failures_;
      std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
    }
  }

  void CheckNear(const float actual, const float expected, const float tolerance,
                 const std::string_view expression, const std::string_view file, const int line) {
    ++checks_;
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      ++failures_;
      std::cerr << file << ':' << line << ": CHECK_NEAR failed: " << expression
                << ", actual=" << actual << ", expected=" << expected << ", tolerance=" << tolerance
                << '\n';
    }
  }

  [[nodiscard]] int Finish(const std::string_view suite) const {
    if (failures_ == 0) {
      std::cout << suite << ": " << checks_ << " checks passed\n";
      return 0;
    }
    std::cerr << suite << ": " << failures_ << " of " << checks_ << " checks failed\n";
    return 1;
  }

 private:
  int checks_{0};
  int failures_{0};
};

}  // namespace tlie::test

#define TLIE_CHECK(context, expression) \
  (context).Check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
#define TLIE_CHECK_NEAR(context, actual, expected, tolerance)                                \
  (context).CheckNear((actual), (expected), (tolerance), #actual " ~= " #expected, __FILE__, \
                      __LINE__)
