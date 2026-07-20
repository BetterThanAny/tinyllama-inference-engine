#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace tlie {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
 public:
  Sha256();
  void Update(std::span<const std::byte> data);
  Sha256Digest Final();

 private:
  void Transform(const std::uint8_t* block);

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> buffer_{};
  std::uint64_t bit_count_{0};
  std::size_t buffer_size_{0};
  bool finalized_{false};
};

Sha256Digest ComputeSha256(std::span<const std::byte> data);
std::string Sha256Hex(const Sha256Digest& digest);
bool ParseSha256(std::string_view text, Sha256Digest* digest);

}  // namespace tlie
