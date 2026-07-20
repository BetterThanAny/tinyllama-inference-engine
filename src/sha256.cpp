#include "tlie/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace tlie {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U};

constexpr std::uint32_t RotateRight(const std::uint32_t value, const unsigned bits) {
  return (value >> bits) | (value << (32U - bits));
}

std::uint8_t HexValue(const char character) {
  if (character >= '0' && character <= '9') {
    return static_cast<std::uint8_t>(character - '0');
  }
  if (character >= 'a' && character <= 'f') {
    return static_cast<std::uint8_t>(character - 'a' + 10);
  }
  if (character >= 'A' && character <= 'F') {
    return static_cast<std::uint8_t>(character - 'A' + 10);
  }
  return 0xffU;
}

}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void Sha256::Transform(const std::uint8_t* block) {
  std::array<std::uint32_t, 64> schedule{};
  for (std::size_t i = 0; i < 16; ++i) {
    const std::size_t offset = i * 4;
    schedule[i] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                  (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
                  (static_cast<std::uint32_t>(block[offset + 2]) << 8U) |
                  static_cast<std::uint32_t>(block[offset + 3]);
  }
  for (std::size_t i = 16; i < schedule.size(); ++i) {
    const std::uint32_t s0 = RotateRight(schedule[i - 15], 7) ^ RotateRight(schedule[i - 15], 18) ^
                             (schedule[i - 15] >> 3U);
    const std::uint32_t s1 = RotateRight(schedule[i - 2], 17) ^ RotateRight(schedule[i - 2], 19) ^
                             (schedule[i - 2] >> 10U);
    schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
  }

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];

  for (std::size_t i = 0; i < schedule.size(); ++i) {
    const std::uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
    const std::uint32_t choose = (e & f) ^ (~e & g);
    const std::uint32_t temp1 = h + sum1 + choose + kRoundConstants[i] + schedule[i];
    const std::uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void Sha256::Update(const std::span<const std::byte> data) {
  if (finalized_) {
    throw std::logic_error("cannot update a finalized SHA-256 context");
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data.data());
  std::size_t remaining = data.size();
  bit_count_ += static_cast<std::uint64_t>(remaining) * 8U;
  while (remaining > 0) {
    const std::size_t chunk = std::min(remaining, buffer_.size() - buffer_size_);
    std::memcpy(buffer_.data() + buffer_size_, bytes, chunk);
    buffer_size_ += chunk;
    bytes += chunk;
    remaining -= chunk;
    if (buffer_size_ == buffer_.size()) {
      Transform(buffer_.data());
      buffer_size_ = 0;
    }
  }
}

Sha256Digest Sha256::Final() {
  if (finalized_) {
    throw std::logic_error("SHA-256 context finalized twice");
  }
  finalized_ = true;
  buffer_[buffer_size_++] = 0x80U;
  if (buffer_size_ > 56) {
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.end(), 0U);
    Transform(buffer_.data());
    buffer_size_ = 0;
  }
  std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.begin() + 56, 0U);
  for (std::size_t i = 0; i < 8; ++i) {
    buffer_[63 - i] = static_cast<std::uint8_t>(bit_count_ >> (i * 8U));
  }
  Transform(buffer_.data());

  Sha256Digest digest{};
  for (std::size_t i = 0; i < state_.size(); ++i) {
    digest[i * 4] = static_cast<std::uint8_t>(state_[i] >> 24U);
    digest[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16U);
    digest[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8U);
    digest[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
  }
  return digest;
}

Sha256Digest ComputeSha256(const std::span<const std::byte> data) {
  Sha256 hash;
  hash.Update(data);
  return hash.Final();
}

std::string Sha256Hex(const Sha256Digest& digest) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.resize(digest.size() * 2);
  for (std::size_t i = 0; i < digest.size(); ++i) {
    result[i * 2] = kHex[digest[i] >> 4U];
    result[i * 2 + 1] = kHex[digest[i] & 0x0fU];
  }
  return result;
}

bool ParseSha256(const std::string_view text, Sha256Digest* digest) {
  if (digest == nullptr || text.size() != digest->size() * 2) {
    return false;
  }
  for (std::size_t i = 0; i < digest->size(); ++i) {
    const std::uint8_t high = HexValue(text[i * 2]);
    const std::uint8_t low = HexValue(text[i * 2 + 1]);
    if (high == 0xffU || low == 0xffU) {
      return false;
    }
    (*digest)[i] = static_cast<std::uint8_t>((high << 4U) | low);
  }
  return true;
}

}  // namespace tlie
