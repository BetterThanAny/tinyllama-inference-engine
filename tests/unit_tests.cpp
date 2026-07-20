#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

#include "test_support.hpp"
#include "tlie/kv_cache.hpp"
#include "tlie/model_config.hpp"
#include "tlie/operators.hpp"
#include "tlie/sha256.hpp"
#include "tlie/weight_store.hpp"

namespace {

using tlie::test::Context;

template <typename T>
void WriteValue(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void WriteFixture(const std::filesystem::path& path, const bool corrupt_checksum = false,
                  const bool wrong_byte_count = false, const float first_value = 1.0F) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  const std::array<char, 8> magic = {'T', 'L', 'I', 'E', 'W', 'G', 'T', '\0'};
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  const std::uint32_t version = 1;
  const std::uint32_t count = 1;
  WriteValue(output, version);
  WriteValue(output, count);
  std::array<std::uint8_t, 32> source{};
  source.fill(0x42U);
  output.write(reinterpret_cast<const char*>(source.data()),
               static_cast<std::streamsize>(source.size()));

  const std::string name = "tensor";
  const std::uint16_t name_length = static_cast<std::uint16_t>(name.size());
  const std::uint8_t dtype = 1;
  const std::uint8_t rank = 2;
  const std::array<float, 4> values = {first_value, 2.0F, 3.0F, 4.0F};
  const std::uint64_t byte_count = wrong_byte_count ? 12 : sizeof(values);
  const auto bytes = std::as_bytes(std::span(values));
  auto checksum = tlie::ComputeSha256(bytes);
  if (corrupt_checksum) {
    checksum[0] ^= 0xffU;
  }
  WriteValue(output, name_length);
  WriteValue(output, dtype);
  WriteValue(output, rank);
  WriteValue(output, byte_count);
  output.write(reinterpret_cast<const char*>(checksum.data()),
               static_cast<std::streamsize>(checksum.size()));
  const std::uint64_t first = 2;
  const std::uint64_t second = 2;
  WriteValue(output, first);
  WriteValue(output, second);
  output.write(name.data(), static_cast<std::streamsize>(name.size()));
  const auto offset = static_cast<std::size_t>(output.tellp());
  const std::size_t padding = (64 - offset % 64) % 64;
  const std::array<char, 64> zero{};
  output.write(zero.data(), static_cast<std::streamsize>(padding));
  output.write(reinterpret_cast<const char*>(values.data()), sizeof(values));
}

std::string FileSha256(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto size = input.tellg();
  input.seekg(0);
  std::vector<std::byte> contents(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(contents.data()), size);
  return tlie::Sha256Hex(tlie::ComputeSha256(contents));
}

void TestSha256(Context& context) {
  const std::string input = "abc";
  const auto digest = tlie::ComputeSha256(std::as_bytes(std::span(input.data(), input.size())));
  TLIE_CHECK(context, tlie::Sha256Hex(digest) ==
                          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  tlie::Sha256Digest parsed{};
  TLIE_CHECK(context, tlie::ParseSha256(tlie::Sha256Hex(digest), &parsed));
  TLIE_CHECK(context, parsed == digest);
  TLIE_CHECK(context, !tlie::ParseSha256("not-a-checksum", &parsed));
}

void TestConfig(Context& context, const std::filesystem::path& temporary) {
  const auto model_config = std::filesystem::path(TLIE_MODEL_DIR) / "config.json";
  const auto loaded = tlie::ModelConfig::Load(model_config);
  TLIE_CHECK(context, loaded.ok());
  if (!loaded.ok()) {
    return;
  }
  TLIE_CHECK(context, loaded.value().ExpectedTensors().size() == 201);
  TLIE_CHECK(context, loaded.value().head_dim() == 64);
  TLIE_CHECK(context, loaded.value().kv_dim() == 256);

  std::ifstream input(model_config);
  nlohmann::json invalid;
  input >> invalid;
  invalid["hidden_size"] = 1024;
  const auto invalid_path = temporary / "invalid-config.json";
  std::ofstream(invalid_path) << invalid;
  const auto rejected = tlie::ModelConfig::Load(invalid_path);
  TLIE_CHECK(context, !rejected.ok());
  TLIE_CHECK(context, rejected.error().code == tlie::ErrorCode::kInvalidConfig);
}

void TestWeightStore(Context& context, const std::filesystem::path& temporary) {
  const auto valid_path = temporary / "valid.tliewgt";
  WriteFixture(valid_path);
  const auto loaded = tlie::WeightStore::Load(valid_path);
  TLIE_CHECK(context, loaded.ok());
  if (loaded.ok()) {
    const auto tensor = loaded.value().Get("tensor");
    TLIE_CHECK(context, tensor.ok());
    if (tensor.ok()) {
      TLIE_CHECK(context, tensor.value().shape == std::vector<std::size_t>({2, 2}));
      TLIE_CHECK_NEAR(context, tensor.value().values[3], 4.0F, 0.0F);
    }
    TLIE_CHECK(context, !loaded.value().Get("missing").ok());
    TLIE_CHECK(context, loaded.value().Validate({tlie::TensorSpec{"tensor", {2, 2}}}).ok());
    const auto shape_error = loaded.value().Validate({tlie::TensorSpec{"tensor", {4, 1}}});
    TLIE_CHECK(context, !shape_error.ok());
    TLIE_CHECK(context, shape_error.error().code == tlie::ErrorCode::kInvalidShape);
  }

  tlie::WeightLoadOptions externally_pinned;
  externally_pinned.expected_file_sha256 = FileSha256(valid_path);
  TLIE_CHECK(context, tlie::WeightStore::Load(valid_path, externally_pinned).ok());
  const auto self_consistent_mutation_path = temporary / "self-consistent-mutation.tliewgt";
  WriteFixture(self_consistent_mutation_path, false, false, 1.5F);
  const auto externally_rejected =
      tlie::WeightStore::Load(self_consistent_mutation_path, externally_pinned);
  TLIE_CHECK(context, !externally_rejected.ok());
  TLIE_CHECK(context, externally_rejected.error().code == tlie::ErrorCode::kChecksumMismatch);

  externally_pinned.expected_file_sha256 = "not-a-checksum";
  const auto malformed_pin = tlie::WeightStore::Load(valid_path, externally_pinned);
  TLIE_CHECK(context, !malformed_pin.ok());
  TLIE_CHECK(context, malformed_pin.error().code == tlie::ErrorCode::kInvalidArgument);

  tlie::WeightLoadOptions limited;
  limited.max_file_bytes = 1;
  const auto oom = tlie::WeightStore::Load(valid_path, limited);
  TLIE_CHECK(context, !oom.ok());
  TLIE_CHECK(context, oom.error().code == tlie::ErrorCode::kOutOfMemory);

  const auto corrupt_path = temporary / "corrupt.tliewgt";
  WriteFixture(corrupt_path, true);
  const auto corrupt = tlie::WeightStore::Load(corrupt_path);
  TLIE_CHECK(context, !corrupt.ok());
  TLIE_CHECK(context, corrupt.error().code == tlie::ErrorCode::kChecksumMismatch);

  const auto shape_path = temporary / "shape.tliewgt";
  WriteFixture(shape_path, false, true);
  const auto shape = tlie::WeightStore::Load(shape_path);
  TLIE_CHECK(context, !shape.ok());
  TLIE_CHECK(context, shape.error().code == tlie::ErrorCode::kInvalidShape);

  const auto truncated_path = temporary / "truncated.tliewgt";
  std::ofstream(truncated_path, std::ios::binary).write("TLIE", 4);
  const auto truncated = tlie::WeightStore::Load(truncated_path);
  TLIE_CHECK(context, !truncated.ok());
  TLIE_CHECK(context, truncated.error().code == tlie::ErrorCode::kInvalidFormat);
}

void TestKvCache(Context& context) {
  auto cache = tlie::KvCache::Create(2, 3, 4);
  TLIE_CHECK(context, cache.ok());
  if (!cache.ok()) {
    return;
  }
  const std::array<float, 4> key = {1, 2, 3, 4};
  const std::array<float, 4> value = {5, 6, 7, 8};
  TLIE_CHECK(context, cache.value().Store(1, 2, key, value).ok());
  const auto keys = cache.value().Keys(1, 3);
  TLIE_CHECK(context, keys.ok());
  if (keys.ok()) {
    TLIE_CHECK_NEAR(context, keys.value()[10], 3.0F, 0.0F);
  }
  const auto out_of_bounds = cache.value().Store(2, 0, key, value);
  TLIE_CHECK(context, !out_of_bounds.ok());
  TLIE_CHECK(context, out_of_bounds.error().code == tlie::ErrorCode::kOutOfBounds);
  const auto wrong_shape = cache.value().Store(0, 0, std::span(key).first(3), value);
  TLIE_CHECK(context, !wrong_shape.ok());
  TLIE_CHECK(context, wrong_shape.error().code == tlie::ErrorCode::kInvalidShape);
  const auto oom = tlie::KvCache::Create(2, 3, 4, 16);
  TLIE_CHECK(context, !oom.ok());
  TLIE_CHECK(context, oom.error().code == tlie::ErrorCode::kOutOfMemory);
}

void TestSamplingAndNumerics(Context& context) {
  const std::array<float, 4> logits = {1.0F, 3.0F, 3.0F, 2.0F};
  const auto sampled = tlie::GreedySample(logits);
  TLIE_CHECK(context, sampled.ok());
  TLIE_CHECK(context, sampled.ok() && sampled.value() == 1);
  const std::array<float, 2> invalid = {0.0F, std::numeric_limits<float>::quiet_NaN()};
  const auto rejected = tlie::GreedySample(invalid);
  TLIE_CHECK(context, !rejected.ok());
  TLIE_CHECK(context, rejected.error().code == tlie::ErrorCode::kNumerical);
}

}  // namespace

int main() {
  Context context;
  const auto temporary = std::filesystem::temp_directory_path() / "tlie-unit-tests-current-process";
  std::filesystem::remove_all(temporary);
  std::filesystem::create_directories(temporary);
  TestSha256(context);
  TestConfig(context, temporary);
  TestWeightStore(context, temporary);
  TestKvCache(context);
  TestSamplingAndNumerics(context);
  std::filesystem::remove_all(temporary);
  return context.Finish("tlie_unit_tests");
}
