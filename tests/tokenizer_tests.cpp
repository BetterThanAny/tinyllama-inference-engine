#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "test_support.hpp"
#include "tlie/tokenizer.hpp"

int main() {
  tlie::test::Context context;
  const auto model_path = std::filesystem::path(TLIE_MODEL_DIR) / "tokenizer.model";
  auto tokenizer = tlie::Tokenizer::Load(model_path);
  TLIE_CHECK(context, tokenizer.ok());
  if (!tokenizer.ok()) {
    return context.Finish("tlie_tokenizer_tests");
  }
  TLIE_CHECK(context, tokenizer.value().bos_id() == 1);
  TLIE_CHECK(context, tokenizer.value().eos_id() == 2);
  TLIE_CHECK(context, tokenizer.value().unk_id() == 0);
  TLIE_CHECK(context, tokenizer.value().vocab_size() == 32000);

  std::ifstream input(std::filesystem::path(TLIE_SOURCE_DIR) / "data/golden/tokenizer_cases.json");
  TLIE_CHECK(context, static_cast<bool>(input));
  if (!input) {
    return context.Finish("tlie_tokenizer_tests");
  }
  nlohmann::json golden;
  input >> golden;
  TLIE_CHECK(context, golden.at("cases").size() >= 7);
  for (const auto& test_case : golden.at("cases")) {
    const auto encoded = tokenizer.value().Encode(test_case.at("text").get<std::string>(),
                                                  test_case.at("add_bos").get<bool>(),
                                                  test_case.at("add_eos").get<bool>());
    TLIE_CHECK(context, encoded.ok());
    if (encoded.ok()) {
      TLIE_CHECK(context, encoded.value() == test_case.at("ids").get<std::vector<int>>());
    }
  }

  const std::string invalid_utf8("\xc0\xaf", 2);
  const auto invalid = tokenizer.value().Encode(invalid_utf8);
  TLIE_CHECK(context, !invalid.ok());
  TLIE_CHECK(context, invalid.error().code == tlie::ErrorCode::kTokenizer);

  auto limited = tlie::Tokenizer::Load(model_path, 4);
  TLIE_CHECK(context, limited.ok());
  if (limited.ok()) {
    const auto too_long = limited.value().Encode("12345");
    TLIE_CHECK(context, !too_long.ok());
    TLIE_CHECK(context, too_long.error().code == tlie::ErrorCode::kOutOfBounds);
  }
  const auto invalid_id = tokenizer.value().Decode({32000});
  TLIE_CHECK(context, !invalid_id.ok());
  TLIE_CHECK(context, invalid_id.error().code == tlie::ErrorCode::kOutOfBounds);
  return context.Finish("tlie_tokenizer_tests");
}
