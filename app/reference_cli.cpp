#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tlie/model.hpp"
#include "tlie/model_config.hpp"
#include "tlie/operators.hpp"
#include "tlie/pinned_model.hpp"
#include "tlie/tokenizer.hpp"
#include "tlie/weight_store.hpp"

namespace {

int Failure(const tlie::Error& error) {
  const nlohmann::json document = {
      {"error", {{"code", tlie::ErrorCodeName(error.code)}, {"message", error.message}}}};
  std::cerr << document.dump() << '\n';
  return EXIT_FAILURE;
}

tlie::Result<int> ParseInteger(const char* text, const std::string& name, const int minimum) {
  const std::string_view input(text);
  int value = 0;
  const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc{} || end != input.data() + input.size() || value < minimum) {
    return tlie::Result<int>::Failure({tlie::ErrorCode::kInvalidArgument,
                                       name + " must be an integer >= " + std::to_string(minimum)});
  }
  return tlie::Result<int>::Success(value);
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: tinyllama_reference MODEL_DIR PROMPT MAX_TOKENS [EXPECTED_FIRST_TOKEN]\n";
    return EXIT_FAILURE;
  }
  const std::filesystem::path model_dir(argv[1]);
  const std::string prompt(argv[2]);
  const auto parsed_max_tokens = ParseInteger(argv[3], "MAX_TOKENS", 1);
  if (!parsed_max_tokens.ok()) {
    return Failure(parsed_max_tokens.error());
  }
  const int max_tokens = parsed_max_tokens.value();
  std::optional<int> expected_first_token;
  if (argc == 5) {
    auto parsed_expected = ParseInteger(argv[4], "EXPECTED_FIRST_TOKEN", 0);
    if (!parsed_expected.ok()) {
      return Failure(parsed_expected.error());
    }
    expected_first_token = parsed_expected.value();
  }
  auto config = tlie::ModelConfig::Load(model_dir / "config.json");
  if (!config.ok()) {
    return Failure(config.error());
  }
  auto tokenizer = tlie::Tokenizer::Load(model_dir / "tokenizer.model");
  if (!tokenizer.ok()) {
    return Failure(tokenizer.error());
  }
  auto prompt_ids = tokenizer.value().Encode(prompt);
  if (!prompt_ids.ok()) {
    return Failure(prompt_ids.error());
  }
  if (prompt_ids.value().size() + static_cast<std::size_t>(max_tokens) >
      config.value().max_position_embeddings) {
    return Failure({tlie::ErrorCode::kOutOfBounds, "prompt plus output exceeds context length"});
  }
  tlie::WeightLoadOptions options;
  options.expected_file_sha256 = tlie::kPinnedConvertedWeightSha256;
  options.expected_source_sha256 = tlie::kPinnedSourceWeightSha256;
  auto weights = tlie::WeightStore::Load(model_dir / "model-fp32.tliewgt", options);
  if (!weights.ok()) {
    return Failure(weights.error());
  }
  auto model = tlie::TinyLlamaCpu::Create(std::move(config).value(), std::move(weights).value(),
                                          prompt_ids.value().size() + max_tokens);
  if (!model.ok()) {
    return Failure(model.error());
  }
  std::vector<float> logits;
  for (std::size_t position = 0; position < prompt_ids.value().size(); ++position) {
    auto output = model.value().Forward(prompt_ids.value()[position], position);
    if (!output.ok()) {
      return Failure(output.error());
    }
    logits = std::move(output).value();
  }
  std::vector<int> generated;
  for (int step = 0; step < max_tokens; ++step) {
    auto sampled = tlie::GreedySample(logits);
    if (!sampled.ok()) {
      return Failure(sampled.error());
    }
    generated.push_back(sampled.value());
    if (step + 1 < max_tokens) {
      auto output = model.value().Forward(sampled.value(), prompt_ids.value().size() + step);
      if (!output.ok()) {
        return Failure(output.error());
      }
      logits = std::move(output).value();
    }
  }
  if (expected_first_token.has_value() && generated.front() != *expected_first_token) {
    std::cerr << "first generated token did not match expectation\n";
    return EXIT_FAILURE;
  }
  auto decoded = tokenizer.value().Decode(generated);
  if (!decoded.ok()) {
    return Failure(decoded.error());
  }
  std::cout << "token_ids:";
  for (const int token : generated) {
    std::cout << ' ' << token;
  }
  std::cout << "\ntext: " << decoded.value() << '\n';
  return EXIT_SUCCESS;
}
