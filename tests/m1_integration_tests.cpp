#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "test_support.hpp"
#include "tlie/model.hpp"
#include "tlie/model_config.hpp"
#include "tlie/operators.hpp"
#include "tlie/pinned_model.hpp"
#include "tlie/weight_store.hpp"

namespace {

bool Close(const float actual, const float expected, const float absolute_tolerance,
           const float relative_tolerance) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <=
             absolute_tolerance + relative_tolerance * std::abs(expected);
}

}  // namespace

int main() {
  tlie::test::Context context;
  const std::filesystem::path model_dir(TLIE_MODEL_DIR);
  const std::filesystem::path golden_dir(TLIE_GOLDEN_DIR);
  std::ifstream metadata_input(golden_dir / "reference.json");
  TLIE_CHECK(context, static_cast<bool>(metadata_input));
  if (!metadata_input) {
    return context.Finish("tlie_m1_integration_tests");
  }
  nlohmann::json metadata;
  metadata_input >> metadata;
  TLIE_CHECK(context, metadata.at("greedy_tokens").size() == 32);

  std::ifstream manifest_input(std::filesystem::path(TLIE_SOURCE_DIR) /
                               "config/model_manifest.json");
  TLIE_CHECK(context, static_cast<bool>(manifest_input));
  if (!manifest_input) {
    return context.Finish("tlie_m1_integration_tests");
  }
  nlohmann::json manifest;
  manifest_input >> manifest;
  const auto source_sha256 =
      manifest.at("files").at("model.safetensors").at("sha256").get<std::string>();
  const auto converted_sha256 =
      manifest.at("converted_format").at("file_sha256").get<std::string>();
  TLIE_CHECK(context, source_sha256 == tlie::kPinnedSourceWeightSha256);
  TLIE_CHECK(context, converted_sha256 == tlie::kPinnedConvertedWeightSha256);
  TLIE_CHECK(context, metadata.at("source_model_sha256").get<std::string>() == source_sha256);

  auto config = tlie::ModelConfig::Load(model_dir / "config.json");
  TLIE_CHECK(context, config.ok());
  if (!config.ok()) {
    return context.Finish("tlie_m1_integration_tests");
  }
  tlie::WeightLoadOptions options;
  options.expected_file_sha256 = converted_sha256;
  options.expected_source_sha256 = source_sha256;
  auto weights = tlie::WeightStore::Load(model_dir / "model-fp32.tliewgt", options);
  TLIE_CHECK(context, weights.ok());
  if (!weights.ok()) {
    std::cerr << tlie::ErrorCodeName(weights.error().code) << ": " << weights.error().message
              << '\n';
    return context.Finish("tlie_m1_integration_tests");
  }
  tlie::WeightLoadOptions golden_options;
  golden_options.expected_file_sha256 = metadata.at("trace_sha256").get<std::string>();
  golden_options.expected_source_sha256 = source_sha256;
  auto golden = tlie::WeightStore::Load(golden_dir / metadata.at("trace_file").get<std::string>(),
                                        golden_options);
  TLIE_CHECK(context, golden.ok());
  if (!golden.ok()) {
    return context.Finish("tlie_m1_integration_tests");
  }
  const auto prompt_ids = metadata.at("prompt_ids").get<std::vector<int>>();
  const auto expected_tokens = metadata.at("greedy_tokens").get<std::vector<int>>();
  auto model = tlie::TinyLlamaCpu::Create(std::move(config).value(), std::move(weights).value(),
                                          prompt_ids.size() + expected_tokens.size());
  TLIE_CHECK(context, model.ok());
  if (!model.ok()) {
    std::cerr << tlie::ErrorCodeName(model.error().code) << ": " << model.error().message << '\n';
    return context.Finish("tlie_m1_integration_tests");
  }

  tlie::ModelTrace trace;
  std::vector<float> logits;
  for (std::size_t position = 0; position < prompt_ids.size(); ++position) {
    auto result = model.value().Forward(prompt_ids[position], position,
                                        position + 1 == prompt_ids.size() ? &trace : nullptr);
    TLIE_CHECK(context, result.ok());
    if (!result.ok()) {
      std::cerr << tlie::ErrorCodeName(result.error().code) << ": " << result.error().message
                << '\n';
      return context.Finish("tlie_m1_integration_tests");
    }
    logits = std::move(result).value();
  }

  const float absolute_tolerance = metadata.at("atol").get<float>();
  const float relative_tolerance = metadata.at("rtol").get<float>();
  for (const auto& [name, actual] : trace.tensors) {
    const auto expected = golden.value().Get(name);
    TLIE_CHECK(context, expected.ok());
    if (!expected.ok()) {
      continue;
    }
    TLIE_CHECK(context, actual.size() == expected.value().values.size());
    if (actual.size() != expected.value().values.size()) {
      continue;
    }
    float maximum_error = 0.0F;
    bool all_close = true;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      maximum_error =
          std::max(maximum_error, std::abs(actual[index] - expected.value().values[index]));
      all_close = all_close && Close(actual[index], expected.value().values[index],
                                     absolute_tolerance, relative_tolerance);
    }
    std::cout << name << " max_abs_error=" << maximum_error << '\n';
    TLIE_CHECK(context, all_close);
  }
  TLIE_CHECK(context, trace.tensors.size() == metadata.at("trace_tensors").size());

  for (std::size_t step = 0; step < expected_tokens.size(); ++step) {
    TLIE_CHECK(context, tlie::AllFinite(logits));
    const auto token = tlie::GreedySample(logits);
    TLIE_CHECK(context, token.ok());
    if (!token.ok()) {
      return context.Finish("tlie_m1_integration_tests");
    }
    TLIE_CHECK(context, token.value() == expected_tokens[step]);
    if (token.value() != expected_tokens[step]) {
      std::cerr << "greedy mismatch at step " << step << ": actual=" << token.value()
                << ", expected=" << expected_tokens[step] << '\n';
      return context.Finish("tlie_m1_integration_tests");
    }
    if (step + 1 < expected_tokens.size()) {
      auto result = model.value().Forward(token.value(), prompt_ids.size() + step);
      TLIE_CHECK(context, result.ok());
      if (!result.ok()) {
        return context.Finish("tlie_m1_integration_tests");
      }
      logits = std::move(result).value();
    }
  }
  return context.Finish("tlie_m1_integration_tests");
}
