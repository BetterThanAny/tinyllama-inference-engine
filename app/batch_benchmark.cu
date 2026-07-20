#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <vector>

#include "tlie/cuda/model.hpp"
#include "tlie/cuda/weight_store.hpp"
#include "tlie/model_config.hpp"
#include "tlie/operators.hpp"
#include "tlie/pinned_model.hpp"
#include "tlie/tokenizer.hpp"

#ifndef TLIE_MODEL_DIR
#define TLIE_MODEL_DIR "models/tinyllama-chat-v1.0"
#endif

namespace {

int Failure(const tlie::Error& error) {
  std::cerr << nlohmann::json(
                   {{"error",
                     {{"code", tlie::ErrorCodeName(error.code)}, {"message", error.message}}}})
                   .dump()
            << '\n';
  return EXIT_FAILURE;
}

tlie::Result<tlie::cuda::TinyLlamaCuda> LoadModel(const std::filesystem::path& model_dir,
                                                  const std::size_t maximum_length) {
  auto config = tlie::ModelConfig::Load(model_dir / "config.json");
  if (!config.ok()) {
    return tlie::Result<tlie::cuda::TinyLlamaCuda>::Failure(config.error());
  }
  tlie::WeightLoadOptions options;
  options.expected_file_sha256 = tlie::kPinnedCudaFp16WeightSha256;
  options.expected_source_sha256 = tlie::kPinnedSourceWeightSha256;
  auto weights = tlie::cuda::CudaWeightStore::Load(model_dir / "model-fp16.tliewgt", options);
  if (!weights.ok()) {
    return tlie::Result<tlie::cuda::TinyLlamaCuda>::Failure(weights.error());
  }
  return tlie::cuda::TinyLlamaCuda::Create(std::move(config).value(), std::move(weights).value(),
                                           maximum_length, false, 4, 4);
}

std::vector<int> RepeatedContext(const std::vector<int>& seed, const std::size_t context) {
  std::vector<int> result;
  result.reserve(context);
  result.push_back(seed.front());
  for (std::size_t index = 1; index < context; ++index) {
    result.push_back(seed[1 + (index - 1) % (seed.size() - 1)]);
  }
  return result;
}

struct WorkloadResult {
  double seconds{0.0};
  std::vector<std::vector<int>> generated;
};

tlie::Result<WorkloadResult> RunWorkload(tlie::cuda::TinyLlamaCuda& model,
                                         const std::vector<int>& prompt,
                                         const std::size_t output_tokens, const std::size_t batch) {
  auto reset = model.Reset();
  if (!reset.ok()) {
    return tlie::Result<WorkloadResult>::Failure(reset.error());
  }
  std::vector<std::size_t> slots(batch);
  std::iota(slots.begin(), slots.end(), 0);
  std::vector<std::size_t> positions(batch);
  std::vector<int> input(batch);
  std::vector<std::vector<float>> logits;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t position = 0; position < prompt.size(); ++position) {
    std::fill(input.begin(), input.end(), prompt[position]);
    std::fill(positions.begin(), positions.end(), position);
    auto output = model.ForwardBatch(input, positions, slots);
    if (!output.ok()) {
      return tlie::Result<WorkloadResult>::Failure(output.error());
    }
    logits = std::move(output).value();
  }
  WorkloadResult result;
  result.generated.assign(batch, {});
  for (auto& tokens : result.generated) {
    tokens.reserve(output_tokens);
  }
  for (std::size_t step = 0; step < output_tokens; ++step) {
    for (std::size_t row = 0; row < batch; ++row) {
      auto sampled = tlie::GreedySample(logits[row]);
      if (!sampled.ok()) {
        return tlie::Result<WorkloadResult>::Failure(sampled.error());
      }
      input[row] = sampled.value();
      result.generated[row].push_back(sampled.value());
      positions[row] = prompt.size() + step;
    }
    if (step + 1 < output_tokens) {
      auto output = model.ForwardBatch(input, positions, slots);
      if (!output.ok()) {
        return tlie::Result<WorkloadResult>::Failure(output.error());
      }
      logits = std::move(output).value();
    }
  }
  result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return tlie::Result<WorkloadResult>::Success(std::move(result));
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

}  // namespace

int main(int argc, char** argv) {
  const bool test_mode = argc == 2 && std::string(argv[1]) == "--test";
  const bool benchmark_mode = argc == 2 && std::string(argv[1]) == "--benchmark";
  if (!test_mode && !benchmark_mode) {
    std::cerr << "usage: batch_benchmark --test|--benchmark\n";
    return EXIT_FAILURE;
  }
  const std::filesystem::path model_dir = TLIE_MODEL_DIR;
  const std::size_t context = test_mode ? 8 : 128;
  const std::size_t output_tokens = test_mode ? 4 : 32;
  const int warmup = test_mode ? 0 : 3;
  const int samples = test_mode ? 1 : 10;
  auto tokenizer = tlie::Tokenizer::Load(model_dir / "tokenizer.model");
  if (!tokenizer.ok()) {
    return Failure(tokenizer.error());
  }
  auto seed = tokenizer.value().Encode("The capital of France is");
  if (!seed.ok() || seed.value().size() < 2) {
    return Failure(seed.ok() ? tlie::Error{tlie::ErrorCode::kTokenizer, "invalid benchmark seed"}
                             : seed.error());
  }
  const auto prompt = RepeatedContext(seed.value(), context);
  auto model = LoadModel(model_dir, context + output_tokens);
  if (!model.ok()) {
    return Failure(model.error());
  }
  for (int iteration = 0; iteration < warmup; ++iteration) {
    auto one = RunWorkload(model.value(), prompt, output_tokens, 1);
    auto four = RunWorkload(model.value(), prompt, output_tokens, 4);
    if (!one.ok() || !four.ok()) {
      return Failure(one.ok() ? four.error() : one.error());
    }
  }
  std::vector<double> batch_one_seconds;
  std::vector<double> batch_four_seconds;
  std::vector<int> reference_tokens;
  bool tokens_match = true;
  for (int sample = 0; sample < samples; ++sample) {
    auto one = RunWorkload(model.value(), prompt, output_tokens, 1);
    auto four = RunWorkload(model.value(), prompt, output_tokens, 4);
    if (!one.ok() || !four.ok()) {
      return Failure(one.ok() ? four.error() : one.error());
    }
    batch_one_seconds.push_back(one.value().seconds);
    batch_four_seconds.push_back(four.value().seconds);
    reference_tokens = one.value().generated.front();
    for (const auto& generated : four.value().generated) {
      tokens_match = tokens_match && generated == reference_tokens;
    }
  }
  const double batch_one_tps = static_cast<double>(output_tokens) / Median(batch_one_seconds);
  const double batch_four_tps = static_cast<double>(4 * output_tokens) / Median(batch_four_seconds);
  const double ratio = batch_four_tps / batch_one_tps;
  const bool passed = tokens_match && (test_mode || ratio >= 1.5);
  int cuda_runtime_version = 0;
  const cudaError_t version_status = cudaRuntimeGetVersion(&cuda_runtime_version);
  if (version_status != cudaSuccess) {
    std::cerr << "cudaRuntimeGetVersion failed: " << cudaGetErrorString(version_status) << '\n';
    return EXIT_FAILURE;
  }
  std::cout << nlohmann::json({{"schema_version", 1},
                               {"mode", test_mode ? "test" : "benchmark"},
                               {"dtype", "float16"},
                               {"sampling", "greedy"},
                               {"cuda_runtime_version", cuda_runtime_version},
                               {"model_fp16_sha256", tlie::kPinnedCudaFp16WeightSha256},
                               {"context", context},
                               {"output_tokens", output_tokens},
                               {"warmup", warmup},
                               {"samples", samples},
                               {"batch_1_total_tokens_per_second", batch_one_tps},
                               {"batch_4_total_tokens_per_second", batch_four_tps},
                               {"batch_4_over_batch_1", ratio},
                               {"tokens_match", tokens_match},
                               {"kv_slots", model.value().kv_slot_count()},
                               {"passed", passed}})
                   .dump()
            << '\n';
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
