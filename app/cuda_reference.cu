#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <vector>
#if defined(TLIE_USE_NVTX3)
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif

#include "tlie/cuda/model.hpp"
#include "tlie/cuda/weight_store.hpp"
#include "tlie/model_config.hpp"
#include "tlie/operators.hpp"
#include "tlie/pinned_model.hpp"
#include "tlie/tokenizer.hpp"

namespace {

enum class WeightMode { kFp16, kInt8 };

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

int Failure(const tlie::Error& error) {
  nlohmann::json document = {
      {"error", {{"code", tlie::ErrorCodeName(error.code)}, {"message", error.message}}}};
  std::cerr << document.dump() << '\n';
  return EXIT_FAILURE;
}

tlie::Result<int> ParsePositive(const char* text, const std::string& name) {
  int value = 0;
  const std::string input(text);
  const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc{} || end != input.data() + input.size() || value <= 0) {
    return tlie::Result<int>::Failure(
        {tlie::ErrorCode::kInvalidArgument, name + " must be a positive integer"});
  }
  return tlie::Result<int>::Success(value);
}

tlie::Result<tlie::cuda::TinyLlamaCuda> LoadModel(const std::filesystem::path& model_dir,
                                                  const std::size_t max_sequence_length,
                                                  const bool allow_extrapolation,
                                                  const WeightMode weight_mode) {
  auto config = tlie::ModelConfig::Load(model_dir / "config.json");
  if (!config.ok()) {
    return tlie::Result<tlie::cuda::TinyLlamaCuda>::Failure(config.error());
  }
  tlie::WeightLoadOptions options;
  options.expected_file_sha256 = weight_mode == WeightMode::kFp16
                                     ? tlie::kPinnedCudaFp16WeightSha256
                                     : tlie::kPinnedCudaInt8WeightSha256;
  options.expected_source_sha256 = tlie::kPinnedSourceWeightSha256;
  auto weights =
      weight_mode == WeightMode::kFp16
          ? tlie::cuda::CudaWeightStore::Load(model_dir / "model-fp16.tliewgt", options)
          : tlie::cuda::CudaWeightStore::LoadInt8(model_dir / "model-int8.tliewgt", options);
  if (!weights.ok()) {
    return tlie::Result<tlie::cuda::TinyLlamaCuda>::Failure(weights.error());
  }
  return tlie::cuda::TinyLlamaCuda::Create(std::move(config).value(), std::move(weights).value(),
                                           max_sequence_length, allow_extrapolation);
}

std::vector<int> RepeatedContext(const std::vector<int>& seed, const std::size_t context) {
  std::vector<int> tokens;
  tokens.reserve(context);
  if (context == 0) {
    return tokens;
  }
  tokens.push_back(seed.front());
  for (std::size_t index = 1; index < context; ++index) {
    tokens.push_back(seed[1 + (index - 1) % (seed.size() - 1)]);
  }
  return tokens;
}

tlie::Result<std::vector<float>> Prefill(tlie::cuda::TinyLlamaCuda& model,
                                         const std::vector<int>& tokens,
                                         std::vector<tlie::cuda::CudaStepTimings>* timings) {
  std::vector<float> logits;
  for (std::size_t position = 0; position < tokens.size(); ++position) {
    tlie::cuda::CudaStepTimings timing;
    auto output = model.Forward(tokens[position], position, &timing);
    if (!output.ok()) {
      return tlie::Result<std::vector<float>>::Failure(output.error());
    }
    logits = std::move(output).value();
    if (timings != nullptr) {
      timings->push_back(timing);
    }
  }
  return tlie::Result<std::vector<float>>::Success(std::move(logits));
}

float SumField(const std::vector<tlie::cuda::CudaStepTimings>& timings,
               const float tlie::cuda::CudaStepTimings::* field) {
  float total = 0.0F;
  for (const auto& timing : timings) {
    total += timing.*field;
  }
  return total;
}

tlie::Result<void> WarmupWorkload(tlie::cuda::TinyLlamaCuda& model, const std::vector<int>& tokens,
                                  const int output_tokens) {
  auto reset = model.Reset();
  if (!reset.ok()) {
    return reset;
  }
  auto logits = Prefill(model, tokens, nullptr);
  if (!logits.ok()) {
    return tlie::Result<void>::Failure(logits.error());
  }
  auto previous_token = tlie::GreedySample(logits.value());
  if (!previous_token.ok()) {
    return tlie::Result<void>::Failure(previous_token.error());
  }
  for (int step = 1; step < output_tokens; ++step) {
    auto output = model.Forward(previous_token.value(), tokens.size() + step - 1);
    if (!output.ok()) {
      return tlie::Result<void>::Failure(output.error());
    }
    logits = std::move(output);
    previous_token = tlie::GreedySample(logits.value());
    if (!previous_token.ok()) {
      return tlie::Result<void>::Failure(previous_token.error());
    }
  }
  return tlie::Result<void>::Success();
}

int Generate(const std::filesystem::path& model_dir, const std::string& prompt,
             const int maximum_tokens, const std::filesystem::path* golden_path,
             const WeightMode weight_mode) {
  auto tokenizer = tlie::Tokenizer::Load(model_dir / "tokenizer.model");
  if (!tokenizer.ok()) {
    return Failure(tokenizer.error());
  }
  auto prompt_ids = tokenizer.value().Encode(prompt);
  if (!prompt_ids.ok()) {
    return Failure(prompt_ids.error());
  }
  const std::size_t total_length =
      prompt_ids.value().size() + static_cast<std::size_t>(maximum_tokens);
  auto model = LoadModel(model_dir, total_length, false, weight_mode);
  if (!model.ok()) {
    return Failure(model.error());
  }
  std::vector<tlie::cuda::CudaStepTimings> prefill_timings;
  tlie::Result<std::vector<float>> logits = [&] {
    NvtxRange range("prefill");
    return Prefill(model.value(), prompt_ids.value(), &prefill_timings);
  }();
  if (!logits.ok()) {
    return Failure(logits.error());
  }
  std::vector<int> generated;
  std::vector<tlie::cuda::CudaStepTimings> decode_timings;
  for (int step = 0; step < maximum_tokens; ++step) {
    auto token = [&] {
      NvtxRange range("sampling");
      return tlie::GreedySample(logits.value());
    }();
    if (!token.ok()) {
      return Failure(token.error());
    }
    generated.push_back(token.value());
    if (step + 1 < maximum_tokens) {
      tlie::cuda::CudaStepTimings timing;
      auto output = [&] {
        NvtxRange range("decode");
        return model.value().Forward(token.value(), prompt_ids.value().size() + step, &timing);
      }();
      if (!output.ok()) {
        return Failure(output.error());
      }
      logits = std::move(output);
      decode_timings.push_back(timing);
    }
  }
  if (golden_path != nullptr) {
    std::ifstream input(*golden_path);
    if (!input) {
      return Failure({tlie::ErrorCode::kIo, "unable to open CUDA golden metadata"});
    }
    nlohmann::json golden;
    input >> golden;
    const auto expected = golden.at("greedy_tokens").get<std::vector<int>>();
    if (generated.size() > expected.size() ||
        !std::equal(generated.begin(), generated.end(), expected.begin())) {
      return Failure(
          {tlie::ErrorCode::kNumerical, "CUDA greedy tokens do not match the fixed FP32 golden"});
    }
  }
  auto decoded = tokenizer.value().Decode(generated);
  if (!decoded.ok()) {
    return Failure(decoded.error());
  }
  nlohmann::json result = {
      {"prompt_tokens", prompt_ids.value().size()},
      {"generated_tokens", generated},
      {"text", decoded.value()},
      {"prefill_compute_ms", SumField(prefill_timings, &tlie::cuda::CudaStepTimings::compute_ms)},
      {"prefill_transfer_ms", SumField(prefill_timings, &tlie::cuda::CudaStepTimings::transfer_ms)},
      {"decode_compute_ms", SumField(decode_timings, &tlie::cuda::CudaStepTimings::compute_ms)},
      {"decode_transfer_ms", SumField(decode_timings, &tlie::cuda::CudaStepTimings::transfer_ms)}};
  std::cout << result.dump() << '\n';
  return EXIT_SUCCESS;
}

int Logits(const std::filesystem::path& model_dir, const std::string& prompt,
           const WeightMode weight_mode) {
  auto tokenizer = tlie::Tokenizer::Load(model_dir / "tokenizer.model");
  if (!tokenizer.ok()) {
    return Failure(tokenizer.error());
  }
  auto prompt_ids = tokenizer.value().Encode(prompt);
  if (!prompt_ids.ok()) {
    return Failure(prompt_ids.error());
  }
  auto model = LoadModel(model_dir, prompt_ids.value().size(), false, weight_mode);
  if (!model.ok()) {
    return Failure(model.error());
  }
  const auto logits = [&] {
    NvtxRange range("prefill");
    return Prefill(model.value(), prompt_ids.value(), nullptr);
  }();
  if (!logits.ok()) {
    return Failure(logits.error());
  }
  std::cout << nlohmann::json(
                   {{"prompt_tokens", prompt_ids.value().size()}, {"logits", logits.value()}})
                   .dump()
            << '\n';
  return EXIT_SUCCESS;
}

int Benchmark(const std::filesystem::path& model_dir, const int context, const int output_tokens,
              const int warmup, const int samples, const WeightMode weight_mode) {
  constexpr const char* kPromptSeed = "The capital of France is";
  constexpr const char* kContextConstruction =
      "first seed token once, then cyclic repetition of remaining seed token IDs";
  if (context > 4096 || output_tokens < 2 || output_tokens > 256) {
    return Failure({tlie::ErrorCode::kOutOfBounds,
                    "M2 benchmark limits are context <= 4096 and output tokens 2..256"});
  }
  auto tokenizer = tlie::Tokenizer::Load(model_dir / "tokenizer.model");
  if (!tokenizer.ok()) {
    return Failure(tokenizer.error());
  }
  auto seed = tokenizer.value().Encode(kPromptSeed);
  if (!seed.ok()) {
    return Failure(seed.error());
  }
  if (seed.value().size() < 2) {
    return Failure({tlie::ErrorCode::kTokenizer,
                    "M2 benchmark prompt seed must encode to at least two tokens"});
  }
  const auto tokens = RepeatedContext(seed.value(), static_cast<std::size_t>(context));
  const bool extrapolated = context + output_tokens > 2048;
  std::size_t free_before_model = 0;
  std::size_t total_device_memory = 0;
  cudaError_t cuda_status = cudaMemGetInfo(&free_before_model, &total_device_memory);
  if (cuda_status != cudaSuccess) {
    return Failure({tlie::ErrorCode::kCuda, "unable to read CUDA memory before model load: " +
                                                std::string(cudaGetErrorString(cuda_status))});
  }
  auto model = LoadModel(model_dir, static_cast<std::size_t>(context + output_tokens), extrapolated,
                         weight_mode);
  if (!model.ok()) {
    return Failure(model.error());
  }
  const std::size_t allocation_count_before_workload = model.value().device_allocation_count();
  std::size_t free_after_model = 0;
  std::size_t total_after_model = 0;
  cuda_status = cudaMemGetInfo(&free_after_model, &total_after_model);
  if (cuda_status != cudaSuccess || total_after_model != total_device_memory ||
      free_after_model > free_before_model) {
    return Failure(
        {tlie::ErrorCode::kCuda, "CUDA memory accounting after model load is inconsistent"});
  }
  const std::size_t model_device_bytes = free_before_model - free_after_model;
  std::size_t minimum_free_device_memory = free_after_model;

  const auto record_device_memory = [&]() -> tlie::Result<void> {
    std::size_t free_device_memory = 0;
    std::size_t current_total_device_memory = 0;
    const cudaError_t status = cudaMemGetInfo(&free_device_memory, &current_total_device_memory);
    if (status != cudaSuccess || current_total_device_memory != total_device_memory) {
      return tlie::Result<void>::Failure(
          {tlie::ErrorCode::kCuda, "unable to read consistent CUDA memory during benchmark"});
    }
    minimum_free_device_memory = std::min(minimum_free_device_memory, free_device_memory);
    return tlie::Result<void>::Success();
  };

  for (int iteration = 0; iteration < warmup; ++iteration) {
    NvtxRange warmup_range("warmup");
    auto warmed = WarmupWorkload(model.value(), tokens, output_tokens);
    if (!warmed.ok()) {
      return Failure(warmed.error());
    }
    auto memory = record_device_memory();
    if (!memory.ok()) {
      return Failure(memory.error());
    }
  }

  nlohmann::json sample_results = nlohmann::json::array();
  std::vector<int> reference_generated_tokens;
  for (int sample = 0; sample < samples; ++sample) {
    auto reset = model.value().Reset();
    if (!reset.ok()) {
      return Failure(reset.error());
    }
    std::vector<tlie::cuda::CudaStepTimings> prefill_timings;
    tlie::Result<std::vector<float>> logits = [&] {
      NvtxRange range("prefill");
      return Prefill(model.value(), tokens, &prefill_timings);
    }();
    if (!logits.ok()) {
      return Failure(logits.error());
    }
    std::vector<tlie::cuda::CudaStepTimings> decode_timings;
    float sampling_ms = 0.0F;
    const auto first_sampling_start = std::chrono::steady_clock::now();
    auto previous_token = tlie::GreedySample(logits.value());
    const float first_sampling_ms = std::chrono::duration<float, std::milli>(
                                        std::chrono::steady_clock::now() - first_sampling_start)
                                        .count();
    if (!previous_token.ok()) {
      return Failure(previous_token.error());
    }
    std::vector<int> generated_tokens{previous_token.value()};
    for (int step = 1; step < output_tokens; ++step) {
      tlie::cuda::CudaStepTimings timing;
      auto output = [&] {
        NvtxRange range("decode");
        return model.value().Forward(previous_token.value(), tokens.size() + step - 1, &timing);
      }();
      if (!output.ok()) {
        return Failure(output.error());
      }
      logits = std::move(output);
      decode_timings.push_back(timing);
      const auto sampling_start = std::chrono::steady_clock::now();
      auto token = [&] {
        NvtxRange range("sampling");
        return tlie::GreedySample(logits.value());
      }();
      sampling_ms += std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() -
                                                              sampling_start)
                         .count();
      if (!token.ok()) {
        return Failure(token.error());
      }
      previous_token = std::move(token);
      generated_tokens.push_back(previous_token.value());
    }
    if (reference_generated_tokens.empty()) {
      reference_generated_tokens = generated_tokens;
    } else if (generated_tokens != reference_generated_tokens) {
      return Failure(
          {tlie::ErrorCode::kNumerical, "greedy tokens changed between benchmark samples"});
    }
    const float decode_compute_ms =
        SumField(decode_timings, &tlie::cuda::CudaStepTimings::compute_ms);
    const float decode_wall_ms = SumField(decode_timings, &tlie::cuda::CudaStepTimings::wall_ms);
    const float decode_steps = static_cast<float>(output_tokens - 1);
    const float prefill_wall_ms = SumField(prefill_timings, &tlie::cuda::CudaStepTimings::wall_ms);
    const float ttft_ms = prefill_wall_ms + first_sampling_ms;
    const float tpot_ms = (decode_wall_ms + sampling_ms) / decode_steps;
    sample_results.push_back(
        {{"prefill_compute_ms",
          SumField(prefill_timings, &tlie::cuda::CudaStepTimings::compute_ms)},
         {"prefill_transfer_ms",
          SumField(prefill_timings, &tlie::cuda::CudaStepTimings::transfer_ms)},
         {"prefill_wall_ms", prefill_wall_ms},
         {"first_sampling_ms", first_sampling_ms},
         {"ttft_ms", ttft_ms},
         {"decode_compute_ms", decode_compute_ms},
         {"decode_transfer_ms",
          SumField(decode_timings, &tlie::cuda::CudaStepTimings::transfer_ms)},
         {"decode_wall_ms", decode_wall_ms},
         {"sampling_ms", sampling_ms},
         {"tpot_ms", tpot_ms},
         {"decode_compute_tokens_per_second", decode_steps * 1000.0F / decode_compute_ms},
         {"decode_tokens_per_second", decode_steps * 1000.0F / (decode_wall_ms + sampling_ms)}});
    auto memory = record_device_memory();
    if (!memory.ok()) {
      return Failure(memory.error());
    }
  }
  const std::size_t allocation_count_after_workload = model.value().device_allocation_count();
  if (allocation_count_after_workload != allocation_count_before_workload) {
    return Failure(
        {tlie::ErrorCode::kCuda, "CUDA device allocation count changed during the token workload"});
  }
  cudaDeviceProp properties{};
  int device = 0;
  cuda_status = cudaGetDevice(&device);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaGetDeviceProperties(&properties, device);
  }
  int driver_version = 0;
  int runtime_version = 0;
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaDriverGetVersion(&driver_version);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaRuntimeGetVersion(&runtime_version);
  }
  if (cuda_status != cudaSuccess) {
    return Failure({tlie::ErrorCode::kCuda, "unable to read CUDA device metadata: " +
                                                std::string(cudaGetErrorString(cuda_status))});
  }
  nlohmann::json result = {
      {"schema_version", 1},
      {"gpu", properties.name},
      {"vram_bytes", properties.totalGlobalMem},
      {"model_device_bytes", model_device_bytes},
      {"engine_peak_device_bytes", free_before_model - minimum_free_device_memory},
      {"device_allocation_count_before_workload", allocation_count_before_workload},
      {"device_allocation_count_after_workload", allocation_count_after_workload},
      {"kv_cache_bytes", model.value().kv_cache_bytes()},
      {"compute_capability",
       std::to_string(properties.major) + "." + std::to_string(properties.minor)},
      {"driver_version", driver_version},
      {"cuda_runtime_version", runtime_version},
      {"model_source_sha256", tlie::kPinnedSourceWeightSha256},
      {"model_weight_sha256", weight_mode == WeightMode::kFp16 ? tlie::kPinnedCudaFp16WeightSha256
                                                               : tlie::kPinnedCudaInt8WeightSha256},
      {"dtype", weight_mode == WeightMode::kFp16 ? "float16" : "int8_weight_only"},
      {"prompt_seed_text", kPromptSeed},
      {"prompt_seed_token_ids", seed.value()},
      {"context_construction", kContextConstruction},
      {"sampling", "greedy"},
      {"batch", 1},
      {"context", context},
      {"output_tokens", output_tokens},
      {"warmup", warmup},
      {"sample_count", samples},
      {"generated_tokens", reference_generated_tokens},
      {"rope_extrapolated_beyond_trained_context", extrapolated},
      {"samples", sample_results}};
  if (weight_mode == WeightMode::kFp16) {
    result["model_fp16_sha256"] = tlie::kPinnedCudaFp16WeightSha256;
  } else {
    result["model_int8_sha256"] = tlie::kPinnedCudaInt8WeightSha256;
  }
  std::cout << result.dump() << '\n';
  return EXIT_SUCCESS;
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: tinyllama_cuda generate MODEL_DIR PROMPT MAX_TOKENS [GOLDEN_JSON]\n"
                 "       tinyllama_cuda generate-int8 MODEL_DIR PROMPT MAX_TOKENS [GOLDEN_JSON]\n"
                 "       tinyllama_cuda logits MODEL_DIR PROMPT\n"
                 "       tinyllama_cuda logits-int8 MODEL_DIR PROMPT\n"
                 "       tinyllama_cuda benchmark MODEL_DIR CONTEXT OUTPUT_TOKENS WARMUP SAMPLES\n"
                 "       tinyllama_cuda benchmark-int8 MODEL_DIR CONTEXT OUTPUT_TOKENS WARMUP "
                 "SAMPLES\n";
    return EXIT_FAILURE;
  }
  const std::string command(argv[1]);
  if ((command == "generate" || command == "generate-int8") && (argc == 5 || argc == 6)) {
    const auto maximum_tokens = ParsePositive(argv[4], "MAX_TOKENS");
    if (!maximum_tokens.ok()) {
      return Failure(maximum_tokens.error());
    }
    const std::filesystem::path golden = argc == 6 ? argv[5] : "";
    return Generate(argv[2], argv[3], maximum_tokens.value(), argc == 6 ? &golden : nullptr,
                    command == "generate" ? WeightMode::kFp16 : WeightMode::kInt8);
  }
  if ((command == "logits" || command == "logits-int8") && argc == 4) {
    return Logits(argv[2], argv[3], command == "logits" ? WeightMode::kFp16 : WeightMode::kInt8);
  }
  if ((command == "benchmark" || command == "benchmark-int8") && argc == 7) {
    const auto context = ParsePositive(argv[3], "CONTEXT");
    const auto output_tokens = ParsePositive(argv[4], "OUTPUT_TOKENS");
    const auto warmup = ParsePositive(argv[5], "WARMUP");
    const auto samples = ParsePositive(argv[6], "SAMPLES");
    if (!context.ok()) {
      return Failure(context.error());
    }
    if (!output_tokens.ok()) {
      return Failure(output_tokens.error());
    }
    if (!warmup.ok()) {
      return Failure(warmup.error());
    }
    if (!samples.ok()) {
      return Failure(samples.error());
    }
    return Benchmark(argv[2], context.value(), output_tokens.value(), warmup.value(),
                     samples.value(),
                     command == "benchmark" ? WeightMode::kFp16 : WeightMode::kInt8);
  }
  std::cerr << "invalid tinyllama_cuda command or arguments\n";
  return EXIT_FAILURE;
}
