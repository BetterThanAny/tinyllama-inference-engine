#include <httplib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tlie/cuda/model.hpp"
#include "tlie/cuda/weight_store.hpp"
#include "tlie/model_config.hpp"
#include "tlie/pinned_model.hpp"
#include "tlie/tokenizer.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kBatchCapacity = 4;
constexpr std::size_t kKvSlots = 20;
constexpr std::size_t kMaximumSequenceLength = 512;
constexpr std::size_t kMaximumOutputTokens = 128;
constexpr const char* kModelId = "tinyllama-1.1b";

enum class RequestState { kQueued, kRunning, kCompleted, kCancelled, kFailed };

const char* StateName(const RequestState state) {
  switch (state) {
    case RequestState::kQueued:
      return "queued";
    case RequestState::kRunning:
      return "running";
    case RequestState::kCompleted:
      return "completed";
    case RequestState::kCancelled:
      return "cancelled";
    case RequestState::kFailed:
      return "failed";
  }
  return "failed";
}

struct Request {
  std::string id;
  std::vector<int> prompt;
  std::size_t maximum_tokens{0};
  float temperature{0.0F};
  float top_p{1.0F};
  bool stream{false};
  Clock::time_point submitted;
  Clock::time_point deadline;
  std::mt19937 random;

  RequestState state{RequestState::kQueued};
  bool cancel_requested{false};
  std::optional<std::size_t> slot;
  std::size_t prompt_index{0};
  std::size_t position{0};
  std::vector<int> generated;
  std::string text;
  std::string error;
  double queue_ms{0.0};
  double ttft_ms{0.0};
  double tpot_ms{0.0};
  double output_tokens_per_second{0.0};
  Clock::time_point first_token;
  Clock::time_point finished;
  std::deque<std::string> chunks;
  mutable std::mutex mutex;
  std::condition_variable changed;
};

bool Terminal(const RequestState state) {
  return state == RequestState::kCompleted || state == RequestState::kCancelled ||
         state == RequestState::kFailed;
}

class KvBlockAllocator {
 public:
  KvBlockAllocator() {
    for (std::size_t slot = 0; slot < kKvSlots; ++slot) {
      free_.push_back(slot);
    }
  }

  std::optional<std::size_t> Acquire() {
    if (free_.empty()) {
      return std::nullopt;
    }
    const std::size_t slot = free_.front();
    free_.pop_front();
    in_use_[slot] = true;
    return slot;
  }

  bool Release(const std::size_t slot) {
    if (slot >= kKvSlots || !in_use_[slot]) {
      return false;
    }
    in_use_[slot] = false;
    free_.push_back(slot);
    return true;
  }

  [[nodiscard]] std::size_t used() const { return kKvSlots - free_.size(); }

 private:
  std::array<bool, kKvSlots> in_use_{};
  std::deque<std::size_t> free_;
};

class Engine {
 public:
  static tlie::Result<std::unique_ptr<Engine>> Create(const std::filesystem::path& model_dir) {
    auto tokenizer = tlie::Tokenizer::Load(model_dir / "tokenizer.model");
    if (!tokenizer.ok()) {
      return tlie::Result<std::unique_ptr<Engine>>::Failure(tokenizer.error());
    }
    auto config = tlie::ModelConfig::Load(model_dir / "config.json");
    if (!config.ok()) {
      return tlie::Result<std::unique_ptr<Engine>>::Failure(config.error());
    }
    tlie::WeightLoadOptions options;
    options.expected_file_sha256 = tlie::kPinnedCudaFp16WeightSha256;
    options.expected_source_sha256 = tlie::kPinnedSourceWeightSha256;
    auto weights = tlie::cuda::CudaWeightStore::Load(model_dir / "model-fp16.tliewgt", options);
    if (!weights.ok()) {
      return tlie::Result<std::unique_ptr<Engine>>::Failure(weights.error());
    }
    auto model =
        tlie::cuda::TinyLlamaCuda::Create(std::move(config).value(), std::move(weights).value(),
                                          kMaximumSequenceLength, false, kBatchCapacity, kKvSlots);
    if (!model.ok()) {
      return tlie::Result<std::unique_ptr<Engine>>::Failure(model.error());
    }
    auto engine =
        std::unique_ptr<Engine>(new Engine(std::move(tokenizer).value(), std::move(model).value()));
    engine->worker_ = std::thread([instance = engine.get()] { instance->Run(); });
    return tlie::Result<std::unique_ptr<Engine>>::Success(std::move(engine));
  }

  ~Engine() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
      work_.notify_all();
    }
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  tlie::Result<std::shared_ptr<Request>> Submit(const std::string& prompt,
                                                const std::size_t maximum_tokens,
                                                const float temperature, const float top_p,
                                                const bool stream,
                                                const std::chrono::milliseconds timeout) {
    auto encoded = [&] {
      std::lock_guard tokenizer_lock(tokenizer_mutex_);
      return tokenizer_.Encode(prompt);
    }();
    if (!encoded.ok()) {
      return tlie::Result<std::shared_ptr<Request>>::Failure(encoded.error());
    }
    if (encoded.value().empty() ||
        encoded.value().size() + maximum_tokens > kMaximumSequenceLength) {
      return tlie::Result<std::shared_ptr<Request>>::Failure(
          {tlie::ErrorCode::kOutOfBounds, "prompt plus max_tokens exceeds server context"});
    }
    auto request = std::make_shared<Request>();
    const std::uint64_t number = next_id_.fetch_add(1);
    request->id = "chatcmpl-" + std::to_string(number);
    request->prompt = std::move(encoded).value();
    request->maximum_tokens = maximum_tokens;
    request->temperature = temperature;
    request->top_p = top_p;
    request->stream = stream;
    request->submitted = Clock::now();
    request->deadline = request->submitted + timeout;
    request->random.seed(static_cast<std::uint32_t>(number));
    {
      std::lock_guard lock(mutex_);
      queued_.push_back(request);
      ++submitted_;
      work_.notify_one();
    }
    return tlie::Result<std::shared_ptr<Request>>::Success(std::move(request));
  }

  void Cancel(const std::shared_ptr<Request>& request) {
    std::lock_guard lock(mutex_);
    if (!Terminal(request->state)) {
      request->cancel_requested = true;
      work_.notify_one();
    }
  }

  nlohmann::json Metrics() const {
    std::lock_guard lock(mutex_);
    return {{"submitted_total", submitted_},
            {"completed_total", completed_},
            {"cancelled_total", cancelled_},
            {"failed_total", failed_},
            {"queued_requests", queued_.size()},
            {"active_sequences", active_.size()},
            {"kv_blocks_used", allocator_.used()},
            {"kv_blocks_capacity", kKvSlots},
            {"kv_utilization", static_cast<double>(allocator_.used()) / kKvSlots},
            {"last_queue_ms", last_queue_ms_},
            {"last_ttft_ms", last_ttft_ms_},
            {"last_tpot_ms", last_tpot_ms_},
            {"last_output_tokens_per_second", last_output_tokens_per_second_}};
  }

 private:
  Engine(tlie::Tokenizer tokenizer, tlie::cuda::TinyLlamaCuda model)
      : tokenizer_(std::move(tokenizer)), model_(std::move(model)) {}

  int Sample(const std::vector<float>& logits, Request& request) {
    if (!(request.temperature > 0.0F)) {
      return static_cast<int>(
          std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<std::pair<float, int>> probabilities;
    probabilities.reserve(logits.size());
    double total = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
      const float probability = std::exp((logits[index] - maximum) / request.temperature);
      probabilities.emplace_back(probability, static_cast<int>(index));
      total += probability;
    }
    std::sort(probabilities.begin(), probabilities.end(),
              [](const auto& left, const auto& right) { return left.first > right.first; });
    double cumulative = 0.0;
    std::size_t retained = 0;
    for (; retained < probabilities.size(); ++retained) {
      cumulative += probabilities[retained].first / total;
      if (cumulative >= request.top_p) {
        ++retained;
        break;
      }
    }
    retained = std::max<std::size_t>(retained, 1);
    double retained_total = 0.0;
    for (std::size_t index = 0; index < retained; ++index) {
      retained_total += probabilities[index].first;
    }
    std::uniform_real_distribution<double> distribution(0.0, retained_total);
    const double target = distribution(request.random);
    double cursor = 0.0;
    for (std::size_t index = 0; index < retained; ++index) {
      cursor += probabilities[index].first;
      if (target <= cursor) {
        return probabilities[index].second;
      }
    }
    return probabilities[retained - 1].second;
  }

  void EmitToken(const std::shared_ptr<Request>& request, const int token) {
    request->generated.push_back(token);
    std::string previous = request->text;
    {
      std::lock_guard tokenizer_lock(tokenizer_mutex_);
      auto decoded = tokenizer_.Decode(request->generated);
      if (!decoded.ok()) {
        request->error = decoded.error().message;
        return;
      }
      request->text = std::move(decoded).value();
    }
    std::string delta =
        request->text.starts_with(previous) ? request->text.substr(previous.size()) : request->text;
    if (request->generated.size() == 1) {
      request->first_token = Clock::now();
      request->ttft_ms =
          std::chrono::duration<double, std::milli>(request->first_token - request->submitted)
              .count();
    }
    if (request->stream) {
      nlohmann::json event = {
          {"id", request->id},
          {"object", "chat.completion.chunk"},
          {"model", kModelId},
          {"choices",
           nlohmann::json::array(
               {{{"index", 0}, {"delta", {{"content", delta}}}, {"finish_reason", nullptr}}})}};
      std::lock_guard request_lock(request->mutex);
      request->chunks.push_back("data: " + event.dump() + "\n\n");
      request->changed.notify_all();
    }
  }

  void Finish(const std::shared_ptr<Request>& request, const RequestState state,
              std::string error = {}) {
    if (Terminal(request->state)) {
      return;
    }
    std::lock_guard request_lock(request->mutex);
    if (request->slot.has_value()) {
      auto reset = model_.ResetSlot(*request->slot);
      if (!reset.ok()) {
        if (error.empty()) {
          error = reset.error().message;
        }
      } else {
        allocator_.Release(*request->slot);
        request->slot.reset();
      }
    }
    request->state = error.empty() ? state : RequestState::kFailed;
    request->error = std::move(error);
    request->finished = Clock::now();
    if (!request->generated.empty()) {
      const double generation_seconds =
          std::chrono::duration<double>(request->finished - request->first_token).count();
      request->output_tokens_per_second = request->generated.size() > 1 && generation_seconds > 0.0
                                              ? (request->generated.size() - 1) / generation_seconds
                                              : 0.0;
      request->tpot_ms = request->generated.size() > 1
                             ? 1000.0 * generation_seconds / (request->generated.size() - 1)
                             : 0.0;
    }
    if (request->state == RequestState::kCompleted) {
      ++completed_;
    } else if (request->state == RequestState::kCancelled) {
      ++cancelled_;
    } else {
      ++failed_;
    }
    last_queue_ms_ = request->queue_ms;
    last_ttft_ms_ = request->ttft_ms;
    last_tpot_ms_ = request->tpot_ms;
    last_output_tokens_per_second_ = request->output_tokens_per_second;
    if (request->stream) {
      request->chunks.push_back("data: [DONE]\n\n");
    }
    request->changed.notify_all();
  }

  void Admit() {
    while (!queued_.empty()) {
      auto slot = allocator_.Acquire();
      if (!slot.has_value()) {
        return;
      }
      auto request = queued_.front();
      queued_.pop_front();
      if (request->cancel_requested || Clock::now() >= request->deadline) {
        allocator_.Release(*slot);
        {
          std::lock_guard request_lock(request->mutex);
          request->state = RequestState::kCancelled;
        }
        ++cancelled_;
        request->changed.notify_all();
        continue;
      }
      {
        std::lock_guard request_lock(request->mutex);
        request->slot = *slot;
        request->state = RequestState::kRunning;
        request->queue_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - request->submitted).count();
      }
      active_.push_back(std::move(request));
    }
  }

  std::vector<std::shared_ptr<Request>> SelectBatch() {
    std::vector<std::shared_ptr<Request>> selected;
    if (active_.empty()) {
      return selected;
    }
    const std::size_t count = active_.size();
    for (int phase = 0; phase < 2 && selected.size() < kBatchCapacity; ++phase) {
      for (std::size_t offset = 0; offset < count && selected.size() < kBatchCapacity; ++offset) {
        const std::size_t index = (round_robin_ + offset) % count;
        const auto& request = active_[index];
        const bool decode = request->prompt_index >= request->prompt.size();
        if ((phase == 0) == decode &&
            std::find(selected.begin(), selected.end(), request) == selected.end()) {
          selected.push_back(request);
        }
      }
    }
    round_robin_ = (round_robin_ + selected.size()) % count;
    return selected;
  }

  void Run() {
    while (true) {
      std::vector<std::shared_ptr<Request>> batch;
      {
        std::unique_lock lock(mutex_);
        work_.wait(lock, [this] { return stopping_ || !queued_.empty() || !active_.empty(); });
        if (stopping_) {
          return;
        }
        Admit();
        const auto now = Clock::now();
        for (const auto& request : active_) {
          if (request->cancel_requested || now >= request->deadline) {
            Finish(request, RequestState::kCancelled);
          }
        }
        std::erase_if(active_, [](const auto& request) { return Terminal(request->state); });
        batch = SelectBatch();
      }
      if (batch.empty()) {
        continue;
      }
      std::vector<int> tokens;
      std::vector<std::size_t> positions;
      std::vector<std::size_t> slots;
      tokens.reserve(batch.size());
      positions.reserve(batch.size());
      slots.reserve(batch.size());
      for (const auto& request : batch) {
        tokens.push_back(request->prompt_index < request->prompt.size()
                             ? request->prompt[request->prompt_index]
                             : request->generated.back());
        positions.push_back(request->position);
        slots.push_back(*request->slot);
      }
      auto logits = model_.ForwardBatch(tokens, positions, slots);
      std::lock_guard lock(mutex_);
      if (!logits.ok()) {
        for (const auto& request : batch) {
          Finish(request, RequestState::kFailed, logits.error().message);
        }
      } else {
        for (std::size_t row = 0; row < batch.size(); ++row) {
          auto& request = batch[row];
          if (request->cancel_requested || Clock::now() >= request->deadline) {
            Finish(request, RequestState::kCancelled);
            continue;
          }
          ++request->position;
          if (request->prompt_index < request->prompt.size()) {
            ++request->prompt_index;
            if (request->prompt_index < request->prompt.size()) {
              continue;
            }
          }
          EmitToken(request, Sample(logits.value()[row], *request));
          if (!request->error.empty()) {
            Finish(request, RequestState::kFailed, request->error);
          } else if (request->generated.size() >= request->maximum_tokens) {
            Finish(request, RequestState::kCompleted);
          }
        }
      }
      std::erase_if(active_, [](const auto& request) { return Terminal(request->state); });
      Admit();
    }
  }

  tlie::Tokenizer tokenizer_;
  tlie::cuda::TinyLlamaCuda model_;
  mutable std::mutex tokenizer_mutex_;
  mutable std::mutex mutex_;
  std::condition_variable work_;
  std::deque<std::shared_ptr<Request>> queued_;
  std::vector<std::shared_ptr<Request>> active_;
  KvBlockAllocator allocator_;
  std::thread worker_;
  bool stopping_{false};
  std::size_t round_robin_{0};
  std::atomic<std::uint64_t> next_id_{1};
  std::uint64_t submitted_{0};
  std::uint64_t completed_{0};
  std::uint64_t cancelled_{0};
  std::uint64_t failed_{0};
  double last_queue_ms_{0.0};
  double last_ttft_ms_{0.0};
  double last_tpot_ms_{0.0};
  double last_output_tokens_per_second_{0.0};
};

nlohmann::json ErrorBody(const std::string& message, const std::string& code) {
  return {{"error", {{"message", message}, {"type", "invalid_request_error"}, {"code", code}}}};
}

void JsonResponse(httplib::Response& response, const int status, const nlohmann::json& body) {
  response.status = status;
  response.set_content(body.dump(), "application/json");
}

std::string PromptFromMessages(const nlohmann::json& messages) {
  std::string prompt;
  for (const auto& message : messages) {
    prompt += message.at("role").get<std::string>() + ": " +
              message.at("content").get<std::string>() + "\n";
  }
  prompt += "assistant:";
  return prompt;
}

nlohmann::json CompletionBody(const Request& request) {
  const char* finish_reason = request.state == RequestState::kCompleted ? "length" : "cancelled";
  return {{"id", request.id},
          {"object", "chat.completion"},
          {"model", kModelId},
          {"choices",
           nlohmann::json::array({{{"index", 0},
                                   {"message", {{"role", "assistant"}, {"content", request.text}}},
                                   {"finish_reason", finish_reason}}})},
          {"usage",
           {{"prompt_tokens", request.prompt.size()},
            {"completion_tokens", request.generated.size()},
            {"total_tokens", request.prompt.size() + request.generated.size()}}},
          {"metrics",
           {{"state", StateName(request.state)},
            {"queue_ms", request.queue_ms},
            {"ttft_ms", request.ttft_ms},
            {"tpot_ms", request.tpot_ms},
            {"output_tokens_per_second", request.output_tokens_per_second}}}};
}

std::optional<int> ParsePort(const std::string& value) {
  int port = 0;
  const char* begin = value.data();
  const char* end = begin + value.size();
  const auto [position, error] = std::from_chars(begin, end, port);
  if (error != std::errc{} || position != end || port < 1 || port > 65535) {
    return std::nullopt;
  }
  return port;
}

}  // namespace

int main(int argc, char** argv) {
  int port = 8080;
  std::filesystem::path model_dir = "models/tinyllama-chat-v1.0";
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--port" && index + 1 < argc) {
      auto parsed = ParsePort(argv[++index]);
      if (!parsed.has_value()) {
        std::cerr
            << ErrorBody("port must be an integer from 1 through 65535", "invalid_port").dump()
            << '\n';
        return EXIT_FAILURE;
      }
      port = *parsed;
    } else if (argument == "--model-dir" && index + 1 < argc) {
      model_dir = argv[++index];
    } else {
      std::cerr << "usage: tinyllama_server [--port PORT] [--model-dir PATH]\n";
      return EXIT_FAILURE;
    }
  }
  auto engine = Engine::Create(model_dir);
  if (!engine.ok()) {
    std::cerr << ErrorBody(engine.error().message, tlie::ErrorCodeName(engine.error().code)).dump()
              << '\n';
    return EXIT_FAILURE;
  }
  httplib::Server server;
  server.new_task_queue = [] { return new httplib::ThreadPool(32); };
  server.Get("/v1/models", [](const httplib::Request&, httplib::Response& response) {
    JsonResponse(response, 200,
                 {{"object", "list"},
                  {"data", nlohmann::json::array(
                               {{{"id", kModelId}, {"object", "model"}, {"owned_by", "tlie"}}})}});
  });
  server.Get("/metrics", [&engine](const httplib::Request&, httplib::Response& response) {
    JsonResponse(response, 200, engine.value()->Metrics());
  });
  server.Post("/v1/chat/completions", [&engine](const httplib::Request& http_request,
                                                httplib::Response& response) {
    try {
      const auto body = nlohmann::json::parse(http_request.body);
      if (!body.contains("messages") || !body.at("messages").is_array() ||
          body.at("messages").empty()) {
        JsonResponse(response, 400,
                     ErrorBody("messages must be a non-empty array", "invalid_messages"));
        return;
      }
      const std::size_t maximum_tokens = body.value("max_tokens", 16U);
      const float temperature = body.value("temperature", 0.0F);
      const float top_p = body.value("top_p", 1.0F);
      const bool stream = body.value("stream", false);
      const int timeout_ms = body.value("timeout_ms", 60000);
      if (maximum_tokens == 0 || maximum_tokens > kMaximumOutputTokens || temperature < 0.0F ||
          !std::isfinite(temperature) || top_p <= 0.0F || top_p > 1.0F || !std::isfinite(top_p) ||
          timeout_ms <= 0) {
        JsonResponse(response, 400,
                     ErrorBody("invalid max_tokens, temperature, top_p, or timeout_ms",
                               "invalid_generation_parameters"));
        return;
      }
      auto submitted =
          engine.value()->Submit(PromptFromMessages(body.at("messages")), maximum_tokens,
                                 temperature, top_p, stream, std::chrono::milliseconds(timeout_ms));
      if (!submitted.ok()) {
        JsonResponse(
            response, 400,
            ErrorBody(submitted.error().message, tlie::ErrorCodeName(submitted.error().code)));
        return;
      }
      const auto request = submitted.value();
      if (!stream) {
        std::unique_lock request_lock(request->mutex);
        request->changed.wait(request_lock, [&request] { return Terminal(request->state); });
        if (request->state == RequestState::kFailed) {
          JsonResponse(response, 500, ErrorBody(request->error, "generation_failed"));
        } else if (request->state == RequestState::kCancelled) {
          JsonResponse(response, 408,
                       ErrorBody("request cancelled or timed out", "request_cancelled"));
        } else {
          JsonResponse(response, 200, CompletionBody(*request));
        }
        return;
      }
      response.set_header("Cache-Control", "no-cache");
      response.set_chunked_content_provider(
          "text/event-stream",
          [request, &engine](std::size_t, httplib::DataSink& sink) {
            std::unique_lock request_lock(request->mutex);
            request->changed.wait_for(request_lock, std::chrono::milliseconds(100), [&request] {
              return !request->chunks.empty() || Terminal(request->state);
            });
            if (!request->chunks.empty()) {
              std::string chunk = std::move(request->chunks.front());
              request->chunks.pop_front();
              request_lock.unlock();
              if (!sink.write(chunk.data(), chunk.size())) {
                engine.value()->Cancel(request);
                return false;
              }
              return true;
            }
            if (Terminal(request->state)) {
              sink.done();
              return false;
            }
            return true;
          },
          [request, &engine](bool success) {
            if (!success) {
              engine.value()->Cancel(request);
            }
          });
    } catch (const std::exception& error) {
      JsonResponse(response, 400, ErrorBody(error.what(), "invalid_json"));
    }
  });
  server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
    if (response.status >= 400 && response.body.empty()) {
      JsonResponse(response, response.status, ErrorBody("HTTP request failed", "http_error"));
    }
  });
  std::cout << nlohmann::json({{"event", "ready"}, {"host", "127.0.0.1"}, {"port", port}}).dump()
            << std::endl;
  if (!server.listen("127.0.0.1", port)) {
    std::cerr << "unable to listen on 127.0.0.1:" << port << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
