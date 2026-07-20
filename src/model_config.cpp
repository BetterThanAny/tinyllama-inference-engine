#include "tlie/model_config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace tlie {
namespace {

Result<ModelConfig> ConfigFailure(const std::string& message) {
  return Result<ModelConfig>::Failure({ErrorCode::kInvalidConfig, message});
}

}  // namespace

Result<ModelConfig> ModelConfig::Load(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return Result<ModelConfig>::Failure(
        {ErrorCode::kIo, "unable to open model config: " + path.string()});
  }
  try {
    nlohmann::json json;
    input >> json;
    if (json.at("model_type").get<std::string>() != "llama") {
      return ConfigFailure("model_type must be llama");
    }
    ModelConfig config;
    config.hidden_size = json.at("hidden_size").get<std::size_t>();
    config.intermediate_size = json.at("intermediate_size").get<std::size_t>();
    config.num_hidden_layers = json.at("num_hidden_layers").get<std::size_t>();
    config.num_attention_heads = json.at("num_attention_heads").get<std::size_t>();
    config.num_key_value_heads = json.at("num_key_value_heads").get<std::size_t>();
    config.max_position_embeddings = json.at("max_position_embeddings").get<std::size_t>();
    config.vocab_size = json.at("vocab_size").get<std::size_t>();
    config.bos_token_id = json.at("bos_token_id").get<int>();
    config.eos_token_id = json.at("eos_token_id").get<int>();
    config.rms_norm_eps = json.at("rms_norm_eps").get<float>();
    config.rope_theta = json.at("rope_theta").get<float>();
    config.attention_bias = json.at("attention_bias").get<bool>();
    config.tie_word_embeddings = json.at("tie_word_embeddings").get<bool>();
    config.hidden_act = json.at("hidden_act").get<std::string>();
    config.source_dtype = json.at("torch_dtype").get<std::string>();
    auto validation = config.ValidateFixedTinyLlama();
    if (!validation.ok()) {
      return Result<ModelConfig>::Failure(validation.error());
    }
    return Result<ModelConfig>::Success(std::move(config));
  } catch (const nlohmann::json::exception& exception) {
    return ConfigFailure("invalid model config JSON: " + std::string(exception.what()));
  }
}

Result<void> ModelConfig::ValidateFixedTinyLlama() const {
  const auto invalid = [](const std::string& detail) {
    return Result<void>::Failure({ErrorCode::kInvalidConfig, detail});
  };
  if (hidden_size != 2048 || intermediate_size != 5632 || num_hidden_layers != 22 ||
      num_attention_heads != 32 || num_key_value_heads != 4 || max_position_embeddings != 2048 ||
      vocab_size != 32000) {
    return invalid("architecture does not match pinned TinyLlama-1.1B-Chat-v1.0");
  }
  if (hidden_size % num_attention_heads != 0 || num_attention_heads % num_key_value_heads != 0) {
    return invalid("attention head dimensions are inconsistent");
  }
  if (bos_token_id != 1 || eos_token_id != 2) {
    return invalid("special token IDs do not match the pinned tokenizer");
  }
  if (rms_norm_eps != 1.0e-5F || rope_theta != 10000.0F || hidden_act != "silu") {
    return invalid("normalization, RoPE, or activation config is unsupported");
  }
  if (attention_bias || tie_word_embeddings || source_dtype != "bfloat16") {
    return invalid("bias, tied embedding, or source dtype differs from the pinned model");
  }
  return Result<void>::Success();
}

std::vector<TensorSpec> ModelConfig::ExpectedTensors() const {
  std::vector<TensorSpec> specs;
  specs.reserve(num_hidden_layers * 9 + 3);
  specs.push_back({"model.embed_tokens.weight", {vocab_size, hidden_size}});
  for (std::size_t layer = 0; layer < num_hidden_layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    specs.push_back({prefix + "input_layernorm.weight", {hidden_size}});
    specs.push_back({prefix + "self_attn.q_proj.weight", {hidden_size, hidden_size}});
    specs.push_back({prefix + "self_attn.k_proj.weight", {kv_dim(), hidden_size}});
    specs.push_back({prefix + "self_attn.v_proj.weight", {kv_dim(), hidden_size}});
    specs.push_back({prefix + "self_attn.o_proj.weight", {hidden_size, hidden_size}});
    specs.push_back({prefix + "post_attention_layernorm.weight", {hidden_size}});
    specs.push_back({prefix + "mlp.gate_proj.weight", {intermediate_size, hidden_size}});
    specs.push_back({prefix + "mlp.up_proj.weight", {intermediate_size, hidden_size}});
    specs.push_back({prefix + "mlp.down_proj.weight", {hidden_size, intermediate_size}});
  }
  specs.push_back({"model.norm.weight", {hidden_size}});
  specs.push_back({"lm_head.weight", {vocab_size, hidden_size}});
  return specs;
}

}  // namespace tlie
