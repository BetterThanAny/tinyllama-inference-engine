#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "tlie/result.hpp"

namespace tlie {

struct TensorSpec {
  std::string name;
  std::vector<std::size_t> shape;
};

struct ModelConfig {
  std::size_t hidden_size{0};
  std::size_t intermediate_size{0};
  std::size_t num_hidden_layers{0};
  std::size_t num_attention_heads{0};
  std::size_t num_key_value_heads{0};
  std::size_t max_position_embeddings{0};
  std::size_t vocab_size{0};
  int bos_token_id{-1};
  int eos_token_id{-1};
  float rms_norm_eps{0.0F};
  float rope_theta{0.0F};
  bool attention_bias{false};
  bool tie_word_embeddings{false};
  std::string hidden_act;
  std::string source_dtype;

  [[nodiscard]] std::size_t head_dim() const { return hidden_size / num_attention_heads; }
  [[nodiscard]] std::size_t kv_dim() const { return head_dim() * num_key_value_heads; }
  [[nodiscard]] std::size_t query_groups() const {
    return num_attention_heads / num_key_value_heads;
  }

  static Result<ModelConfig> Load(const std::filesystem::path& path);
  [[nodiscard]] Result<void> ValidateFixedTinyLlama() const;
  [[nodiscard]] std::vector<TensorSpec> ExpectedTensors() const;
};

}  // namespace tlie
