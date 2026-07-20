#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tlie/result.hpp"

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace tlie {

class Tokenizer {
 public:
  Tokenizer();
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;

  static Result<Tokenizer> Load(const std::filesystem::path& path,
                                std::size_t max_input_bytes = 1U << 20U);
  [[nodiscard]] Result<std::vector<int>> Encode(std::string_view text, bool add_bos = true,
                                                bool add_eos = false) const;
  [[nodiscard]] Result<std::string> Decode(const std::vector<int>& ids) const;
  [[nodiscard]] int bos_id() const;
  [[nodiscard]] int eos_id() const;
  [[nodiscard]] int unk_id() const;
  [[nodiscard]] int vocab_size() const;

 private:
  explicit Tokenizer(std::unique_ptr<sentencepiece::SentencePieceProcessor> processor,
                     std::size_t max_input_bytes);

  std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
  std::size_t max_input_bytes_{0};
};

}  // namespace tlie
