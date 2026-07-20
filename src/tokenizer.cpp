#include "tlie/tokenizer.hpp"

#include <sentencepiece_processor.h>

namespace tlie {
namespace {

bool IsValidUtf8(const std::string_view text) {
  std::size_t index = 0;
  while (index < text.size()) {
    const auto first = static_cast<unsigned char>(text[index]);
    std::size_t continuation = 0;
    std::uint32_t codepoint = 0;
    if (first <= 0x7fU) {
      ++index;
      continue;
    }
    if ((first & 0xe0U) == 0xc0U) {
      continuation = 1;
      codepoint = first & 0x1fU;
      if (codepoint < 2) {
        return false;
      }
    } else if ((first & 0xf0U) == 0xe0U) {
      continuation = 2;
      codepoint = first & 0x0fU;
    } else if ((first & 0xf8U) == 0xf0U) {
      continuation = 3;
      codepoint = first & 0x07U;
    } else {
      return false;
    }
    if (index + continuation >= text.size()) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation; ++offset) {
      const auto byte = static_cast<unsigned char>(text[index + offset]);
      if ((byte & 0xc0U) != 0x80U) {
        return false;
      }
      codepoint = (codepoint << 6U) | (byte & 0x3fU);
    }
    if ((continuation == 2 && codepoint < 0x800U) || (continuation == 3 && codepoint < 0x10000U) ||
        codepoint > 0x10ffffU || (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
      return false;
    }
    index += continuation + 1;
  }
  return true;
}

}  // namespace

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer::Tokenizer(std::unique_ptr<sentencepiece::SentencePieceProcessor> processor,
                     const std::size_t max_input_bytes)
    : processor_(std::move(processor)), max_input_bytes_(max_input_bytes) {}

Result<Tokenizer> Tokenizer::Load(const std::filesystem::path& path,
                                  const std::size_t max_input_bytes) {
  if (max_input_bytes == 0) {
    return Result<Tokenizer>::Failure(
        {ErrorCode::kInvalidArgument, "tokenizer max input length must be positive"});
  }
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  const auto status = processor->Load(path.string());
  if (!status.ok()) {
    return Result<Tokenizer>::Failure(
        {ErrorCode::kTokenizer, "unable to load tokenizer model: " + status.ToString()});
  }
  if (processor->bos_id() != 1 || processor->eos_id() != 2 || processor->unk_id() != 0 ||
      processor->GetPieceSize() != 32000) {
    return Result<Tokenizer>::Failure(
        {ErrorCode::kTokenizer, "tokenizer special IDs or vocabulary differ from pinned model"});
  }
  return Result<Tokenizer>::Success(Tokenizer(std::move(processor), max_input_bytes));
}

Result<std::vector<int>> Tokenizer::Encode(const std::string_view text, const bool add_bos,
                                           const bool add_eos) const {
  if (!processor_) {
    return Result<std::vector<int>>::Failure(
        {ErrorCode::kTokenizer, "tokenizer is not initialized"});
  }
  if (text.size() > max_input_bytes_) {
    return Result<std::vector<int>>::Failure(
        {ErrorCode::kOutOfBounds, "tokenizer input exceeds configured byte limit"});
  }
  if (!IsValidUtf8(text)) {
    return Result<std::vector<int>>::Failure(
        {ErrorCode::kTokenizer, "tokenizer input is not valid UTF-8"});
  }
  std::vector<int> pieces;
  const auto status = processor_->Encode(std::string(text), &pieces);
  if (!status.ok()) {
    return Result<std::vector<int>>::Failure(
        {ErrorCode::kTokenizer, "tokenizer encode failed: " + status.ToString()});
  }
  if (add_bos) {
    pieces.insert(pieces.begin(), processor_->bos_id());
  }
  if (add_eos) {
    pieces.push_back(processor_->eos_id());
  }
  return Result<std::vector<int>>::Success(std::move(pieces));
}

Result<std::string> Tokenizer::Decode(const std::vector<int>& ids) const {
  if (!processor_) {
    return Result<std::string>::Failure({ErrorCode::kTokenizer, "tokenizer is not initialized"});
  }
  for (const int id : ids) {
    if (id < 0 || id >= processor_->GetPieceSize()) {
      return Result<std::string>::Failure(
          {ErrorCode::kOutOfBounds, "token ID is outside the tokenizer vocabulary"});
    }
  }
  std::string text;
  const auto status = processor_->Decode(ids, &text);
  if (!status.ok()) {
    return Result<std::string>::Failure(
        {ErrorCode::kTokenizer, "tokenizer decode failed: " + status.ToString()});
  }
  return Result<std::string>::Success(std::move(text));
}

int Tokenizer::bos_id() const { return processor_ ? processor_->bos_id() : -1; }
int Tokenizer::eos_id() const { return processor_ ? processor_->eos_id() : -1; }
int Tokenizer::unk_id() const { return processor_ ? processor_->unk_id() : -1; }
int Tokenizer::vocab_size() const { return processor_ ? processor_->GetPieceSize() : 0; }

}  // namespace tlie
