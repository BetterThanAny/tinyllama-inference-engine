#include "tlie/weight_store.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sstream>

namespace tlie {
namespace {

constexpr char kMagic[8] = {'T', 'L', 'I', 'E', 'W', 'G', 'T', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kAlignment = 64;

Result<WeightStore> LoadFailure(const ErrorCode code, const std::string& message) {
  return Result<WeightStore>::Failure({code, message});
}

bool CheckedMultiply(const std::uint64_t left, const std::uint64_t right, std::uint64_t* result) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

class Reader {
 public:
  Reader(const std::byte* data, const std::size_t size) : data_(data), size_(size) {}

  template <typename T>
  bool Read(T* value) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (remaining() < sizeof(T)) {
      return false;
    }
    std::memcpy(value, data_ + offset_, sizeof(T));
    offset_ += sizeof(T);
    return true;
  }

  bool ReadBytes(void* output, const std::size_t count) {
    if (remaining() < count) {
      return false;
    }
    std::memcpy(output, data_ + offset_, count);
    offset_ += count;
    return true;
  }

  bool SkipToAlignment(const std::size_t alignment) {
    const std::size_t padding = (alignment - (offset_ % alignment)) % alignment;
    if (remaining() < padding) {
      return false;
    }
    offset_ += padding;
    return true;
  }

  [[nodiscard]] const std::byte* current() const { return data_ + offset_; }
  [[nodiscard]] std::size_t remaining() const { return size_ - offset_; }
  [[nodiscard]] std::size_t offset() const { return offset_; }
  bool Skip(const std::size_t count) {
    if (remaining() < count) {
      return false;
    }
    offset_ += count;
    return true;
  }

 private:
  const std::byte* data_;
  std::size_t size_;
  std::size_t offset_{0};
};

}  // namespace

WeightStore::~WeightStore() { Reset(); }

WeightStore::WeightStore(WeightStore&& other) noexcept { *this = std::move(other); }

WeightStore& WeightStore::operator=(WeightStore&& other) noexcept {
  if (this != &other) {
    Reset();
    file_descriptor_ = other.file_descriptor_;
    mapping_ = other.mapping_;
    mapping_size_ = other.mapping_size_;
    source_checksum_ = other.source_checksum_;
    tensors_ = std::move(other.tensors_);
    other.file_descriptor_ = -1;
    other.mapping_ = nullptr;
    other.mapping_size_ = 0;
    other.tensors_.clear();
  }
  return *this;
}

void WeightStore::Reset() noexcept {
  tensors_.clear();
  if (mapping_ != nullptr) {
    munmap(const_cast<std::byte*>(mapping_), mapping_size_);
  }
  if (file_descriptor_ >= 0) {
    close(file_descriptor_);
  }
  file_descriptor_ = -1;
  mapping_ = nullptr;
  mapping_size_ = 0;
}

Result<WeightStore> WeightStore::Load(const std::filesystem::path& path,
                                      const WeightLoadOptions& options) {
  if constexpr (std::endian::native != std::endian::little) {
    return LoadFailure(ErrorCode::kInvalidFormat, "TLIEWGT requires a little-endian host");
  }
  const int descriptor = open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    return LoadFailure(ErrorCode::kIo, "unable to open weight file: " + path.string() + ": " +
                                           std::strerror(errno));
  }
  struct stat metadata{};
  if (fstat(descriptor, &metadata) != 0 || metadata.st_size <= 0) {
    close(descriptor);
    return LoadFailure(ErrorCode::kIo, "unable to stat non-empty weight file: " + path.string());
  }
  const auto file_bytes = static_cast<std::uint64_t>(metadata.st_size);
  if (file_bytes > options.max_file_bytes) {
    close(descriptor);
    return LoadFailure(ErrorCode::kOutOfMemory,
                       "weight file exceeds configured memory budget before mapping");
  }
  if (file_bytes > std::numeric_limits<std::size_t>::max()) {
    close(descriptor);
    return LoadFailure(ErrorCode::kOutOfMemory, "weight file cannot be addressed by this process");
  }
  void* mapping =
      mmap(nullptr, static_cast<std::size_t>(file_bytes), PROT_READ, MAP_PRIVATE, descriptor, 0);
  if (mapping == MAP_FAILED) {
    close(descriptor);
    return LoadFailure(ErrorCode::kOutOfMemory,
                       "unable to map weight file: " + std::string(std::strerror(errno)));
  }

  WeightStore store;
  store.file_descriptor_ = descriptor;
  store.mapping_ = static_cast<const std::byte*>(mapping);
  store.mapping_size_ = static_cast<std::size_t>(file_bytes);

  if (!options.expected_file_sha256.empty()) {
    Sha256Digest expected{};
    if (!ParseSha256(options.expected_file_sha256, &expected)) {
      return LoadFailure(ErrorCode::kInvalidArgument, "expected file SHA-256 is malformed");
    }
    const auto entire_file = std::span<const std::byte>(store.mapping_, store.mapping_size_);
    if (ComputeSha256(entire_file) != expected) {
      return LoadFailure(ErrorCode::kChecksumMismatch,
                         "weight file SHA-256 does not match the pinned manifest");
    }
  }

  Reader reader(store.mapping_, store.mapping_size_);
  char magic[sizeof(kMagic)]{};
  std::uint32_t version = 0;
  std::uint32_t tensor_count = 0;
  if (!reader.ReadBytes(magic, sizeof(magic)) || !reader.Read(&version) ||
      !reader.Read(&tensor_count) ||
      !reader.ReadBytes(store.source_checksum_.data(), store.source_checksum_.size())) {
    return LoadFailure(ErrorCode::kInvalidFormat, "truncated TLIEWGT header");
  }
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 || version != kVersion) {
    return LoadFailure(ErrorCode::kInvalidFormat, "invalid TLIEWGT magic or version");
  }
  if (!options.expected_source_sha256.empty()) {
    Sha256Digest expected{};
    if (!ParseSha256(options.expected_source_sha256, &expected)) {
      return LoadFailure(ErrorCode::kInvalidArgument, "expected source SHA-256 is malformed");
    }
    if (expected != store.source_checksum_) {
      return LoadFailure(ErrorCode::kChecksumMismatch,
                         "source model SHA-256 does not match the pinned manifest");
    }
  }
  if (tensor_count == 0 || tensor_count > 100000) {
    return LoadFailure(ErrorCode::kInvalidFormat, "implausible TLIEWGT tensor count");
  }

  try {
    store.tensors_.reserve(tensor_count);
    for (std::uint32_t index = 0; index < tensor_count; ++index) {
      std::uint16_t name_length = 0;
      std::uint8_t raw_dtype = 0;
      std::uint8_t rank = 0;
      std::uint64_t byte_count = 0;
      Sha256Digest checksum{};
      if (!reader.Read(&name_length) || !reader.Read(&raw_dtype) || !reader.Read(&rank) ||
          !reader.Read(&byte_count) || !reader.ReadBytes(checksum.data(), checksum.size())) {
        return LoadFailure(ErrorCode::kInvalidFormat, "truncated TLIEWGT tensor record");
      }
      if (name_length == 0 || name_length > 4096 || rank == 0 || rank > 8) {
        return LoadFailure(ErrorCode::kInvalidFormat, "invalid tensor name length or rank");
      }
      if (raw_dtype != static_cast<std::uint8_t>(DType::kFloat32)) {
        return LoadFailure(ErrorCode::kInvalidDtype, "TLIEWGT M1 accepts only float32 tensors");
      }
      std::vector<std::size_t> shape;
      shape.reserve(rank);
      std::uint64_t elements = 1;
      for (std::uint8_t dimension_index = 0; dimension_index < rank; ++dimension_index) {
        std::uint64_t dimension = 0;
        if (!reader.Read(&dimension) || dimension == 0 ||
            dimension > std::numeric_limits<std::size_t>::max()) {
          return LoadFailure(ErrorCode::kInvalidShape, "invalid TLIEWGT tensor dimension");
        }
        if (!CheckedMultiply(elements, dimension, &elements)) {
          return LoadFailure(ErrorCode::kInvalidShape, "TLIEWGT tensor element count overflows");
        }
        shape.push_back(static_cast<std::size_t>(dimension));
      }
      std::string name(name_length, '\0');
      if (!reader.ReadBytes(name.data(), name.size()) || !reader.SkipToAlignment(kAlignment)) {
        return LoadFailure(ErrorCode::kInvalidFormat, "truncated TLIEWGT tensor metadata");
      }
      std::uint64_t expected_bytes = 0;
      if (!CheckedMultiply(elements, sizeof(float), &expected_bytes) ||
          expected_bytes != byte_count) {
        return LoadFailure(ErrorCode::kInvalidShape,
                           "tensor byte count does not match shape for " + name);
      }
      if (byte_count > reader.remaining()) {
        return LoadFailure(ErrorCode::kInvalidFormat, "truncated tensor payload for " + name);
      }
      const auto* float_data = reinterpret_cast<const float*>(reader.current());
      const auto payload =
          std::span<const std::byte>(reader.current(), static_cast<std::size_t>(byte_count));
      if (options.verify_tensor_checksums && ComputeSha256(payload) != checksum) {
        return LoadFailure(ErrorCode::kChecksumMismatch, "tensor checksum mismatch for " + name);
      }
      TensorView view{name, DType::kFloat32, std::move(shape),
                      std::span<const float>(float_data, static_cast<std::size_t>(elements)),
                      checksum};
      if (!store.tensors_.emplace(name, std::move(view)).second) {
        return LoadFailure(ErrorCode::kInvalidFormat, "duplicate tensor name: " + name);
      }
      if (!reader.Skip(static_cast<std::size_t>(byte_count))) {
        return LoadFailure(ErrorCode::kInvalidFormat, "truncated tensor payload for " + name);
      }
    }
  } catch (const std::bad_alloc&) {
    return LoadFailure(ErrorCode::kOutOfMemory, "allocation failed while indexing TLIEWGT");
  }
  if (reader.offset() != store.mapping_size_) {
    return LoadFailure(ErrorCode::kInvalidFormat, "unexpected trailing bytes in TLIEWGT");
  }
  return Result<WeightStore>::Success(std::move(store));
}

Result<TensorView> WeightStore::Get(const std::string_view name) const {
  const auto iterator = tensors_.find(std::string(name));
  if (iterator == tensors_.end()) {
    return Result<TensorView>::Failure(
        {ErrorCode::kInvalidShape, "required tensor is missing: " + std::string(name)});
  }
  return Result<TensorView>::Success(iterator->second);
}

Result<void> WeightStore::Validate(const std::vector<TensorSpec>& expected) const {
  if (tensors_.size() != expected.size()) {
    return Result<void>::Failure({ErrorCode::kInvalidShape, "tensor count mismatch: expected " +
                                                                std::to_string(expected.size()) +
                                                                ", got " +
                                                                std::to_string(tensors_.size())});
  }
  for (const auto& spec : expected) {
    const auto tensor = Get(spec.name);
    if (!tensor.ok()) {
      return Result<void>::Failure(tensor.error());
    }
    if (tensor.value().dtype != DType::kFloat32) {
      return Result<void>::Failure(
          {ErrorCode::kInvalidDtype, "tensor has unsupported dtype: " + spec.name});
    }
    if (tensor.value().shape != spec.shape) {
      return Result<void>::Failure(
          {ErrorCode::kInvalidShape, "tensor has unexpected shape: " + spec.name});
    }
  }
  return Result<void>::Success();
}

}  // namespace tlie
