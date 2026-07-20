#include "tlie/cuda/weight_store.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

namespace tlie::cuda {
namespace {

constexpr char kMagic[8] = {'T', 'L', 'I', 'E', 'W', 'G', 'T', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint8_t kFloat32Dtype = 1;
constexpr std::uint8_t kFloat16Dtype = 2;
constexpr std::uint8_t kInt8Dtype = 3;
constexpr std::size_t kAlignment = 64;

Result<CudaWeightStore> Failure(const ErrorCode code, const std::string& message) {
  return Result<CudaWeightStore>::Failure({code, message});
}

bool CheckedMultiply(const std::uint64_t left, const std::uint64_t right, std::uint64_t* result) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

class Mapping {
 public:
  ~Mapping() {
    if (data_ != nullptr) {
      munmap(const_cast<std::byte*>(data_), size_);
    }
    if (descriptor_ >= 0) {
      close(descriptor_);
    }
  }

  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  Mapping() = default;

  Result<void> Open(const std::filesystem::path& path, const std::uint64_t max_file_bytes) {
    descriptor_ = open(path.c_str(), O_RDONLY);
    if (descriptor_ < 0) {
      return Result<void>::Failure(
          {ErrorCode::kIo,
           "unable to open FP16 weight file: " + path.string() + ": " + std::strerror(errno)});
    }
    struct stat metadata{};
    if (fstat(descriptor_, &metadata) != 0 || metadata.st_size <= 0) {
      return Result<void>::Failure(
          {ErrorCode::kIo, "unable to stat non-empty FP16 weight file: " + path.string()});
    }
    const auto bytes = static_cast<std::uint64_t>(metadata.st_size);
    if (bytes > max_file_bytes || bytes > std::numeric_limits<std::size_t>::max()) {
      return Result<void>::Failure(
          {ErrorCode::kOutOfMemory, "FP16 weight file exceeds the configured mapping budget"});
    }
    size_ = static_cast<std::size_t>(bytes);
    void* mapping = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
    if (mapping == MAP_FAILED) {
      data_ = nullptr;
      return Result<void>::Failure(
          {ErrorCode::kOutOfMemory,
           "unable to map FP16 weight file: " + std::string(std::strerror(errno))});
    }
    data_ = static_cast<const std::byte*>(mapping);
    return Result<void>::Success();
  }

  [[nodiscard]] const std::byte* data() const { return data_; }
  [[nodiscard]] std::size_t size() const { return size_; }

 private:
  int descriptor_{-1};
  const std::byte* data_{nullptr};
  std::size_t size_{0};
};

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

  bool AlignAndValidateZero(const std::size_t alignment) {
    const std::size_t padding = (alignment - offset_ % alignment) % alignment;
    if (remaining() < padding) {
      return false;
    }
    for (std::size_t index = 0; index < padding; ++index) {
      if (data_[offset_ + index] != std::byte{0}) {
        return false;
      }
    }
    offset_ += padding;
    return true;
  }

  bool Skip(const std::size_t count) {
    if (remaining() < count) {
      return false;
    }
    offset_ += count;
    return true;
  }

  [[nodiscard]] const std::byte* current() const { return data_ + offset_; }
  [[nodiscard]] std::size_t remaining() const { return size_ - offset_; }
  [[nodiscard]] std::size_t offset() const { return offset_; }

 private:
  const std::byte* data_;
  std::size_t size_;
  std::size_t offset_{0};
};

}  // namespace

CudaWeightStore::~CudaWeightStore() { Reset(); }

CudaWeightStore::CudaWeightStore(CudaWeightStore&& other) noexcept { *this = std::move(other); }

CudaWeightStore& CudaWeightStore::operator=(CudaWeightStore&& other) noexcept {
  if (this != &other) {
    Reset();
    source_checksum_ = other.source_checksum_;
    allocations_ = std::move(other.allocations_);
    tensors_ = std::move(other.tensors_);
    other.allocations_.clear();
    other.tensors_.clear();
  }
  return *this;
}

void CudaWeightStore::Reset() noexcept {
  tensors_.clear();
  for (void* allocation : allocations_) {
    if (allocation != nullptr) {
      cudaFree(allocation);
    }
  }
  allocations_.clear();
}

Result<CudaWeightStore> CudaWeightStore::Load(const std::filesystem::path& path,
                                              const WeightLoadOptions& options) {
  if constexpr (std::endian::native != std::endian::little) {
    return Failure(ErrorCode::kInvalidFormat, "FP16 TLIEWGT requires a little-endian host");
  }
  if (options.expected_file_sha256.empty() || options.expected_source_sha256.empty()) {
    return Failure(ErrorCode::kInvalidArgument,
                   "FP16 CUDA weights require externally pinned file and source SHA-256 values");
  }
  Mapping mapping;
  auto mapped = mapping.Open(path, options.max_file_bytes);
  if (!mapped.ok()) {
    return Result<CudaWeightStore>::Failure(mapped.error());
  }
  Sha256Digest expected_file{};
  if (!ParseSha256(options.expected_file_sha256, &expected_file)) {
    return Failure(ErrorCode::kInvalidArgument, "expected FP16 file SHA-256 is malformed");
  }
  if (ComputeSha256(std::span<const std::byte>(mapping.data(), mapping.size())) != expected_file) {
    return Failure(ErrorCode::kChecksumMismatch,
                   "FP16 weight file SHA-256 does not match the pinned manifest");
  }

  Reader reader(mapping.data(), mapping.size());
  char magic[sizeof(kMagic)]{};
  std::uint32_t version = 0;
  std::uint32_t tensor_count = 0;
  CudaWeightStore store;
  if (!reader.ReadBytes(magic, sizeof(magic)) || !reader.Read(&version) ||
      !reader.Read(&tensor_count) ||
      !reader.ReadBytes(store.source_checksum_.data(), store.source_checksum_.size())) {
    return Failure(ErrorCode::kInvalidFormat, "truncated FP16 TLIEWGT header");
  }
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 || version != kVersion || tensor_count == 0 ||
      tensor_count > 100000) {
    return Failure(ErrorCode::kInvalidFormat, "invalid FP16 TLIEWGT magic, version, or count");
  }
  Sha256Digest expected_source{};
  if (!ParseSha256(options.expected_source_sha256, &expected_source)) {
    return Failure(ErrorCode::kInvalidArgument, "expected source SHA-256 is malformed");
  }
  if (store.source_checksum_ != expected_source) {
    return Failure(ErrorCode::kChecksumMismatch,
                   "FP16 source model SHA-256 does not match the pinned manifest");
  }

  try {
    store.allocations_.reserve(tensor_count);
    store.tensors_.reserve(tensor_count);
    for (std::uint32_t index = 0; index < tensor_count; ++index) {
      std::uint16_t name_length = 0;
      std::uint8_t dtype = 0;
      std::uint8_t rank = 0;
      std::uint64_t byte_count = 0;
      Sha256Digest checksum{};
      if (!reader.Read(&name_length) || !reader.Read(&dtype) || !reader.Read(&rank) ||
          !reader.Read(&byte_count) || !reader.ReadBytes(checksum.data(), checksum.size())) {
        return Failure(ErrorCode::kInvalidFormat, "truncated FP16 tensor record");
      }
      if (name_length == 0 || name_length > 4096 || rank == 0 || rank > 8) {
        return Failure(ErrorCode::kInvalidFormat, "invalid FP16 tensor name length or rank");
      }
      if (dtype != kFloat16Dtype) {
        return Failure(ErrorCode::kInvalidDtype, "CUDA TLIEWGT accepts only float16 tensors");
      }
      std::vector<std::size_t> shape;
      shape.reserve(rank);
      std::uint64_t elements = 1;
      for (std::uint8_t dimension_index = 0; dimension_index < rank; ++dimension_index) {
        std::uint64_t dimension = 0;
        if (!reader.Read(&dimension) || dimension == 0 ||
            dimension > std::numeric_limits<std::size_t>::max() ||
            !CheckedMultiply(elements, dimension, &elements)) {
          return Failure(ErrorCode::kInvalidShape, "invalid or overflowing FP16 tensor shape");
        }
        shape.push_back(static_cast<std::size_t>(dimension));
      }
      std::string name(name_length, '\0');
      if (!reader.ReadBytes(name.data(), name.size()) || !reader.AlignAndValidateZero(kAlignment)) {
        return Failure(ErrorCode::kInvalidFormat, "invalid FP16 tensor metadata or padding");
      }
      std::uint64_t expected_bytes = 0;
      if (!CheckedMultiply(elements, sizeof(__half), &expected_bytes) ||
          expected_bytes != byte_count) {
        return Failure(ErrorCode::kInvalidShape,
                       "FP16 tensor byte count does not match shape for " + name);
      }
      if (byte_count > reader.remaining()) {
        return Failure(ErrorCode::kInvalidFormat, "truncated FP16 tensor payload for " + name);
      }
      const auto payload =
          std::span<const std::byte>(reader.current(), static_cast<std::size_t>(byte_count));
      if (options.verify_tensor_checksums && ComputeSha256(payload) != checksum) {
        return Failure(ErrorCode::kChecksumMismatch, "FP16 tensor checksum mismatch for " + name);
      }
      void* device = nullptr;
      cudaError_t cuda_status = cudaMalloc(&device, static_cast<std::size_t>(byte_count));
      if (cuda_status != cudaSuccess) {
        const ErrorCode code =
            cuda_status == cudaErrorMemoryAllocation ? ErrorCode::kOutOfMemory : ErrorCode::kCuda;
        return Failure(
            code, "CUDA allocation failed for " + name + ": " + cudaGetErrorString(cuda_status));
      }
      store.allocations_.push_back(device);
      cuda_status = cudaMemcpy(device, payload.data(), payload.size(), cudaMemcpyHostToDevice);
      if (cuda_status != cudaSuccess) {
        return Failure(ErrorCode::kCuda, "CUDA weight upload failed for " + name + ": " +
                                             cudaGetErrorString(cuda_status));
      }
      CudaTensorView view{name,
                          std::move(shape),
                          CudaWeightDType::kFloat16,
                          static_cast<const __half*>(device),
                          nullptr,
                          nullptr,
                          static_cast<std::size_t>(elements),
                          checksum};
      if (!store.tensors_.emplace(name, std::move(view)).second) {
        return Failure(ErrorCode::kInvalidFormat, "duplicate FP16 tensor name: " + name);
      }
      if (!reader.Skip(static_cast<std::size_t>(byte_count))) {
        return Failure(ErrorCode::kInvalidFormat, "truncated FP16 tensor payload for " + name);
      }
    }
  } catch (const std::bad_alloc&) {
    return Failure(ErrorCode::kOutOfMemory, "allocation failed while indexing FP16 TLIEWGT");
  }
  if (reader.offset() != mapping.size()) {
    return Failure(ErrorCode::kInvalidFormat, "unexpected trailing bytes in FP16 TLIEWGT");
  }
  return Result<CudaWeightStore>::Success(std::move(store));
}

Result<CudaWeightStore> CudaWeightStore::LoadInt8(const std::filesystem::path& path,
                                                  const WeightLoadOptions& options) {
  if constexpr (std::endian::native != std::endian::little) {
    return Failure(ErrorCode::kInvalidFormat, "INT8 TLIEWGT requires a little-endian host");
  }
  if (options.expected_file_sha256.empty() || options.expected_source_sha256.empty()) {
    return Failure(ErrorCode::kInvalidArgument,
                   "INT8 CUDA weights require externally pinned file and source SHA-256 values");
  }
  Mapping mapping;
  auto mapped = mapping.Open(path, options.max_file_bytes);
  if (!mapped.ok()) {
    return Result<CudaWeightStore>::Failure(mapped.error());
  }
  Sha256Digest expected_file{};
  if (!ParseSha256(options.expected_file_sha256, &expected_file)) {
    return Failure(ErrorCode::kInvalidArgument, "expected INT8 file SHA-256 is malformed");
  }
  if (ComputeSha256(std::span<const std::byte>(mapping.data(), mapping.size())) != expected_file) {
    return Failure(ErrorCode::kChecksumMismatch,
                   "INT8 weight file SHA-256 does not match the pinned manifest");
  }

  Reader reader(mapping.data(), mapping.size());
  char magic[sizeof(kMagic)]{};
  std::uint32_t version = 0;
  std::uint32_t record_count = 0;
  CudaWeightStore store;
  if (!reader.ReadBytes(magic, sizeof(magic)) || !reader.Read(&version) ||
      !reader.Read(&record_count) ||
      !reader.ReadBytes(store.source_checksum_.data(), store.source_checksum_.size())) {
    return Failure(ErrorCode::kInvalidFormat, "truncated INT8 TLIEWGT header");
  }
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 || version != kVersion || record_count == 0 ||
      record_count > 100000) {
    return Failure(ErrorCode::kInvalidFormat, "invalid INT8 TLIEWGT magic, version, or count");
  }
  Sha256Digest expected_source{};
  if (!ParseSha256(options.expected_source_sha256, &expected_source)) {
    return Failure(ErrorCode::kInvalidArgument, "expected source SHA-256 is malformed");
  }
  if (store.source_checksum_ != expected_source) {
    return Failure(ErrorCode::kChecksumMismatch,
                   "INT8 source model SHA-256 does not match the pinned manifest");
  }

  struct RawRecord {
    std::vector<std::size_t> shape;
    std::uint8_t dtype{0};
    const void* values{nullptr};
    std::size_t elements{0};
    Sha256Digest checksum{};
  };
  std::unordered_map<std::string, RawRecord> records;
  try {
    store.allocations_.reserve(record_count);
    records.reserve(record_count);
    for (std::uint32_t index = 0; index < record_count; ++index) {
      std::uint16_t name_length = 0;
      std::uint8_t dtype = 0;
      std::uint8_t rank = 0;
      std::uint64_t byte_count = 0;
      Sha256Digest checksum{};
      if (!reader.Read(&name_length) || !reader.Read(&dtype) || !reader.Read(&rank) ||
          !reader.Read(&byte_count) || !reader.ReadBytes(checksum.data(), checksum.size())) {
        return Failure(ErrorCode::kInvalidFormat, "truncated INT8 tensor record");
      }
      if (name_length == 0 || name_length > 4096 || rank == 0 || rank > 8) {
        return Failure(ErrorCode::kInvalidFormat, "invalid INT8 tensor name length or rank");
      }
      if (dtype != kFloat32Dtype && dtype != kFloat16Dtype && dtype != kInt8Dtype) {
        return Failure(ErrorCode::kInvalidDtype, "INT8 TLIEWGT contains an unsupported dtype");
      }
      std::vector<std::size_t> shape;
      shape.reserve(rank);
      std::uint64_t elements = 1;
      for (std::uint8_t dimension_index = 0; dimension_index < rank; ++dimension_index) {
        std::uint64_t dimension = 0;
        if (!reader.Read(&dimension) || dimension == 0 ||
            dimension > std::numeric_limits<std::size_t>::max() ||
            !CheckedMultiply(elements, dimension, &elements)) {
          return Failure(ErrorCode::kInvalidShape, "invalid or overflowing INT8 tensor shape");
        }
        shape.push_back(static_cast<std::size_t>(dimension));
      }
      std::string name(name_length, '\0');
      if (!reader.ReadBytes(name.data(), name.size()) || !reader.AlignAndValidateZero(kAlignment)) {
        return Failure(ErrorCode::kInvalidFormat, "invalid INT8 tensor metadata or padding");
      }
      const std::uint64_t element_bytes =
          dtype == kFloat32Dtype ? sizeof(float) : (dtype == kFloat16Dtype ? sizeof(__half) : 1U);
      std::uint64_t expected_bytes = 0;
      if (!CheckedMultiply(elements, element_bytes, &expected_bytes) ||
          expected_bytes != byte_count) {
        return Failure(ErrorCode::kInvalidShape,
                       "INT8 tensor byte count does not match shape for " + name);
      }
      if (byte_count > reader.remaining()) {
        return Failure(ErrorCode::kInvalidFormat, "truncated INT8 tensor payload for " + name);
      }
      const auto payload =
          std::span<const std::byte>(reader.current(), static_cast<std::size_t>(byte_count));
      if (options.verify_tensor_checksums && ComputeSha256(payload) != checksum) {
        return Failure(ErrorCode::kChecksumMismatch, "INT8 tensor checksum mismatch for " + name);
      }
      void* device = nullptr;
      cudaError_t cuda_status = cudaMalloc(&device, static_cast<std::size_t>(byte_count));
      if (cuda_status != cudaSuccess) {
        const ErrorCode code =
            cuda_status == cudaErrorMemoryAllocation ? ErrorCode::kOutOfMemory : ErrorCode::kCuda;
        return Failure(code, "CUDA INT8 allocation failed for " + name + ": " +
                                 cudaGetErrorString(cuda_status));
      }
      store.allocations_.push_back(device);
      cuda_status = cudaMemcpy(device, payload.data(), payload.size(), cudaMemcpyHostToDevice);
      if (cuda_status != cudaSuccess) {
        return Failure(ErrorCode::kCuda, "CUDA INT8 upload failed for " + name + ": " +
                                             cudaGetErrorString(cuda_status));
      }
      if (!records
               .emplace(name, RawRecord{std::move(shape), dtype, device,
                                        static_cast<std::size_t>(elements), checksum})
               .second) {
        return Failure(ErrorCode::kInvalidFormat, "duplicate INT8 tensor name: " + name);
      }
      if (!reader.Skip(static_cast<std::size_t>(byte_count))) {
        return Failure(ErrorCode::kInvalidFormat, "truncated INT8 tensor payload for " + name);
      }
    }
  } catch (const std::bad_alloc&) {
    return Failure(ErrorCode::kOutOfMemory, "allocation failed while indexing INT8 TLIEWGT");
  }
  if (reader.offset() != mapping.size()) {
    return Failure(ErrorCode::kInvalidFormat, "unexpected trailing bytes in INT8 TLIEWGT");
  }

  store.tensors_.reserve(records.size());
  for (const auto& [name, record] : records) {
    if (name.ends_with(".scale")) {
      continue;
    }
    CudaTensorView view;
    view.name = name;
    view.shape = record.shape;
    view.elements = record.elements;
    view.checksum = record.checksum;
    if (record.dtype == kFloat16Dtype) {
      if (record.shape.size() != 1 && record.shape.size() != 2) {
        return Failure(ErrorCode::kInvalidDtype,
                       "mixed INT8 model permits FP16 only for rank-1 or rank-2 tensors: " + name);
      }
      view.dtype = CudaWeightDType::kFloat16;
      view.values = static_cast<const __half*>(record.values);
    } else if (record.dtype == kInt8Dtype) {
      if (record.shape.size() != 2) {
        return Failure(ErrorCode::kInvalidDtype,
                       "per-channel INT8 weights must be rank-2: " + name);
      }
      const auto scale = records.find(name + ".scale");
      if (scale == records.end() || scale->second.dtype != kFloat32Dtype ||
          scale->second.shape != std::vector<std::size_t>{record.shape[0]}) {
        return Failure(ErrorCode::kInvalidShape,
                       "INT8 per-output-channel scale is missing or invalid for " + name);
      }
      view.dtype = CudaWeightDType::kInt8PerOutputChannel;
      view.quantized_values = static_cast<const std::int8_t*>(record.values);
      view.scales = static_cast<const float*>(scale->second.values);
    } else {
      return Failure(ErrorCode::kInvalidDtype,
                     "standalone FP32 tensors are not model weights: " + name);
    }
    store.tensors_.emplace(name, std::move(view));
  }
  return Result<CudaWeightStore>::Success(std::move(store));
}

Result<CudaTensorView> CudaWeightStore::Get(const std::string_view name) const {
  const auto iterator = tensors_.find(std::string(name));
  if (iterator == tensors_.end()) {
    return Result<CudaTensorView>::Failure(
        {ErrorCode::kInvalidShape, "required CUDA tensor is missing: " + std::string(name)});
  }
  return Result<CudaTensorView>::Success(iterator->second);
}

Result<void> CudaWeightStore::Validate(const std::vector<TensorSpec>& expected) const {
  if (tensors_.size() != expected.size()) {
    return Result<void>::Failure(
        {ErrorCode::kInvalidShape, "CUDA tensor count does not match model config"});
  }
  for (const auto& spec : expected) {
    const auto tensor = Get(spec.name);
    if (!tensor.ok()) {
      return Result<void>::Failure(tensor.error());
    }
    if (tensor.value().shape != spec.shape) {
      return Result<void>::Failure(
          {ErrorCode::kInvalidShape, "CUDA tensor has unexpected shape: " + spec.name});
    }
  }
  return Result<void>::Success();
}

}  // namespace tlie::cuda
