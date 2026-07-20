#pragma once

#include <string_view>

namespace tlie {

inline constexpr std::string_view kPinnedSourceWeightSha256 =
    "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933";
inline constexpr std::string_view kPinnedConvertedWeightSha256 =
    "277f8aa3757b47b208c02682e851590bc42e819279251ac186406bd67b05beaf";
inline constexpr std::string_view kPinnedCudaFp16WeightSha256 =
    "1038f532a16e316f1953cf721cb2a54783cd1327b6a22e45ce71dcc2c574ab63";
inline constexpr std::string_view kPinnedCudaInt8WeightSha256 =
    "3e8b49987c37b3df0a6b785c2874c96619a5d63b378796448e41b56943082ccd";

}  // namespace tlie
