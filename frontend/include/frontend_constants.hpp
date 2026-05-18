#pragma once

#include <string_view>

namespace tc::frontend {

inline constexpr std::string_view kSyntheticAddSuffix = ".add";
inline constexpr std::string_view kSyntheticMatMulSuffix = ".matmul";
inline constexpr std::string_view kGemmMatMulNodePrefix = "gemm_matmul_";
inline constexpr std::string_view kGemmAddNodePrefix = "gemm_add_";
inline constexpr std::string_view kConvAddNodePrefix = "conv_add_";
inline constexpr std::string_view kGemmMatMulOutputSuffix = "__gemm_matmul_";
inline constexpr std::string_view kConvOutputSuffix = "__conv_";

inline constexpr std::string_view kAutoPadNotSet = "NOTSET";
inline constexpr std::string_view kAutoPadSameUpper = "SAME_UPPER";

} // namespace tc::frontend
