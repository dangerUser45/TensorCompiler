#pragma once

#include <string_view>

namespace tc::frontend {

enum class OpKind
{
    kUnknown = 0,
    kRelu,
    kAdd,
    kMul,
    kConv,
    kMatMul,
    kReshape,
    kTranspose,
    kMaxPool
};

OpKind OpKindFromString(std::string_view op_type) noexcept;

const char* ToString(OpKind op_kind) noexcept;

} // namespace tc::frontend
