#pragma once

#include "op_kind.hpp"

#include <cstddef>

namespace tc::frontend {

struct OpArity final
{
    std::size_t min = 0;
    std::size_t max = 0;
};

struct OpTraits final
{
    OpKind kind = OpKind::kUnknown;
    OpArity inputs;
    std::size_t outputs = 0;
    bool supports_attributes = false;
    bool strict_mlir_supported = false;
};

const OpTraits* GetOpTraits(OpKind kind) noexcept;
bool IsKnownOpKind(OpKind kind) noexcept;
bool SupportsAttributes(OpKind kind) noexcept;
bool IsStrictMlirSupported(OpKind kind) noexcept;

} // namespace tc::frontend
