#include "op_traits.hpp"

#include <array>

namespace {

using tc::frontend::OpArity;
using tc::frontend::OpKind;
using tc::frontend::OpTraits;

constexpr std::array<OpTraits, 8> kTraits = {
    OpTraits{ OpKind::kRelu, OpArity{ 1, 1 }, 1, false, true },
    OpTraits{ OpKind::kAdd, OpArity{ 2, 2 }, 1, false, true },
    OpTraits{ OpKind::kMul, OpArity{ 2, 2 }, 1, false, true },
    OpTraits{ OpKind::kConv, OpArity{ 2, 2 }, 1, true, true },
    OpTraits{ OpKind::kMatMul, OpArity{ 2, 2 }, 1, false, true },
    OpTraits{ OpKind::kReshape, OpArity{ 2, 2 }, 1, false, true },
    OpTraits{ OpKind::kTranspose, OpArity{ 1, 1 }, 1, true, true },
    OpTraits{ OpKind::kMaxPool, OpArity{ 1, 1 }, 1, true, true },
};

} // namespace

namespace tc::frontend {

const OpTraits* GetOpTraits(OpKind kind) noexcept
{
    for (const auto& traits : kTraits) {
        if (traits.kind == kind) {
            return &traits;
        }
    }
    return nullptr;
}

bool IsKnownOpKind(OpKind kind) noexcept
{
    return GetOpTraits(kind) != nullptr;
}

bool SupportsAttributes(OpKind kind) noexcept
{
    const OpTraits* traits = GetOpTraits(kind);
    return traits != nullptr && traits->supports_attributes;
}

bool IsStrictMlirSupported(OpKind kind) noexcept
{
    const OpTraits* traits = GetOpTraits(kind);
    return traits != nullptr && traits->strict_mlir_supported;
}

} // namespace tc::frontend
