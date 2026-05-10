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

inline OpKind OpKindFromString(std::string_view op_type) noexcept
{
    if (op_type == "Relu") {
        return OpKind::kRelu;
    }
    if (op_type == "Add") {
        return OpKind::kAdd;
    }
    if (op_type == "Mul") {
        return OpKind::kMul;
    }
    if (op_type == "Conv") {
        return OpKind::kConv;
    }
    if (op_type == "MatMul") {
        return OpKind::kMatMul;
    }
    if (op_type == "Reshape") {
        return OpKind::kReshape;
    }
    if (op_type == "Transpose") {
        return OpKind::kTranspose;
    }
    if (op_type == "MaxPool") {
        return OpKind::kMaxPool;
    }
    return OpKind::kUnknown;
}

inline const char* ToString(OpKind op_kind) noexcept
{
    switch (op_kind) {
        case OpKind::kRelu:
            return "Relu";
        case OpKind::kAdd:
            return "Add";
        case OpKind::kMul:
            return "Mul";
        case OpKind::kConv:
            return "Conv";
        case OpKind::kMatMul:
            return "MatMul";
        case OpKind::kReshape:
            return "Reshape";
        case OpKind::kTranspose:
            return "Transpose";
        case OpKind::kMaxPool:
            return "MaxPool";
        case OpKind::kUnknown:
            return "Unknown";
    }
    return "Unknown";
}

} // namespace tc::frontend
