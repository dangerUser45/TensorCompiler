#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mlir_pipeline.hpp"

namespace tc::backend {

struct TensorType final
{
    std::vector<int64_t> shape;
    std::string element_type;

    std::size_t element_count() const noexcept;
};

struct MlirValueDecl final
{
    std::string name;
    TensorType type;
};

enum class FrontendMlirOpKind
{
    kConstant,
    kTensorEmpty,
    kBroadcast,
    kAddF,
    kMulF,
    kMaximumF,
    kMatMul,
    kReshape,
    kMaxPoolNchw,
    kTranspose,
    kConv2DNchwFchw,
    kReturn
};

struct FrontendMlirOp final
{
    FrontendMlirOpKind kind = FrontendMlirOpKind::kReturn;
    std::string result;
    std::vector<std::string> operands;
    TensorType result_type;
    std::vector<float> constant_values;
    bool constant_is_splat = false;
    std::vector<int64_t> dimensions;
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
};

struct FrontendMlirModule final
{
    std::string entry_name;
    std::vector<MlirValueDecl> inputs;
    TensorType output_type;
    std::vector<FrontendMlirOp> ops;
};

std::string FormatTensorType(const TensorType& type);

bool ParseFrontendMlirModule(const std::string& source,
                             const std::string& source_name,
                             FrontendMlirModule& out_module,
                             BackendDiagnostic& out_diagnostic);

} // namespace tc::backend
