#include <iostream>
#include <string>

#include "frontend_mlir.hpp"

namespace {

const char* kExecConvReluMlir = R"mlir(
module {
  func.func @tc_model(%arg0: tensor<1x2x8x8xf32>) -> tensor<1x2x7x7xf32> {
    %cst0 = arith.constant dense<[0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]> : tensor<2x2x2x2xf32>
    %t0 = tensor.empty() : tensor<1x2x7x7xf32>
    %t1 = linalg.conv_2d_nchw_fchw ins(%arg0, %cst0 : tensor<1x2x8x8xf32>, tensor<2x2x2x2xf32>) outs(%t0 : tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32>
    %cst1 = arith.constant dense<[0.25, -0.5]> : tensor<2xf32>
    %t2 = tensor.empty() : tensor<1x2x7x7xf32>
    %t3 = linalg.broadcast ins(%cst1 : tensor<2xf32>) outs(%t2 : tensor<1x2x7x7xf32>) dimensions = [1]
    %t4 = arith.addf %t1, %t3 : tensor<1x2x7x7xf32>
    %t5 = arith.constant dense<0.0> : tensor<1x2x7x7xf32>
    %t6 = arith.maximumf %t4, %t5 : tensor<1x2x7x7xf32>
    return %t6 : tensor<1x2x7x7xf32>
  }
}
)mlir";

bool Expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "TEST ERROR: " << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    tc::backend::FrontendMlirModule module;
    tc::backend::BackendDiagnostic diagnostic;
    if (!Expect(tc::backend::ParseFrontendMlirModule(
                    kExecConvReluMlir, "exec_conv_relu.mlir", module, diagnostic),
                "exec Conv Relu MLIR must parse")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(module.entry_name == "tc_model", "entry must be tc_model") ||
        !Expect(module.inputs.size() == 1, "module must have one input") ||
        !Expect(module.inputs[0].type.shape == std::vector<int64_t>({ 1, 2, 8, 8 }),
                "input shape must be parsed") ||
        !Expect(module.output_type.shape == std::vector<int64_t>({ 1, 2, 7, 7 }),
                "output shape must be parsed") ||
        !Expect(module.ops.size() == 10, "module operation count must match")) {
        return 1;
    }

    bool saw_conv = false;
    bool saw_broadcast = false;
    bool saw_relu = false;
    for (const auto& op : module.ops) {
        saw_conv = saw_conv ||
                   op.kind == tc::backend::FrontendMlirOpKind::kConv2DNchwFchw;
        saw_broadcast = saw_broadcast ||
                        op.kind == tc::backend::FrontendMlirOpKind::kBroadcast;
        saw_relu =
            saw_relu || op.kind == tc::backend::FrontendMlirOpKind::kMaximumF;
    }

    if (!Expect(saw_conv, "parser must detect linalg Conv") ||
        !Expect(saw_broadcast, "parser must detect linalg broadcast") ||
        !Expect(saw_relu, "parser must detect arith.maximumf")) {
        return 1;
    }

    tc::backend::FrontendMlirModule bad_module;
    if (!Expect(!tc::backend::ParseFrontendMlirModule(
                    "module { func.func @main() { return } }",
                    "bad.mlir",
                    bad_module,
                    diagnostic),
                "non tc_model entry must fail")) {
        return 1;
    }

    return 0;
}
