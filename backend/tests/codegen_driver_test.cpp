#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "codegen_driver.hpp"

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
    tc::backend::BackendDiagnostic diagnostic;
    std::string llvm_ir;
    if (!Expect(tc::backend::EmitLlvmIrFromMlirText(kExecConvReluMlir,
                                                    "exec_conv_relu.mlir",
                                                    {},
                                                    {},
                                                    llvm_ir,
                                                    diagnostic),
                "exec Conv Relu MLIR must emit LLVM IR")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(llvm_ir.find("define void @tc_model(") != std::string::npos,
                "LLVM IR must define tc_model") ||
        !Expect(llvm_ir.find("fmul float") != std::string::npos,
                "LLVM IR must contain fmul") ||
        !Expect(llvm_ir.find("fadd float") != std::string::npos,
                "LLVM IR must contain fadd") ||
        !Expect(llvm_ir.find("fcmp ogt float") != std::string::npos,
                "LLVM IR must contain Relu comparison") ||
        !Expect(llvm_ir.find("linalg.") == std::string::npos,
                "LLVM IR must not contain linalg ops") ||
        !Expect(llvm_ir.find("arith.") == std::string::npos,
                "LLVM IR must not contain arith ops")) {
        std::cerr << llvm_ir << '\n';
        return 1;
    }

    const std::filesystem::path ir_path =
        std::filesystem::temp_directory_path() / "tc_codegen_driver_test.ll";
    {
        std::ofstream out(ir_path, std::ios::out | std::ios::trunc);
        out << llvm_ir;
    }
    const std::string validate_cmd =
        "llc -filetype=null " + ir_path.string();
    if (!Expect(std::system(validate_cmd.c_str()) == 0,
                "llc must accept generated LLVM IR")) {
        std::cerr << llvm_ir << '\n';
        return 1;
    }

    std::string bad_ir;
    if (!Expect(!tc::backend::EmitLlvmIrFromMlirText("module { }",
                                                     "bad.mlir",
                                                     {},
                                                     {},
                                                     bad_ir,
                                                     diagnostic),
                "invalid MLIR must fail LLVM IR emission")) {
        return 1;
    }

    return 0;
}
