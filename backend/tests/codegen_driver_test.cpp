#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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

const char* kReshapeMlir = R"mlir(
module {
  func.func @tc_model(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %t0 = tensor.reshape %arg0 : tensor<2x3xf32> to tensor<3x2xf32>
    return %t0 : tensor<3x2xf32>
  }
}
)mlir";

const char* kPaddedConvMlir = R"mlir(
module {
  func.func @tc_model(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
    %cst0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]> : tensor<1x1x3x3xf32>
    %t0 = tensor.empty() : tensor<1x1x2x2xf32>
    %t1 = linalg.conv_2d_nchw_fchw {pads = [1, 1, 1, 1]} ins(%arg0, %cst0 : tensor<1x1x2x2xf32>, tensor<1x1x3x3xf32>) outs(%t0 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    return %t1 : tensor<1x1x2x2xf32>
  }
}
)mlir";

const char* kMaxPoolMlir = R"mlir(
module {
  func.func @tc_model(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
    %t0 = tensor.empty() : tensor<1x1x2x2xf32>
    %t1 = linalg.pooling_nchw_max {kernel = [2, 2], strides = [2, 2]} ins(%arg0 : tensor<1x1x4x4xf32>) outs(%t0 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    return %t1 : tensor<1x1x2x2xf32>
  }
}
)mlir";

const char* kNonExactFloatConstantMlir = R"mlir(
module {
  func.func @tc_model(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    %cst0 = arith.constant dense<1.018964529> : tensor<1xf32>
    %t0 = arith.addf %arg0, %cst0 : tensor<1xf32>
    return %t0 : tensor<1xf32>
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

bool ValidateLlvmIr(const std::string& llvm_ir,
                    const std::filesystem::path& ir_path)
{
    {
        std::ofstream out(ir_path, std::ios::out | std::ios::trunc);
        out << llvm_ir;
    }
    const std::string validate_cmd = "llc -filetype=null " + ir_path.string();
    return std::system(validate_cmd.c_str()) == 0;
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

    if (!Expect(ValidateLlvmIr(llvm_ir,
                               std::filesystem::temp_directory_path() /
                                   "tc_codegen_driver_test.ll"),
                "llc must accept generated LLVM IR")) {
        std::cerr << llvm_ir << '\n';
        return 1;
    }

    std::string reshape_ir;
    if (!Expect(
            tc::backend::EmitLlvmIrFromMlirText(
                kReshapeMlir, "reshape.mlir", {}, {}, reshape_ir, diagnostic),
            "reshape MLIR must emit LLVM IR")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(reshape_ir.find("tensor.reshape") == std::string::npos,
                "LLVM IR must not contain tensor.reshape") ||
        !Expect(ValidateLlvmIr(reshape_ir,
                               std::filesystem::temp_directory_path() /
                                   "tc_codegen_reshape_test.ll"),
                "llc must accept reshape LLVM IR")) {
        std::cerr << reshape_ir << '\n';
        return 1;
    }

    std::string padded_conv_ir;
    if (!Expect(tc::backend::EmitLlvmIrFromMlirText(kPaddedConvMlir,
                                                    "padded_conv.mlir",
                                                    {},
                                                    {},
                                                    padded_conv_ir,
                                                    diagnostic),
                "padded Conv MLIR must emit LLVM IR")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(padded_conv_ir.find("linalg.conv") == std::string::npos,
                "LLVM IR must not contain linalg Conv") ||
        !Expect(ValidateLlvmIr(padded_conv_ir,
                               std::filesystem::temp_directory_path() /
                                   "tc_codegen_padded_conv_test.ll"),
                "llc must accept padded Conv LLVM IR")) {
        std::cerr << padded_conv_ir << '\n';
        return 1;
    }

    std::string maxpool_ir;
    if (!Expect(
            tc::backend::EmitLlvmIrFromMlirText(
                kMaxPoolMlir, "maxpool.mlir", {}, {}, maxpool_ir, diagnostic),
            "MaxPool MLIR must emit LLVM IR")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(maxpool_ir.find("linalg.pooling") == std::string::npos,
                "LLVM IR must not contain linalg pooling") ||
        !Expect(maxpool_ir.find("fcmp ogt float") != std::string::npos,
                "MaxPool LLVM IR must compare window values") ||
        !Expect(ValidateLlvmIr(maxpool_ir,
                               std::filesystem::temp_directory_path() /
                                   "tc_codegen_maxpool_test.ll"),
                "llc must accept MaxPool LLVM IR")) {
        std::cerr << maxpool_ir << '\n';
        return 1;
    }

    std::string non_exact_constant_ir;
    if (!Expect(tc::backend::EmitLlvmIrFromMlirText(kNonExactFloatConstantMlir,
                                                    "non_exact_constant.mlir",
                                                    {},
                                                    {},
                                                    non_exact_constant_ir,
                                                    diagnostic),
                "non-exact float constant MLIR must emit LLVM IR")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(non_exact_constant_ir.find("0x") != std::string::npos,
                "non-exact float constants must be hex literals") ||
        !Expect(ValidateLlvmIr(non_exact_constant_ir,
                               std::filesystem::temp_directory_path() /
                                   "tc_codegen_non_exact_constant_test.ll"),
                "llc must accept non-exact float constant LLVM IR")) {
        std::cerr << non_exact_constant_ir << '\n';
        return 1;
    }

    std::string bad_ir;
    if (!Expect(!tc::backend::EmitLlvmIrFromMlirText(
                    "module { }", "bad.mlir", {}, {}, bad_ir, diagnostic),
                "invalid MLIR must fail LLVM IR emission")) {
        return 1;
    }

    std::string asm_text;
    if (!Expect(tc::backend::EmitAsmFromMlirText(kExecConvReluMlir,
                                                 "exec_conv_relu.mlir",
                                                 {},
                                                 {},
                                                 asm_text,
                                                 diagnostic),
                "exec Conv Relu MLIR must emit assembly")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(asm_text.find("tc_model") != std::string::npos,
                "assembly must contain tc_model symbol")) {
        std::cerr << asm_text << '\n';
        return 1;
    }

    return 0;
}
