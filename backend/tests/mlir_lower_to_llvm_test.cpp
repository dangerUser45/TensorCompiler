#include <iostream>
#include <string>

#include "mlir_context.hpp"
#include "mlir_pipeline.hpp"

namespace {

const char* kScalarAddModule = R"mlir(
module {
  func.func @add(%lhs: i32, %rhs: i32) -> i32 {
    %sum = arith.addi %lhs, %rhs : i32
    return %sum : i32
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
    if (!Expect(tc::backend::IsMlirAvailable(),
                "MLIR support must be enabled for mlir_lower_to_llvm_test")) {
        return 1;
    }

    tc::backend::ParsedMlirModule pipeline_module;
    tc::backend::BackendDiagnostic diagnostic;
    if (!Expect(tc::backend::ParseMlirModuleFromString(kScalarAddModule,
                                                       "scalar_add.mlir",
                                                       pipeline_module,
                                                       diagnostic),
                "scalar add MLIR must parse successfully")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(tc::backend::RunMlirPassPipeline(
                    pipeline_module, "canonicalize,cse", diagnostic),
                "textual MLIR pass pipeline must run successfully")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    const std::string optimized_module = pipeline_module.Dump();
    if (!Expect(optimized_module.find("func.func @add") != std::string::npos,
                "optimized module must still contain the original func op")) {
        std::cerr << optimized_module << '\n';
        return 1;
    }

    tc::backend::ParsedMlirModule llvm_module;
    if (!Expect(
            tc::backend::ParseMlirModuleFromString(
                kScalarAddModule, "scalar_add.mlir", llvm_module, diagnostic),
            "scalar add MLIR must reparse before LLVM lowering")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(
            tc::backend::LowerMlirModuleToLlvmDialect(llvm_module, diagnostic),
            "lowering MLIR module to LLVM dialect must succeed")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    const std::string llvm_dump = llvm_module.Dump();
    if (!Expect(llvm_dump.find("llvm.func @add") != std::string::npos,
                "LLVM lowering must produce llvm.func") ||
        !Expect(llvm_dump.find("llvm.add") != std::string::npos,
                "LLVM lowering must produce llvm.add") ||
        !Expect(llvm_dump.find("llvm.return") != std::string::npos,
                "LLVM lowering must produce llvm.return") ||
        !Expect(llvm_dump.find("func.func") == std::string::npos,
                "LLVM lowering must remove func.func") ||
        !Expect(llvm_dump.find("arith.addi") == std::string::npos,
                "LLVM lowering must remove arith.addi")) {
        std::cerr << llvm_dump << '\n';
        return 1;
    }

    return 0;
}
