#include <iostream>
#include <string>

#include "mlir_context.hpp"
#include "mlir_pipeline.hpp"

namespace {

const char* kValidModule = R"mlir(
module {
  func.func @identity(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %c0 = arith.constant 0.0 : f32
    return %arg0 : tensor<2xf32>
  }
}
)mlir";

const char* kInvalidModule = R"mlir(
module {
  func.func @broken(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %c0 = arith.constant 0.0 :
    return %arg0 : tensor<2xf32>
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
                "MLIR support must be enabled for mlir_parse_test")) {
        return 1;
    }

    tc::backend::ParsedMlirModule parsed_module;
    tc::backend::BackendDiagnostic diagnostic;
    if (!Expect(tc::backend::ParseMlirModuleFromString(
                    kValidModule, "valid_identity.mlir", parsed_module,
                    diagnostic),
                "valid MLIR module should parse successfully")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    const std::string dumped_module = parsed_module.Dump();
    if (!Expect(!parsed_module.empty(), "parsed module must not be empty") ||
        !Expect(dumped_module.find("func.func @identity") != std::string::npos,
                "dumped module must contain parsed function name") ||
        !Expect(dumped_module.find("arith.constant") != std::string::npos,
                "dumped module must contain parsed arith op")) {
        return 1;
    }

    parsed_module.clear();
    diagnostic = tc::backend::BackendDiagnostic{};
    if (!Expect(!tc::backend::ParseMlirModuleFromString(
                    kInvalidModule, "invalid_identity.mlir", parsed_module,
                    diagnostic),
                "invalid MLIR module must fail to parse")) {
        return 1;
    }

    const std::string formatted_diagnostic =
        tc::backend::FormatBackendDiagnostic(diagnostic);
    if (!Expect(!diagnostic.message.empty(),
                "invalid parse must return a non-empty diagnostic") ||
        !Expect(formatted_diagnostic.find("backend:") == 0,
                "diagnostic must be marked as backend stage") ||
        !Expect(formatted_diagnostic.find("failed to parse MLIR module") !=
                    std::string::npos,
                "diagnostic must mention parse failure")) {
        std::cerr << formatted_diagnostic << '\n';
        return 1;
    }

    return 0;
}
