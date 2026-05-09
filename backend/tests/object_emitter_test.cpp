#include <filesystem>
#include <iostream>
#include <string>

#include "object_emitter.hpp"

namespace {

const char* kTinyReluMlir = R"mlir(
module {
  func.func @tc_model(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %t0 = arith.constant dense<0.0> : tensor<1x2xf32>
    %t1 = arith.maximumf %arg0, %t0 : tensor<1x2xf32>
    return %t1 : tensor<1x2xf32>
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
    const std::filesystem::path object_path =
        std::filesystem::temp_directory_path() / "tc_object_emitter_test.o";
    std::error_code ec;
    std::filesystem::remove(object_path, ec);

    tc::backend::BackendDiagnostic diagnostic;
    if (!Expect(tc::backend::EmitObjectFromMlirText(kTinyReluMlir,
                                                    "tiny_relu.mlir",
                                                    {},
                                                    {},
                                                    object_path.string(),
                                                    diagnostic),
                "object emission must succeed")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(std::filesystem::exists(object_path),
                "object file must be created") ||
        !Expect(std::filesystem::file_size(object_path) > 0,
                "object file must not be empty")) {
        return 1;
    }

    std::filesystem::remove(object_path, ec);
    return 0;
}
