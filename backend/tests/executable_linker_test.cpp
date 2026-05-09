#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "executable_linker.hpp"
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

const char* kTinyMetadata = R"json({
  "model_entry": "tc_model",
  "input_count": 1,
  "inputs": [
    {
      "name": "input",
      "dtype": "float32",
      "shape": [1, 2],
      "layout": "",
      "byte_size": 8
    }
  ],
  "output_count": 1,
  "outputs": [
    {
      "name": "output",
      "dtype": "float32",
      "shape": [1, 2],
      "layout": "",
      "byte_size": 8
    }
  ]
}
)json";

bool Expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "TEST ERROR: " << message << '\n';
        return false;
    }
    return true;
}

bool WriteFloats(const std::filesystem::path& path,
                 const std::vector<float>& values)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    return out.good();
}

bool ReadFloats(const std::filesystem::path& path, std::vector<float>& values)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        return false;
    }
    values.resize(static_cast<std::size_t>(size) / sizeof(float));
    in.read(reinterpret_cast<char*>(values.data()), size);
    return in.good() || in.gcount() == size;
}

} // namespace

int main()
{
    const std::filesystem::path temp = std::filesystem::temp_directory_path();
    const std::filesystem::path object_path = temp / "tc_linker_model.o";
    const std::filesystem::path metadata_path = temp / "tc_linker_metadata.json";
    const std::filesystem::path executable_path = temp / "tc_linker_runner";
    const std::filesystem::path input_path = temp / "tc_linker_input.f32";
    const std::filesystem::path output_path = temp / "tc_linker_output.f32";

    std::error_code ec;
    std::filesystem::remove(object_path, ec);
    std::filesystem::remove(metadata_path, ec);
    std::filesystem::remove(executable_path, ec);
    std::filesystem::remove(input_path, ec);
    std::filesystem::remove(output_path, ec);

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

    {
        std::ofstream metadata_out(metadata_path, std::ios::out | std::ios::trunc);
        metadata_out << kTinyMetadata;
    }

    if (!Expect(tc::backend::LinkExecutableWithRuntime(object_path.string(),
                                                       metadata_path.string(),
                                                       executable_path.string(),
                                                       diagnostic),
                "linking executable must succeed")) {
        std::cerr << tc::backend::FormatBackendDiagnostic(diagnostic) << '\n';
        return 1;
    }

    if (!Expect(std::filesystem::exists(executable_path),
                "executable must be created")) {
        return 1;
    }

    if (!Expect(WriteFloats(input_path, { -1.0F, 2.0F }),
                "input file must be written")) {
        return 1;
    }

    const std::string command = executable_path.string() + " --input " +
                                input_path.string() + " --output " +
                                output_path.string();
    if (!Expect(std::system(command.c_str()) == 0, "executable must run")) {
        return 1;
    }

    std::vector<float> output;
    if (!Expect(ReadFloats(output_path, output), "output file must be readable") ||
        !Expect(output.size() == 2, "output must contain two floats") ||
        !Expect(std::fabs(output[0] - 0.0F) < 1.0e-6F,
                "Relu output[0] must be 0") ||
        !Expect(std::fabs(output[1] - 2.0F) < 1.0e-6F,
                "Relu output[1] must be 2")) {
        return 1;
    }

    std::filesystem::remove(object_path, ec);
    std::filesystem::remove(metadata_path, ec);
    std::filesystem::remove(executable_path, ec);
    std::filesystem::remove(input_path, ec);
    std::filesystem::remove(output_path, ec);
    return 0;
}
