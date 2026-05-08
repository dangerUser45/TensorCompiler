#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "test_driver_utils.hpp"

namespace {

struct ModelCase final
{
    std::string model_file;
    std::vector<std::string> mlir_expectations;
    bool expect_emit_mlir = true;
};

bool Contains(std::string_view text, std::string_view fragment)
{
    return text.find(fragment) != std::string_view::npos;
}

struct Config final
{
    tc::frontend::testutil::DriverModelsDirArgs driver_models_dir_args;
};

Config g_config;

bool ParseCustomArgs(int* argc, char** argv)
{
    return tc::frontend::testutil::ParseDriverModelsDirArgs(
        argc, argv, g_config.driver_models_dir_args);
}

} // namespace

TEST(VerifyEmitMatrix, MandatoryModelsPassVerifyAndEmitMlir)
{
    ASSERT_TRUE(
        std::filesystem::exists(g_config.driver_models_dir_args.driver_path))
        << "driver not found: " << g_config.driver_models_dir_args.driver_path;
    ASSERT_TRUE(
        std::filesystem::exists(g_config.driver_models_dir_args.models_dir))
        << "models dir not found: "
        << g_config.driver_models_dir_args.models_dir;

    const std::vector<ModelCase> cases = {
        { "single_relu.onnx", { "arith.maximumf" } },
        { "add.onnx", { "arith.addf", "op=Add" } },
        { "matmul_2d.onnx", { "linalg.matmul", "op=MatMul" } },
        { "matmul_addmm.onnx", { "linalg.matmul", "arith.addf" } },
        { "conv_2d_nchw_fchw.onnx", {}, false },
    };

    for (const auto& test_case : cases) {
        const std::filesystem::path model_path =
            std::filesystem::path(g_config.driver_models_dir_args.models_dir) /
            test_case.model_file;
        ASSERT_TRUE(std::filesystem::exists(model_path))
            << "model not found: " << model_path.string();

        const std::string verify_cmd =
            tc::frontend::testutil::ShellQuote(
                g_config.driver_models_dir_args.driver_path) +
            " " + tc::frontend::testutil::ShellQuote(model_path.string()) +
            " --verify";
        const auto verify_result =
            tc::frontend::testutil::RunCommandWithCapturedOutput(
                verify_cmd, "tc_matrix_verify");
        EXPECT_EQ(verify_result.exit_code, 0)
            << "verify failed for model: " << test_case.model_file
            << "\noutput:\n"
            << verify_result.output;

        const std::filesystem::path mlir_output =
            tc::frontend::testutil::MakeTempPath(
                "tc_matrix_" + test_case.model_file, ".mlir");
        const std::string emit_cmd =
            tc::frontend::testutil::ShellQuote(
                g_config.driver_models_dir_args.driver_path) +
            " " + tc::frontend::testutil::ShellQuote(model_path.string()) +
            " --emit-mlir=" +
            tc::frontend::testutil::ShellQuote(mlir_output.string());

        const auto emit_result =
            tc::frontend::testutil::RunCommandWithCapturedOutput(
                emit_cmd, "tc_matrix_emit");
        if (!test_case.expect_emit_mlir) {
            EXPECT_NE(emit_result.exit_code, 0)
                << "emit-mlir unexpectedly passed for model: "
                << test_case.model_file;
            EXPECT_TRUE(Contains(emit_result.output, "unsupported graph"))
                << "missing unsupported graph diagnostic for model: "
                << test_case.model_file << "\noutput:\n"
                << emit_result.output;
            continue;
        }
        EXPECT_EQ(emit_result.exit_code, 0)
            << "emit-mlir failed for model: " << test_case.model_file
            << "\noutput:\n"
            << emit_result.output;
        ASSERT_TRUE(std::filesystem::exists(mlir_output))
            << "missing mlir output for model: " << test_case.model_file
            << "\noutput:\n"
            << emit_result.output;

        const std::string mlir_text =
            tc::frontend::testutil::ReadFileToString(mlir_output);
        for (const auto& expected : test_case.mlir_expectations) {
            EXPECT_NE(mlir_text.find(expected), std::string::npos)
                << "missing MLIR substring '" << expected
                << "' for model: " << test_case.model_file;
        }

        std::error_code ec;
        std::filesystem::remove(mlir_output, ec);
    }
}

int main(int argc, char** argv)
{
    if (!ParseCustomArgs(&argc, argv)) {
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
