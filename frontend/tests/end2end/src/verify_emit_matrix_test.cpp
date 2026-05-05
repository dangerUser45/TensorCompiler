#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "test_driver_utils.hpp"

namespace {

struct ModelCase final
{
    std::string model_file;
    std::vector<std::string> mlir_expectations;
};

struct Config final
{
    std::string driver_path;
    std::string models_dir;
};

Config g_config;

bool ParseCustomArgs(int* argc, char** argv)
{
    int write_idx = 1;
    for (int read_idx = 1; read_idx < *argc; ++read_idx) {
        const std::string arg = argv[read_idx];
        if (arg == "--driver") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --driver value is required\n";
                return false;
            }
            g_config.driver_path = argv[++read_idx];
            continue;
        }
        if (arg == "--models-dir") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --models-dir value is required\n";
                return false;
            }
            g_config.models_dir = argv[++read_idx];
            continue;
        }
        argv[write_idx++] = argv[read_idx];
    }
    *argc = write_idx;

    if (g_config.driver_path.empty() || g_config.models_dir.empty()) {
        std::cerr << "ERROR: both --driver and --models-dir are required\n";
        return false;
    }
    return true;
}

int RunCommand(const std::string& command, std::string& out_log)
{
    const std::filesystem::path log_output =
        std::filesystem::temp_directory_path() /
        ("tc_matrix_" + std::to_string(getpid()) + ".log");

    const int system_status = std::system(
        (command + " >" +
         tc::frontend::testutil::ShellQuote(log_output.string()) + " 2>&1")
            .c_str());
    const int exit_code = tc::frontend::testutil::DecodeExitCode(system_status);
    out_log = tc::frontend::testutil::ReadFileToString(log_output);
    std::error_code ec;
    std::filesystem::remove(log_output, ec);
    return exit_code;
}

} // namespace

TEST(VerifyEmitMatrix, MandatoryModelsPassVerifyAndEmitMlir)
{
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_path))
        << "driver not found: " << g_config.driver_path;
    ASSERT_TRUE(std::filesystem::exists(g_config.models_dir))
        << "models dir not found: " << g_config.models_dir;

    const std::vector<ModelCase> cases = {
        { "single_relu.onnx", { "arith.maximumf" } },
        { "add.onnx", { "arith.addf", "op=Add" } },
        { "matmul_2d.onnx", { "linalg.matmul", "op=MatMul" } },
        { "matmul_addmm.onnx", { "linalg.matmul", "arith.addf" } },
    };

    for (const auto& test_case : cases) {
        const std::filesystem::path model_path =
            std::filesystem::path(g_config.models_dir) / test_case.model_file;
        ASSERT_TRUE(std::filesystem::exists(model_path))
            << "model not found: " << model_path.string();

        const std::string verify_cmd =
            tc::frontend::testutil::ShellQuote(g_config.driver_path) + " " +
            tc::frontend::testutil::ShellQuote(model_path.string()) +
            " --verify";
        std::string verify_log;
        const int verify_exit = RunCommand(verify_cmd, verify_log);
        EXPECT_EQ(verify_exit, 0)
            << "verify failed for model: " << test_case.model_file
            << "\noutput:\n"
            << verify_log;

        const std::filesystem::path mlir_output =
            std::filesystem::temp_directory_path() /
            ("tc_matrix_" + test_case.model_file + "_" +
             std::to_string(getpid()) + ".mlir");
        const std::string emit_cmd =
            tc::frontend::testutil::ShellQuote(g_config.driver_path) + " " +
            tc::frontend::testutil::ShellQuote(model_path.string()) +
            " --emit-mlir=" +
            tc::frontend::testutil::ShellQuote(mlir_output.string());

        std::string emit_log;
        const int emit_exit = RunCommand(emit_cmd, emit_log);
        EXPECT_EQ(emit_exit, 0)
            << "emit-mlir failed for model: " << test_case.model_file
            << "\noutput:\n"
            << emit_log;
        ASSERT_TRUE(std::filesystem::exists(mlir_output))
            << "missing mlir output for model: " << test_case.model_file
            << "\noutput:\n"
            << emit_log;

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
