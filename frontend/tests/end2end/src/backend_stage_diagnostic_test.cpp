#include <filesystem>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "test_driver_utils.hpp"

namespace {

struct Config final
{
    tc::frontend::testutil::DriverModelArgs driver_model_args;
    std::string output_path;
};

Config g_config;

bool ParseCustomArgs(int* argc, char** argv)
{
    if (!tc::frontend::testutil::ParseDriverModelArgs(
            argc, argv, g_config.driver_model_args)) {
        return false;
    }

    int write_idx = 1;
    for (int read_idx = 1; read_idx < *argc; ++read_idx) {
        const std::string arg = argv[read_idx];
        if (arg == "--output") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --output value is required\n";
                return false;
            }
            g_config.output_path = argv[++read_idx];
            continue;
        }
        argv[write_idx++] = argv[read_idx];
    }
    *argc = write_idx;

    if (g_config.output_path.empty()) {
        std::cerr << "ERROR: --output is required\n";
        return false;
    }
    return true;
}

} // namespace

TEST(BackendStageDiagnostic, EmitMlirFailureReportsBackendStage)
{
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_model_args.driver_path))
        << "driver not found: " << g_config.driver_model_args.driver_path;
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_model_args.model_path))
        << "model not found: " << g_config.driver_model_args.model_path;

    const std::string command =
        tc::frontend::testutil::ShellQuote(
            g_config.driver_model_args.driver_path) +
        " " +
        tc::frontend::testutil::ShellQuote(
            g_config.driver_model_args.model_path) +
        " --emit-mlir=" +
        tc::frontend::testutil::ShellQuote(g_config.output_path);
    const auto result = tc::frontend::testutil::RunCommandWithCapturedOutput(
        command, "tc_backend_diag");

    EXPECT_NE(result.exit_code, 0) << "expected emit-mlir failure";
    EXPECT_NE(tc::frontend::testutil::ToLower(result.output).find("backend"),
              std::string::npos)
        << "missing backend-stage diagnostic marker:\n"
        << result.output;
}

int main(int argc, char** argv)
{
    if (!ParseCustomArgs(&argc, argv)) {
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
