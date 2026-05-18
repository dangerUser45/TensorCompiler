#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "test_driver_utils.hpp"

namespace {

struct Config final
{
    tc::frontend::testutil::DriverModelArgs driver_model_args;
    std::vector<std::string> expected_substrings;
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
        if (arg == "--expect") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --expect value is required\n";
                return false;
            }
            g_config.expected_substrings.push_back(argv[++read_idx]);
            continue;
        }
        argv[write_idx++] = argv[read_idx];
    }
    *argc = write_idx;

    return true;
}

} // namespace

TEST(EmitMlir, WritesMlirFileForSupportedModel)
{
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_model_args.driver_path))
        << "driver not found: " << g_config.driver_model_args.driver_path;
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_model_args.model_path))
        << "model not found: " << g_config.driver_model_args.model_path;

    const std::filesystem::path mlir_output =
        tc::frontend::testutil::MakeTempPath("tc_emit_mlir", ".mlir");

    const std::string command =
        tc::frontend::testutil::ShellQuote(
            g_config.driver_model_args.driver_path) +
        " " +
        tc::frontend::testutil::ShellQuote(
            g_config.driver_model_args.model_path) +
        " --emit-mlir=" +
        tc::frontend::testutil::ShellQuote(mlir_output.string());

    const auto result = tc::frontend::testutil::RunCommandWithCapturedOutput(
        command, "tc_emit_mlir");

    EXPECT_EQ(result.exit_code, 0) << "driver output:\n" << result.output;
    ASSERT_TRUE(std::filesystem::exists(mlir_output))
        << "mlir output is missing, driver output:\n"
        << result.output;

    const std::string mlir_text =
        tc::frontend::testutil::ReadFileToString(mlir_output);
    const std::vector<std::string> default_expectations = {
        "module {", "func.func @tc_model("
    };
    const std::vector<std::string>& expectations =
        g_config.expected_substrings.empty() ? default_expectations
                                             : g_config.expected_substrings;
    for (const auto& needle : expectations) {
        EXPECT_NE(mlir_text.find(needle), std::string::npos)
            << "missing expected substring: " << needle << "\nmlir output:\n"
            << mlir_text;
    }

    std::error_code ec;
    std::filesystem::remove(mlir_output, ec);
}

int main(int argc, char** argv)
{
    if (!ParseCustomArgs(&argc, argv)) {
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
