#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <unistd.h>

#include <gtest/gtest.h>

#include "test_driver_utils.hpp"

namespace {

struct Config final
{
    tc::frontend::testutil::DriverModelArgs driver_model_args;
};

Config g_config;

bool ParseCustomArgs(int* argc, char** argv)
{
    return tc::frontend::testutil::ParseDriverModelArgs(
        argc, argv, g_config.driver_model_args);
}

} // namespace

TEST(UnsupportedOperator, FailsWithClearDiagnostic)
{
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_model_args.driver_path))
        << "driver not found: " << g_config.driver_model_args.driver_path;
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_model_args.model_path))
        << "model not found: " << g_config.driver_model_args.model_path;

    std::filesystem::path tmp_output =
        std::filesystem::temp_directory_path() /
        ("tc_unsupported_op_" + std::to_string(getpid()) + ".log");

    const std::string command =
        tc::frontend::testutil::ShellQuote(
            g_config.driver_model_args.driver_path) +
        " " +
        tc::frontend::testutil::ShellQuote(
            g_config.driver_model_args.model_path) +
        " >" + tc::frontend::testutil::ShellQuote(tmp_output.string()) +
        " 2>&1";

    const int system_status = std::system(command.c_str());
    const int exit_code = tc::frontend::testutil::DecodeExitCode(system_status);
    const std::string output =
        tc::frontend::testutil::ReadFileToString(tmp_output);
    std::error_code ec;
    std::filesystem::remove(tmp_output, ec);

    EXPECT_NE(exit_code, 0) << "expected failure for unsupported model";
    EXPECT_NE(
        tc::frontend::testutil::ToLower(output).find("unsupported operator"),
        std::string::npos)
        << "unexpected diagnostic output:\n"
        << output;
}

int main(int argc, char** argv)
{
    if (!ParseCustomArgs(&argc, argv)) {
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
