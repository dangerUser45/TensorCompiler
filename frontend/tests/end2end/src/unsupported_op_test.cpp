#include <filesystem>
#include <iostream>
#include <string>

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

    const std::string command = tc::frontend::testutil::ShellQuote(
                                    g_config.driver_model_args.driver_path) +
                                " " +
                                tc::frontend::testutil::ShellQuote(
                                    g_config.driver_model_args.model_path);
    const auto result = tc::frontend::testutil::RunCommandWithCapturedOutput(
        command, "tc_unsupported_op");

    EXPECT_NE(result.exit_code, 0) << "expected failure for unsupported model";
    EXPECT_NE(tc::frontend::testutil::ToLower(result.output)
                  .find("unsupported operator"),
              std::string::npos)
        << "unexpected diagnostic output:\n"
        << result.output;
    EXPECT_NE(tc::frontend::testutil::ToLower(result.output).find("import"),
              std::string::npos)
        << "missing import-stage diagnostic marker:\n"
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
