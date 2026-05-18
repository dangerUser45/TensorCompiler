#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "test_driver_utils.hpp"

namespace {

tc::frontend::testutil::DriverModelArgs g_args;

bool ParseCustomArgs(int* argc, char** argv)
{
    return tc::frontend::testutil::ParseDriverModelArgs(argc, argv, g_args);
}

} // namespace

TEST(VerifyExec, RejectsNonExecutableModelWithSemanticStage)
{
    ASSERT_TRUE(std::filesystem::exists(g_args.driver_path))
        << "driver not found: " << g_args.driver_path;
    ASSERT_TRUE(std::filesystem::exists(g_args.model_path))
        << "model not found: " << g_args.model_path;

    const std::string command =
        tc::frontend::testutil::ShellQuote(g_args.driver_path) + " " +
        tc::frontend::testutil::ShellQuote(g_args.model_path) +
        " --verify-exec";

    const auto result = tc::frontend::testutil::RunCommandWithCapturedOutput(
        command, "tc_verify_exec");

    EXPECT_EQ(result.exit_code, 2) << result.output;
    EXPECT_NE(result.output.find("[semantic]"), std::string::npos)
        << result.output;
    EXPECT_NE(result.output.find("expects exactly 1 runtime input"),
              std::string::npos)
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
