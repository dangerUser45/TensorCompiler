#include <filesystem>
#include <string>
#include <vector>

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

TEST(Metadata, EmitMetadataWritesDeterministicJson)
{
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_model_args.driver_path))
        << "driver not found: " << g_config.driver_model_args.driver_path;
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_model_args.model_path))
        << "model not found: " << g_config.driver_model_args.model_path;

    const std::filesystem::path metadata_output =
        tc::frontend::testutil::MakeTempPath("tc_metadata", ".json");
    const std::string command =
        tc::frontend::testutil::ShellQuote(
            g_config.driver_model_args.driver_path) +
        " " +
        tc::frontend::testutil::ShellQuote(
            g_config.driver_model_args.model_path) +
        " --emit-metadata=" +
        tc::frontend::testutil::ShellQuote(metadata_output.string());

    const auto result = tc::frontend::testutil::RunCommandWithCapturedOutput(
        command, "tc_metadata");
    EXPECT_EQ(result.exit_code, 0) << "driver output:\n" << result.output;
    ASSERT_TRUE(std::filesystem::exists(metadata_output))
        << "metadata output is missing, driver output:\n"
        << result.output;

    const std::string json =
        tc::frontend::testutil::ReadFileToString(metadata_output);
    const std::vector<std::string> expected = {
        "\"model_entry\": \"tc_model\"",
        "\"input_count\": 1",
        "\"name\": \"input_0\"",
        "\"shape\": [1, 2, 8, 8]",
        "\"byte_size\": 512",
        "\"output_count\": 1",
        "\"name\": \"output_0\"",
        "\"shape\": [1, 2, 7, 7]",
        "\"byte_size\": 392",
    };
    for (const auto& needle : expected) {
        EXPECT_NE(json.find(needle), std::string::npos)
            << "missing expected substring: " << needle << "\njson:\n"
            << json;
    }

    std::error_code ec;
    std::filesystem::remove(metadata_output, ec);
}

int main(int argc, char** argv)
{
    if (!ParseCustomArgs(&argc, argv)) {
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
