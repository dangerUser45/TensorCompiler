#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace {

struct Config final
{
    std::string driver_path;
    std::string model_path;
};

Config g_config;

std::string ShellQuote(const std::string& value)
{
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::string ReadFileToString(const std::filesystem::path& path)
{
    std::ifstream input(path);
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

int DecodeExitCode(int system_status)
{
    if (system_status == -1) {
        return -1;
    }
    if (WIFEXITED(system_status)) {
        return WEXITSTATUS(system_status);
    }
    return -1;
}

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

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
        if (arg == "--model") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --model value is required\n";
                return false;
            }
            g_config.model_path = argv[++read_idx];
            continue;
        }
        argv[write_idx++] = argv[read_idx];
    }
    *argc = write_idx;

    if (g_config.driver_path.empty() || g_config.model_path.empty()) {
        std::cerr << "ERROR: both --driver and --model are required\n";
        return false;
    }
    return true;
}

} // namespace

TEST(UnsupportedOperator, FailsWithClearDiagnostic)
{
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_path))
        << "driver not found: " << g_config.driver_path;
    ASSERT_TRUE(std::filesystem::exists(g_config.model_path))
        << "model not found: " << g_config.model_path;

    std::filesystem::path tmp_output =
        std::filesystem::temp_directory_path() /
        ("tc_unsupported_op_" + std::to_string(getpid()) + ".log");

    const std::string command =
        ShellQuote(g_config.driver_path) + " " + ShellQuote(g_config.model_path) +
        " >" + ShellQuote(tmp_output.string()) + " 2>&1";

    const int system_status = std::system(command.c_str());
    const int exit_code = DecodeExitCode(system_status);
    const std::string output = ReadFileToString(tmp_output);
    std::error_code ec;
    std::filesystem::remove(tmp_output, ec);

    EXPECT_NE(exit_code, 0) << "expected failure for unsupported model";
    EXPECT_NE(ToLower(output).find("unsupported operator"), std::string::npos)
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
