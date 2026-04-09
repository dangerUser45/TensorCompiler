#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace {

struct Config final
{
    std::string driver_path;
    std::string model_path;
    std::vector<std::string> expected_substrings;
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
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
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

    if (g_config.driver_path.empty() || g_config.model_path.empty()) {
        std::cerr << "ERROR: both --driver and --model are required\n";
        return false;
    }
    return true;
}

} // namespace

TEST(EmitMlir, WritesMlirFileForSupportedModel)
{
    ASSERT_TRUE(std::filesystem::exists(g_config.driver_path))
        << "driver not found: " << g_config.driver_path;
    ASSERT_TRUE(std::filesystem::exists(g_config.model_path))
        << "model not found: " << g_config.model_path;

    const std::filesystem::path mlir_output =
        std::filesystem::temp_directory_path() /
        ("tc_emit_mlir_" + std::to_string(getpid()) + ".mlir");
    const std::filesystem::path log_output =
        std::filesystem::temp_directory_path() /
        ("tc_emit_mlir_" + std::to_string(getpid()) + ".log");

    const std::string command =
        ShellQuote(g_config.driver_path) + " " +
        ShellQuote(g_config.model_path) +
        " --emit-mlir=" + ShellQuote(mlir_output.string()) + " >" +
        ShellQuote(log_output.string()) + " 2>&1";

    const int system_status = std::system(command.c_str());
    const int exit_code = DecodeExitCode(system_status);
    const std::string log = ReadFileToString(log_output);

    EXPECT_EQ(exit_code, 0) << "driver output:\n" << log;
    ASSERT_TRUE(std::filesystem::exists(mlir_output))
        << "mlir output is missing, driver output:\n"
        << log;

    const std::string mlir_text = ReadFileToString(mlir_output);
    const std::vector<std::string> default_expectations = {
        "module {", "func.func @main("
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
    std::filesystem::remove(log_output, ec);
}

int main(int argc, char** argv)
{
    if (!ParseCustomArgs(&argc, argv)) {
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
