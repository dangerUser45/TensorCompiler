#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace tc::frontend::testutil {

struct DriverModelArgs final
{
    std::string driver_path;
    std::string model_path;
};

struct DriverModelsDirArgs final
{
    std::string driver_path;
    std::string models_dir;
};

struct CommandResult final
{
    int exit_code = -1;
    std::string output;
};

inline std::string ShellQuote(const std::string& value)
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

inline std::string ReadFileToString(const std::filesystem::path& path)
{
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

inline int DecodeExitCode(int system_status)
{
    if (system_status == -1) {
        return -1;
    }
    if (WIFEXITED(system_status)) {
        return WEXITSTATUS(system_status);
    }
    return -1;
}

inline std::string ToLower(std::string value)
{
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    return value;
}

inline bool ParseDriverModelArgs(int* argc,
                                 char** argv,
                                 DriverModelArgs& out_args)
{
    out_args = DriverModelArgs{};

    int write_idx = 1;
    for (int read_idx = 1; read_idx < *argc; ++read_idx) {
        const std::string arg = argv[read_idx];
        if (arg == "--driver") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --driver value is required\n";
                return false;
            }
            out_args.driver_path = argv[++read_idx];
            continue;
        }
        if (arg == "--model") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --model value is required\n";
                return false;
            }
            out_args.model_path = argv[++read_idx];
            continue;
        }
        argv[write_idx++] = argv[read_idx];
    }
    *argc = write_idx;

    if (out_args.driver_path.empty() || out_args.model_path.empty()) {
        std::cerr << "ERROR: both --driver and --model are required\n";
        return false;
    }
    return true;
}

inline bool ParseDriverModelsDirArgs(int* argc,
                                     char** argv,
                                     DriverModelsDirArgs& out_args)
{
    out_args = DriverModelsDirArgs{};

    int write_idx = 1;
    for (int read_idx = 1; read_idx < *argc; ++read_idx) {
        const std::string arg = argv[read_idx];
        if (arg == "--driver") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --driver value is required\n";
                return false;
            }
            out_args.driver_path = argv[++read_idx];
            continue;
        }
        if (arg == "--models-dir") {
            if (read_idx + 1 >= *argc) {
                std::cerr << "ERROR: --models-dir value is required\n";
                return false;
            }
            out_args.models_dir = argv[++read_idx];
            continue;
        }
        argv[write_idx++] = argv[read_idx];
    }
    *argc = write_idx;

    if (out_args.driver_path.empty() || out_args.models_dir.empty()) {
        std::cerr << "ERROR: both --driver and --models-dir are required\n";
        return false;
    }
    return true;
}

inline std::filesystem::path MakeTempPath(const std::string& prefix,
                                          const std::string& extension)
{
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (prefix + "_" + std::to_string(getpid()) + "_" +
            std::to_string(tick) + extension);
}

inline CommandResult RunCommandWithCapturedOutput(const std::string& command,
                                                  const std::string& log_prefix)
{
    const std::filesystem::path log_output = MakeTempPath(log_prefix, ".log");
    const std::string redirect_command =
        command + " >" + ShellQuote(log_output.string()) + " 2>&1";

    CommandResult result;
    const int system_status = std::system(redirect_command.c_str());
    result.exit_code = DecodeExitCode(system_status);
    result.output = ReadFileToString(log_output);

    std::error_code ec;
    std::filesystem::remove(log_output, ec);
    return result;
}

} // namespace tc::frontend::testutil
