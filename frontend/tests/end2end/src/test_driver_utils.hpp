#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <sys/wait.h>

namespace tc::frontend::testutil {

struct DriverModelArgs final
{
    std::string driver_path;
    std::string model_path;
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

} // namespace tc::frontend::testutil
