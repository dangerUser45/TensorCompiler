#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options final
{
    std::string actual_path;
    std::string expected_path;
    float atol = 1.0e-5F;
    float rtol = 1.0e-4F;
};

bool ParseFloat(const std::string& text, float& out_value)
{
    char* end = nullptr;
    out_value = std::strtof(text.c_str(), &end);
    return end != text.c_str() && *end == '\0';
}

bool ParseArgs(int argc, char** argv, Options& out_options)
{
    if (argc < 3) {
        return false;
    }

    out_options.actual_path = argv[1];
    out_options.expected_path = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--atol" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out_options.atol)) {
                return false;
            }
            continue;
        }
        if (arg == "--rtol" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out_options.rtol)) {
                return false;
            }
            continue;
        }
        return false;
    }
    return true;
}

bool ReadFloats(const std::string& path, std::vector<float>& out_values)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "ERROR: failed to open " << path << '\n';
        return false;
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        std::cerr << "ERROR: invalid float file size for " << path << '\n';
        return false;
    }

    out_values.resize(static_cast<std::size_t>(size) / sizeof(float));
    in.read(reinterpret_cast<char*>(out_values.data()), size);
    return in.good() || in.gcount() == size;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        std::cerr << "Usage: " << argv[0]
                  << " <actual.f32> <expected.f32> [--atol N] [--rtol N]\n";
        return 2;
    }

    std::vector<float> actual;
    std::vector<float> expected;
    if (!ReadFloats(options.actual_path, actual) ||
        !ReadFloats(options.expected_path, expected)) {
        return 1;
    }

    if (actual.size() != expected.size()) {
        std::cerr << "ERROR: length mismatch: actual=" << actual.size()
                  << ", expected=" << expected.size() << '\n';
        return 1;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float limit =
            options.atol + options.rtol * std::fabs(expected[i]);
        const float diff = std::fabs(actual[i] - expected[i]);
        if (diff > limit) {
            std::cerr << "ERROR: value mismatch at index " << i
                      << ": actual=" << actual[i]
                      << ", expected=" << expected[i] << ", diff=" << diff
                      << ", limit=" << limit << '\n';
            return 1;
        }
    }

    return 0;
}
