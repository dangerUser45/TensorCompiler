#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef TC_MODEL_INPUT_BYTES
#define TC_MODEL_INPUT_BYTES 0
#endif

#ifndef TC_MODEL_OUTPUT_BYTES
#define TC_MODEL_OUTPUT_BYTES 0
#endif

extern "C" void tc_model(const float* input0, float* output0);

namespace {

struct Options final
{
    std::string input_path;
    std::string output_path;
};

void PrintUsage(const char* argv0)
{
    std::cerr << "Usage: " << argv0
              << " --input <input.f32> --output <output.f32>\n";
}

bool ParseArgs(int argc, char** argv, Options& out_options)
{
    out_options = Options{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            out_options.input_path = argv[++i];
            continue;
        }
        if (arg == "--output" && i + 1 < argc) {
            out_options.output_path = argv[++i];
            continue;
        }
        return false;
    }
    return !out_options.input_path.empty() && !out_options.output_path.empty();
}

bool ReadExactFile(const std::string& path,
                   std::vector<float>& out_values,
                   std::size_t expected_bytes)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "ERROR[runtime]: failed to open input file: " << path
                  << '\n';
        return false;
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0 || static_cast<std::size_t>(size) != expected_bytes) {
        std::cerr << "ERROR[runtime]: input byte size mismatch: expected "
                  << expected_bytes << ", got " << size << '\n';
        return false;
    }

    out_values.resize(expected_bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(out_values.data()),
            static_cast<std::streamsize>(expected_bytes));
    return in.good() || in.gcount() == static_cast<std::streamsize>(expected_bytes);
}

bool WriteExactFile(const std::string& path, const std::vector<float>& values)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "ERROR[runtime]: failed to open output file: " << path
                  << '\n';
        return false;
    }

    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    return out.good();
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        PrintUsage(argv[0]);
        return 1;
    }

    if (TC_MODEL_INPUT_BYTES == 0 || TC_MODEL_OUTPUT_BYTES == 0) {
        std::cerr << "ERROR[runtime]: model byte sizes were not configured\n";
        return 1;
    }

    std::vector<float> input;
    if (!ReadExactFile(options.input_path, input, TC_MODEL_INPUT_BYTES)) {
        return 1;
    }

    std::vector<float> output(TC_MODEL_OUTPUT_BYTES / sizeof(float), 0.0F);
    tc_model(input.data(), output.data());

    if (!WriteExactFile(options.output_path, output)) {
        return 1;
    }

    return 0;
}
