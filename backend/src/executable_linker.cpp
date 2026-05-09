#include "executable_linker.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

#ifndef TC_BACKEND_RUNTIME_DIR
#define TC_BACKEND_RUNTIME_DIR "backend/runtime"
#endif

namespace {

std::string ShellQuote(const std::filesystem::path& path)
{
    std::string quoted = "'";
    for (const char c : path.string()) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

bool ReadFileToString(const std::filesystem::path& path, std::string& out_text)
{
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    out_text = buffer.str();
    return true;
}

bool ExtractByteSizes(const std::string& metadata_json,
                      std::string& input_bytes,
                      std::string& output_bytes)
{
    static const std::regex kByteSizeRegex(R"("byte_size"\s*:\s*([0-9]+))");
    std::sregex_iterator it(metadata_json.begin(),
                            metadata_json.end(),
                            kByteSizeRegex);
    const std::sregex_iterator end;
    if (it == end) {
        return false;
    }
    input_bytes = (*it)[1].str();
    ++it;
    if (it == end) {
        return false;
    }
    output_bytes = (*it)[1].str();
    return true;
}

} // namespace

namespace tc::backend {

bool LinkExecutableWithRuntime(const std::string& model_object_path,
                               const std::string& metadata_json_path,
                               const std::string& output_executable_path,
                               BackendDiagnostic& out_diagnostic)
{
    out_diagnostic = BackendDiagnostic{};

    if (model_object_path.empty() || metadata_json_path.empty() ||
        output_executable_path.empty()) {
        out_diagnostic.stage = "executable-link";
        out_diagnostic.message = "model object, metadata, and executable paths are required";
        return false;
    }

    const std::filesystem::path model_object(model_object_path);
    const std::filesystem::path metadata(metadata_json_path);
    const std::filesystem::path executable(output_executable_path);
    if (!std::filesystem::exists(model_object)) {
        out_diagnostic.stage = "executable-link";
        out_diagnostic.message = "model object file does not exist: " +
                                 model_object.string();
        return false;
    }
    if (!std::filesystem::exists(metadata)) {
        out_diagnostic.stage = "executable-link";
        out_diagnostic.message = "metadata file does not exist: " +
                                 metadata.string();
        return false;
    }

    std::string metadata_json;
    if (!ReadFileToString(metadata, metadata_json)) {
        out_diagnostic.stage = "executable-link";
        out_diagnostic.message = "failed to read metadata file: " +
                                 metadata.string();
        return false;
    }

    std::string input_bytes;
    std::string output_bytes;
    if (!ExtractByteSizes(metadata_json, input_bytes, output_bytes)) {
        out_diagnostic.stage = "executable-link";
        out_diagnostic.message = "metadata JSON does not contain input/output byte_size";
        return false;
    }

    const std::filesystem::path parent = executable.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            out_diagnostic.stage = "executable-link";
            out_diagnostic.message =
                "failed to create executable directory: " + ec.message();
            return false;
        }
    }

    const std::filesystem::path runtime_source =
        std::filesystem::path(TC_BACKEND_RUNTIME_DIR) / "tc_model_runner.cpp";
    if (!std::filesystem::exists(runtime_source)) {
        out_diagnostic.stage = "executable-link";
        out_diagnostic.message = "runtime source does not exist: " +
                                 runtime_source.string();
        return false;
    }

    std::string command = "clang++ -std=c++17 -O2";
    command += " -DTC_MODEL_INPUT_BYTES=" + input_bytes;
    command += " -DTC_MODEL_OUTPUT_BYTES=" + output_bytes;
    command += " " + ShellQuote(runtime_source);
    command += " " + ShellQuote(model_object);
    command += " -o " + ShellQuote(executable);

    if (std::system(command.c_str()) != 0) {
        out_diagnostic.stage = "executable-link";
        out_diagnostic.message = "clang++ failed while linking executable";
        return false;
    }

    return true;
}

} // namespace tc::backend
