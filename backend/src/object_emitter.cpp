#include "object_emitter.hpp"

#include "codegen_driver.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

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

std::filesystem::path MakeTempPath(const std::string& stem,
                                   const std::string& extension)
{
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "_" + std::to_string(tick) + extension);
}

bool WriteStringToFile(const std::filesystem::path& path,
                       const std::string& text)
{
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out << text;
    return out.good();
}

} // namespace

namespace tc::backend {

bool EmitObjectFromMlirText(const std::string& mlir_text,
                            const std::string& source_name,
                            const std::string& pass_pipeline,
                            const std::string& target_triple,
                            const std::string& output_object_path,
                            BackendDiagnostic& out_diagnostic)
{
    out_diagnostic = BackendDiagnostic{};

    if (output_object_path.empty()) {
        out_diagnostic.stage = "object-emission";
        out_diagnostic.message = "output object path is empty";
        return false;
    }

    std::string llvm_ir;
    if (!EmitLlvmIrFromMlirText(mlir_text,
                                source_name,
                                pass_pipeline,
                                target_triple,
                                llvm_ir,
                                out_diagnostic)) {
        return false;
    }

    const std::filesystem::path object_path(output_object_path);
    const std::filesystem::path parent = object_path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            out_diagnostic.stage = "object-emission";
            out_diagnostic.message =
                "failed to create object output directory: " + ec.message();
            return false;
        }
    }

    const std::filesystem::path ir_path =
        MakeTempPath("tc_backend_object", ".ll");
    if (!WriteStringToFile(ir_path, llvm_ir)) {
        out_diagnostic.stage = "object-emission";
        out_diagnostic.message = "failed to write temporary LLVM IR file";
        return false;
    }

    std::string command = "llc -O0 -filetype=obj";
    if (!target_triple.empty()) {
        command += " -mtriple=" + target_triple;
    }
    command += " -o " + ShellQuote(object_path) + " " + ShellQuote(ir_path);

    if (std::system(command.c_str()) != 0) {
        out_diagnostic.stage = "object-emission";
        out_diagnostic.message = "llc failed while emitting object file";
        std::error_code ec;
        std::filesystem::remove(ir_path, ec);
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(ir_path, ec);
    out_diagnostic = BackendDiagnostic{};
    return true;
}

} // namespace tc::backend
