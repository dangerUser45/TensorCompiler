#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include "dump_graph.hpp"
#include "graph_hash.hpp"
#include "graph_verifier.hpp"
#include "mlir_emitter.hpp"
#include "model_metadata.hpp"
#include "onnx.pb.h"
#include "onnx_importer.hpp"

#ifndef TC_DEFAULT_DUMP_DIR
#define TC_DEFAULT_DUMP_DIR "build/dump"
#endif

#ifndef TC_DEFAULT_HASH_DIR
#define TC_DEFAULT_HASH_DIR "build/hash"
#endif

#ifndef TC_DEFAULT_MLIR_DIR
#define TC_DEFAULT_MLIR_DIR "build/mlir"
#endif

#ifndef TC_DEFAULT_METADATA_DIR
#define TC_DEFAULT_METADATA_DIR "build/metadata"
#endif

namespace {

struct Options final
{
    std::string input_path;
    std::string dump_path;
    std::string hash_path;
    std::string mlir_path;
    std::string metadata_path;
    bool verify = false;
    bool verify_exec = false;
    bool dump_requested = false;
    bool hash_requested = false;
    bool emit_mlir_requested = false;
    bool emit_metadata_requested = false;
};

inline std::string BuildUsage(const char* argv0)
{
    return std::string("Usage: ") + argv0 +
           " <model.onnx> [--verify] [--verify-exec] [--dump[=<output.dot>]] "
           "[--hash[=<output.hash>]] [--emit-mlir[=<output.mlir>]] "
           "[--emit-metadata[=<output.json>]]";
}

std::string BuildPathByModelName(const std::string& input_path,
                                 const std::string& dir,
                                 const std::string& ext)
{
    const std::filesystem::path input(input_path);
    std::string model_name = input.stem().string();
    if (model_name.empty()) {
        model_name = "model";
    }
    return (std::filesystem::path(dir) / (model_name + ext)).string();
}

bool ParseArgs(int argc,
               char** argv,
               Options& out_options,
               std::string& out_error)
{
    out_options = Options{};
    out_error.clear();

    static const option kLongOptions[] = {
        { "verify", no_argument, nullptr, 'v' },
        { "verify-exec", no_argument, nullptr, 'x' },
        { "dump", optional_argument, nullptr, 'd' },
        { "hash", optional_argument, nullptr, 'h' },
        { "emit-mlir", optional_argument, nullptr, 'm' },
        { "emit-metadata", optional_argument, nullptr, 'M' },
        { nullptr, 0, nullptr, 0 }
    };

    opterr = 0;
    optind = 1;

    int opt = 0;
    while ((opt = getopt_long(
                argc, argv, "vd::h::m::M::", kLongOptions, nullptr)) != -1) {
        switch (opt) {
            case 'v':
                out_options.verify = true;
                break;
            case 'x':
                out_options.verify_exec = true;
                break;
            case 'd':
                out_options.dump_requested = true;
                if (optarg != nullptr) {
                    out_options.dump_path = optarg;
                    if (out_options.dump_path.empty()) {
                        out_error = "ERROR: --dump path is empty";
                        return false;
                    }
                }
                break;
            case 'h':
                out_options.hash_requested = true;
                if (optarg != nullptr) {
                    out_options.hash_path = optarg;
                    if (out_options.hash_path.empty()) {
                        out_error = "ERROR: --hash path is empty";
                        return false;
                    }
                }
                break;
            case 'm':
                out_options.emit_mlir_requested = true;
                if (optarg != nullptr) {
                    out_options.mlir_path = optarg;
                    if (out_options.mlir_path.empty()) {
                        out_error = "ERROR: --emit-mlir path is empty";
                        return false;
                    }
                }
                break;
            case 'M':
                out_options.emit_metadata_requested = true;
                if (optarg != nullptr) {
                    out_options.metadata_path = optarg;
                    if (out_options.metadata_path.empty()) {
                        out_error = "ERROR: --emit-metadata path is empty";
                        return false;
                    }
                }
                break;
            case '?':
                out_error = "ERROR: unknown option or missing value";
                return false;
        }
    }

    if (optind >= argc) {
        out_error = "ERROR: model path is required";
        return false;
    }

    out_options.input_path = argv[optind];
    ++optind;

    if (optind < argc && (argc - optind) == 1) {
        std::string* unresolved_output_path = nullptr;
        std::size_t unresolved_count = 0;

        auto mark_if_unresolved = [&](bool requested, std::string& path) {
            if (requested && path.empty()) {
                unresolved_output_path = &path;
                ++unresolved_count;
            }
        };
        mark_if_unresolved(out_options.dump_requested, out_options.dump_path);
        mark_if_unresolved(out_options.hash_requested, out_options.hash_path);
        mark_if_unresolved(out_options.emit_mlir_requested,
                           out_options.mlir_path);
        mark_if_unresolved(out_options.emit_metadata_requested,
                           out_options.metadata_path);

        if (unresolved_count == 1) {
            *unresolved_output_path = argv[optind];
            ++optind;
        } else if (unresolved_count > 1) {
            out_error = "ERROR: ambiguous output path; use explicit "
                        "--dump=, --hash=, --emit-mlir= or "
                        "--emit-metadata=";
            return false;
        }
    }

    if (optind < argc) {
        out_error = "ERROR: multiple input files are not supported";
        return false;
    }

    if (out_options.dump_requested && out_options.dump_path.empty()) {
        out_options.dump_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_DUMP_DIR, ".dot");
    }

    if (out_options.hash_requested && out_options.hash_path.empty()) {
        out_options.hash_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_HASH_DIR, ".hash");
    }

    if (out_options.emit_mlir_requested && out_options.mlir_path.empty()) {
        out_options.mlir_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_MLIR_DIR, ".mlir");
    }

    if (out_options.emit_metadata_requested &&
        out_options.metadata_path.empty()) {
        out_options.metadata_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_METADATA_DIR, ".json");
    }

    return true;
}

void PrintVerifyReport(const tc::frontend::verify::Report& report)
{
    std::cout << "[semantic] errors: " << report.error_count() << '\n';
    std::cout << "[semantic] warnings: " << report.warning_count() << '\n';
    for (const auto& diagnostic : report.diagnostics()) {
        std::cout << "[semantic]["
                  << tc::frontend::verify::SeverityToString(diagnostic.severity)
                  << "] " << diagnostic.message << '\n';
    }
}

std::string StripErrorPrefix(std::string message)
{
    constexpr std::string_view kErrorPrefix = "ERROR: ";
    if (message.rfind(kErrorPrefix.data(), 0) == 0) {
        return message.substr(kErrorPrefix.size());
    }
    return message;
}

void PrintStageError(std::string_view stage, std::string message)
{
    const std::string normalized = StripErrorPrefix(std::move(message));
    std::cerr << "ERROR[" << stage
              << "]: " << (normalized.empty() ? "unknown failure" : normalized)
              << '\n';
}

bool EnsureParentDirExists(const std::string& file_path)
{
    const std::filesystem::path path(file_path);
    const std::filesystem::path parent = path.parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    return !ec;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    std::string error;
    if (!ParseArgs(argc, argv, options, error)) {
        PrintStageError("cli", error);
        if (error.rfind("Usage:", 0) != 0) {
            std::cerr << BuildUsage(argv[0]) << '\n';
        }
        return 1;
    }

    ::onnx::ModelProto model;
    tc::frontend::Graph graph_ir;
    if (!tc::frontend::onnx::ImportOnnxToGraph(
            options.input_path, model, graph_ir, error)) {
        PrintStageError("import", error);
        return 1;
    }

    bool verified = true;
    if (options.verify || options.verify_exec) {
        tc::frontend::verify::Report report;
        verified = options.verify_exec
                       ? tc::frontend::verify::VerifyGraphForExecutable(
                             graph_ir, report)
                       : tc::frontend::verify::VerifyGraphForExecution(graph_ir,
                                                                       report);
        PrintVerifyReport(report);
    }

    if (options.dump_requested) {
        if (!EnsureParentDirExists(options.dump_path)) {
            PrintStageError("frontend",
                            "ERROR: failed to create dump directory for " +
                                options.dump_path);
            return 1;
        }

        std::ofstream out(options.dump_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            PrintStageError("frontend",
                            "ERROR: failed to open dump file: " +
                                options.dump_path);
            return 1;
        }
        tc::frontend::DumpGraph dumper(out);
        dumper.dump(graph_ir);
        std::cout << "Graph dump written to: " << options.dump_path << '\n';
    }

    if (options.hash_requested) {
        if (!EnsureParentDirExists(options.hash_path)) {
            PrintStageError("frontend",
                            "ERROR: failed to create hash directory for " +
                                options.hash_path);
            return 1;
        }

        std::ofstream out(options.hash_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            PrintStageError("frontend",
                            "ERROR: failed to open hash file: " +
                                options.hash_path);
            return 1;
        }

        const std::size_t graph_hash = tc::frontend::HashGraph(graph_ir);
        out << std::hex << std::setfill('0')
            << std::setw(static_cast<int>(sizeof(std::size_t) * 2))
            << graph_hash << '\n';
        std::cout << "Graph hash written to: " << options.hash_path << '\n';
    }

    if (options.emit_mlir_requested) {
        if (!EnsureParentDirExists(options.mlir_path)) {
            PrintStageError("backend",
                            "ERROR: failed to create mlir directory for " +
                                options.mlir_path);
            return 1;
        }

        std::string mlir_text;
        if (!tc::frontend::mlir::EmitMlirModule(graph_ir, mlir_text, error)) {
            PrintStageError("backend",
                            error.empty() ? "ERROR: failed to emit MLIR"
                                          : error);
            return 1;
        }

        std::ofstream out(options.mlir_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            PrintStageError("backend",
                            "ERROR: failed to open mlir file: " +
                                options.mlir_path);
            return 1;
        }

        out << mlir_text;
        if (!out.good()) {
            PrintStageError("backend",
                            "ERROR: failed to write mlir file: " +
                                options.mlir_path);
            return 1;
        }
        std::cout << "MLIR written to: " << options.mlir_path << '\n';
    }

    if (options.emit_metadata_requested) {
        if (!EnsureParentDirExists(options.metadata_path)) {
            PrintStageError("frontend",
                            "ERROR: failed to create metadata directory for " +
                                options.metadata_path);
            return 1;
        }

        std::string metadata_json;
        if (!tc::frontend::metadata::BuildMetadataJson(
                graph_ir, metadata_json, error)) {
            PrintStageError("frontend",
                            error.empty() ? "ERROR: failed to emit metadata"
                                          : error);
            return 1;
        }

        std::ofstream out(options.metadata_path,
                          std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            PrintStageError("frontend",
                            "ERROR: failed to open metadata file: " +
                                options.metadata_path);
            return 1;
        }
        out << metadata_json;
        if (!out.good()) {
            PrintStageError("frontend",
                            "ERROR: failed to write metadata file: " +
                                options.metadata_path);
            return 1;
        }
        std::cout << "Metadata written to: " << options.metadata_path << '\n';
    }

    if ((options.verify || options.verify_exec) && !verified) {
        return 2;
    }

    return 0;
}
