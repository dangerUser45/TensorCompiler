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
#include "onnx_importer.hpp"

#if TC_FRONTEND_HAS_BACKEND
#include "codegen_driver.hpp"
#include "executable_linker.hpp"
#include "object_emitter.hpp"
#endif

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

#ifndef TC_DEFAULT_LLVM_DIR
#define TC_DEFAULT_LLVM_DIR "build/llvm"
#endif

#ifndef TC_DEFAULT_ASM_DIR
#define TC_DEFAULT_ASM_DIR "build/asm"
#endif

#ifndef TC_DEFAULT_OBJECT_DIR
#define TC_DEFAULT_OBJECT_DIR "build/object"
#endif

#ifndef TC_DEFAULT_EXE_DIR
#define TC_DEFAULT_EXE_DIR "build/bin"
#endif

namespace {

enum LongOptionCode
{
    kEmitLlvm = 1000,
    kEmitAsm,
    kEmitObject,
    kEmitExe,
    kTarget,
    kPassPipeline
};

struct Options final
{
    std::string input_path;
    std::string dump_path;
    std::string hash_path;
    std::string mlir_path;
    std::string metadata_path;
    std::string llvm_path;
    std::string asm_path;
    std::string object_path;
    std::string exe_path;
    std::string target_triple;
    std::string pass_pipeline;
    bool verify = false;
    bool verify_exec = false;
    bool dump_requested = false;
    bool hash_requested = false;
    bool emit_mlir_requested = false;
    bool emit_metadata_requested = false;
    bool emit_llvm_requested = false;
    bool emit_asm_requested = false;
    bool emit_object_requested = false;
    bool emit_exe_requested = false;
};

inline std::string BuildUsage(const char* argv0)
{
    return std::string("Usage: ") + argv0 +
           " <model.onnx> [--verify] [--verify-exec] [--dump[=<output.dot>]] "
           "[--hash[=<output.hash>]] [--emit-mlir[=<output.mlir>]] "
           "[--emit-metadata[=<output.json>]] [--emit-llvm[=<output.ll>]] "
           "[--emit-asm[=<output.s>]] [--emit-object[=<output.o>]] "
           "[--emit-exe[=<output_binary>]] [--target=<triple>] "
           "[--pass-pipeline=<pipeline>]";
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
        { "emit-llvm", optional_argument, nullptr, kEmitLlvm },
        { "emit-asm", optional_argument, nullptr, kEmitAsm },
        { "emit-object", optional_argument, nullptr, kEmitObject },
        { "emit-exe", optional_argument, nullptr, kEmitExe },
        { "target", required_argument, nullptr, kTarget },
        { "pass-pipeline", required_argument, nullptr, kPassPipeline },
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
            case kEmitLlvm:
                out_options.emit_llvm_requested = true;
                if (optarg != nullptr) {
                    out_options.llvm_path = optarg;
                    if (out_options.llvm_path.empty()) {
                        out_error = "ERROR: --emit-llvm path is empty";
                        return false;
                    }
                }
                break;
            case kEmitAsm:
                out_options.emit_asm_requested = true;
                if (optarg != nullptr) {
                    out_options.asm_path = optarg;
                    if (out_options.asm_path.empty()) {
                        out_error = "ERROR: --emit-asm path is empty";
                        return false;
                    }
                }
                break;
            case kEmitObject:
                out_options.emit_object_requested = true;
                if (optarg != nullptr) {
                    out_options.object_path = optarg;
                    if (out_options.object_path.empty()) {
                        out_error = "ERROR: --emit-object path is empty";
                        return false;
                    }
                }
                break;
            case kEmitExe:
                out_options.emit_exe_requested = true;
                if (optarg != nullptr) {
                    out_options.exe_path = optarg;
                    if (out_options.exe_path.empty()) {
                        out_error = "ERROR: --emit-exe path is empty";
                        return false;
                    }
                }
                break;
            case kTarget:
                out_options.target_triple = optarg == nullptr ? "" : optarg;
                if (out_options.target_triple.empty()) {
                    out_error = "ERROR: --target value is empty";
                    return false;
                }
                break;
            case kPassPipeline:
                out_options.pass_pipeline = optarg == nullptr ? "" : optarg;
                if (out_options.pass_pipeline.empty()) {
                    out_error = "ERROR: --pass-pipeline value is empty";
                    return false;
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
        mark_if_unresolved(out_options.emit_llvm_requested,
                           out_options.llvm_path);
        mark_if_unresolved(out_options.emit_asm_requested,
                           out_options.asm_path);
        mark_if_unresolved(out_options.emit_object_requested,
                           out_options.object_path);
        mark_if_unresolved(out_options.emit_exe_requested,
                           out_options.exe_path);

        if (unresolved_count == 1) {
            *unresolved_output_path = argv[optind];
            ++optind;
        } else if (unresolved_count > 1) {
            out_error = "ERROR: ambiguous output path; use explicit "
                        "--dump=, --hash=, --emit-mlir= or "
                        "--emit-metadata=, --emit-llvm=, --emit-asm=, "
                        "--emit-object= or --emit-exe=";
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

    if (out_options.emit_llvm_requested && out_options.llvm_path.empty()) {
        out_options.llvm_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_LLVM_DIR, ".ll");
    }

    if (out_options.emit_asm_requested && out_options.asm_path.empty()) {
        out_options.asm_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_ASM_DIR, ".s");
    }

    if ((out_options.emit_object_requested || out_options.emit_exe_requested) &&
        out_options.object_path.empty()) {
        out_options.object_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_OBJECT_DIR, ".o");
    }

    if (out_options.emit_exe_requested && out_options.exe_path.empty()) {
        out_options.exe_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_EXE_DIR, "");
    }

    if ((out_options.emit_exe_requested ||
         out_options.emit_metadata_requested) &&
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

bool WriteTextFile(const std::string& file_path,
                   const std::string& contents,
                   std::string_view stage)
{
    if (!EnsureParentDirExists(file_path)) {
        PrintStageError(
            stage, "ERROR: failed to create output directory for " + file_path);
        return false;
    }

    std::ofstream out(file_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        PrintStageError(stage,
                        "ERROR: failed to open output file: " + file_path);
        return false;
    }

    out << contents;
    if (!out.good()) {
        PrintStageError(stage,
                        "ERROR: failed to write output file: " + file_path);
        return false;
    }
    return true;
}

bool AnyBackendOutputRequested(const Options& options) noexcept
{
    return options.emit_llvm_requested || options.emit_asm_requested ||
           options.emit_object_requested || options.emit_exe_requested;
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

    tc::frontend::Graph graph_ir;
    if (!tc::frontend::onnx::ImportOnnxToGraph(
            options.input_path, graph_ir, error)) {
        PrintStageError("import", error);
        return 1;
    }

    bool verified = true;
    const bool backend_output_requested = AnyBackendOutputRequested(options);
    if (options.verify || options.verify_exec || backend_output_requested) {
        tc::frontend::verify::Report report;
        verified = (options.verify_exec || backend_output_requested)
                       ? tc::frontend::verify::VerifyGraphForExecutable(
                             graph_ir, report)
                       : tc::frontend::verify::VerifyGraphForExecution(graph_ir,
                                                                       report);
        if (options.verify || options.verify_exec || !verified) {
            PrintVerifyReport(report);
        }
    }

    if (backend_output_requested && !verified) {
        return 2;
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

    if (backend_output_requested) {
#if !TC_FRONTEND_HAS_BACKEND
        PrintStageError("backend",
                        "ERROR: frontend_driver was built without backend "
                        "codegen support");
        return 1;
#else
        std::string mlir_text;
        if (!tc::frontend::mlir::EmitMlirModule(graph_ir, mlir_text, error)) {
            PrintStageError("backend",
                            error.empty() ? "ERROR: failed to emit MLIR"
                                          : error);
            return 1;
        }

        tc::backend::BackendDiagnostic backend_diagnostic;
        if (options.emit_llvm_requested) {
            std::string llvm_ir;
            if (!tc::backend::EmitLlvmIrFromMlirText(mlir_text,
                                                     options.input_path,
                                                     options.pass_pipeline,
                                                     options.target_triple,
                                                     llvm_ir,
                                                     backend_diagnostic)) {
                PrintStageError(
                    "backend",
                    tc::backend::FormatBackendDiagnostic(backend_diagnostic));
                return 1;
            }
            if (!WriteTextFile(options.llvm_path, llvm_ir, "backend")) {
                return 1;
            }
            std::cout << "LLVM IR written to: " << options.llvm_path << '\n';
        }

        if (options.emit_asm_requested) {
            std::string asm_text;
            if (!tc::backend::EmitAsmFromMlirText(mlir_text,
                                                  options.input_path,
                                                  options.pass_pipeline,
                                                  options.target_triple,
                                                  asm_text,
                                                  backend_diagnostic)) {
                PrintStageError(
                    "backend",
                    tc::backend::FormatBackendDiagnostic(backend_diagnostic));
                return 1;
            }
            if (!WriteTextFile(options.asm_path, asm_text, "backend")) {
                return 1;
            }
            std::cout << "ASM written to: " << options.asm_path << '\n';
        }

        if (options.emit_object_requested || options.emit_exe_requested) {
            if (!tc::backend::EmitObjectFromMlirText(mlir_text,
                                                     options.input_path,
                                                     options.pass_pipeline,
                                                     options.target_triple,
                                                     options.object_path,
                                                     backend_diagnostic)) {
                PrintStageError(
                    "backend",
                    tc::backend::FormatBackendDiagnostic(backend_diagnostic));
                return 1;
            }
            std::cout << "Object written to: " << options.object_path << '\n';
        }

        if (options.emit_exe_requested) {
            std::string metadata_json;
            if (!tc::frontend::metadata::BuildMetadataJson(
                    graph_ir, metadata_json, error)) {
                PrintStageError("frontend",
                                error.empty() ? "ERROR: failed to emit metadata"
                                              : error);
                return 1;
            }
            if (!WriteTextFile(
                    options.metadata_path, metadata_json, "frontend")) {
                return 1;
            }
            std::cout << "Metadata written to: " << options.metadata_path
                      << '\n';

            if (!tc::backend::LinkExecutableWithRuntime(options.object_path,
                                                        options.metadata_path,
                                                        options.exe_path,
                                                        backend_diagnostic)) {
                PrintStageError(
                    "backend",
                    tc::backend::FormatBackendDiagnostic(backend_diagnostic));
                return 1;
            }
            std::cout << "Executable written to: " << options.exe_path << '\n';
        }
#endif
    }

    if ((options.verify || options.verify_exec) && !verified) {
        return 2;
    }

    return 0;
}
