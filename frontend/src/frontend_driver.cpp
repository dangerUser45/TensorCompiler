#include <array>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <optional>
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

#include "driver_defaults.hpp"

namespace {

using namespace tc::frontend::driver;

enum LongOptionCode
{
    kEmitLlvm = 1000,
    kEmitAsm,
    kEmitObject,
    kEmitExe,
    kTarget,
    kPassPipeline
};

enum class ExitCode : int
{
    kOk = 0,
    kError = 1,        // CLI / import / emission / IO error
    kVerifyFailed = 2, // semantic verification reported errors
};

struct Options final
{
    std::string input_path;
    std::optional<std::string> dump_path;
    std::optional<std::string> hash_path;
    std::optional<std::string> mlir_path;
    std::optional<std::string> metadata_path;
    std::optional<std::string> llvm_path;
    std::optional<std::string> asm_path;
    std::optional<std::string> object_path;
    std::optional<std::string> exe_path;
    std::string target_triple;
    std::string pass_pipeline;
    bool verify = false;
    bool verify_exec = false;
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
                                 std::string_view dir,
                                 std::string_view ext)
{
    const std::filesystem::path input(input_path);
    std::string model_name = input.stem().string();
    if (model_name.empty()) {
        model_name = "model";
    }
    return (std::filesystem::path(dir) / (model_name + std::string(ext)))
        .string();
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

    auto set_opt_path = [&](std::optional<std::string>& slot,
                            std::string_view flag_name) -> bool {
        if (slot.has_value()) {
            out_error = std::string("ERROR: --") + std::string(flag_name) +
                        " specified more than once";
            return false;
        }
        if (optarg == nullptr) {
            slot = std::string{};
            return true;
        }
        if (*optarg == '\0') {
            out_error = std::string("ERROR: --") + std::string(flag_name) +
                        " path is empty";
            return false;
        }
        slot = std::string(optarg);
        return true;
    };

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
                if (!set_opt_path(out_options.dump_path, "dump"))
                    return false;
                break;
            case 'h':
                if (!set_opt_path(out_options.hash_path, "hash"))
                    return false;
                break;
            case 'm':
                if (!set_opt_path(out_options.mlir_path, "emit-mlir"))
                    return false;
                break;
            case 'M':
                if (!set_opt_path(out_options.metadata_path, "emit-metadata"))
                    return false;
                break;
            case kEmitLlvm:
                if (!set_opt_path(out_options.llvm_path, "emit-llvm"))
                    return false;
                break;
            case kEmitAsm:
                if (!set_opt_path(out_options.asm_path, "emit-asm"))
                    return false;
                break;
            case kEmitObject:
                if (!set_opt_path(out_options.object_path, "emit-object"))
                    return false;
                break;
            case kEmitExe:
                if (!set_opt_path(out_options.exe_path, "emit-exe"))
                    return false;
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
        std::optional<std::string>* unresolved_slot = nullptr;
        std::size_t unresolved_count = 0;
        auto mark = [&](std::optional<std::string>& slot) {
            if (slot.has_value() && slot->empty()) {
                unresolved_slot = &slot;
                ++unresolved_count;
            }
        };
        mark(out_options.dump_path);
        mark(out_options.hash_path);
        mark(out_options.mlir_path);
        mark(out_options.metadata_path);
        mark(out_options.llvm_path);
        mark(out_options.asm_path);
        mark(out_options.object_path);
        mark(out_options.exe_path);

        if (unresolved_count == 1) {
            *unresolved_slot = argv[optind];
            ++optind;
        } else if (unresolved_count > 1) {
            out_error =
                "ERROR: ambiguous output path; use explicit "
                "--dump=, --hash=, --emit-mlir=, --emit-metadata=, "
                "--emit-llvm=, --emit-asm=, --emit-object= or --emit-exe=";
            return false;
        }
    }

    if (optind < argc) {
        out_error = "ERROR: multiple input files are not supported";
        return false;
    }

    struct PathDefault
    {
        std::optional<std::string>& slot;
        std::string_view dir;
        std::string_view ext;
    };
    const std::array<PathDefault, 8> path_defaults{ {
        { out_options.dump_path, kDefaultDumpDir, ".dot" },
        { out_options.hash_path, kDefaultHashDir, ".hash" },
        { out_options.mlir_path, kDefaultMlirDir, ".mlir" },
        { out_options.metadata_path, kDefaultMetadataDir, ".json" },
        { out_options.llvm_path, kDefaultLlvmDir, ".ll" },
        { out_options.asm_path, kDefaultAsmDir, ".s" },
        { out_options.object_path, kDefaultObjectDir, ".o" },
        { out_options.exe_path, kDefaultExeDir, "" },
    } };
    for (const auto& d : path_defaults) {
        if (d.slot.has_value() && d.slot->empty()) {
            *d.slot =
                BuildPathByModelName(out_options.input_path, d.dir, d.ext);
        }
    }

    // --emit-exe implies object and metadata defaults if not already requested.
    if (out_options.exe_path.has_value() &&
        !out_options.object_path.has_value()) {
        out_options.object_path = BuildPathByModelName(
            out_options.input_path, kDefaultObjectDir, ".o");
    }
    if (out_options.exe_path.has_value() &&
        !out_options.metadata_path.has_value()) {
        out_options.metadata_path = BuildPathByModelName(
            out_options.input_path, kDefaultMetadataDir, ".json");
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
    return options.llvm_path.has_value() || options.asm_path.has_value() ||
           options.object_path.has_value() || options.exe_path.has_value();
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
        return static_cast<int>(ExitCode::kError);
    }

    tc::frontend::Graph graph_ir;
    if (!tc::frontend::onnx::ImportOnnxToGraph(
            options.input_path, graph_ir, error)) {
        PrintStageError("import", error);
        return static_cast<int>(ExitCode::kError);
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
        return static_cast<int>(ExitCode::kVerifyFailed);
    }

    std::optional<std::string> mlir_text_cache;
    auto get_mlir_text = [&]() -> const std::string* {
        if (mlir_text_cache) {
            return &*mlir_text_cache;
        }
        std::string text;
        if (!tc::frontend::mlir::EmitMlirModule(graph_ir, text, error)) {
            PrintStageError("frontend",
                            error.empty() ? "ERROR: failed to emit MLIR"
                                          : error);
            return nullptr;
        }
        mlir_text_cache = std::move(text);
        return &*mlir_text_cache;
    };

    std::optional<std::string> metadata_cache;
    auto get_metadata_json = [&]() -> const std::string* {
        if (metadata_cache) {
            return &*metadata_cache;
        }
        std::string json;
        if (!tc::frontend::metadata::BuildMetadataJson(graph_ir, json, error)) {
            PrintStageError("frontend",
                            error.empty() ? "ERROR: failed to emit metadata"
                                          : error);
            return nullptr;
        }
        metadata_cache = std::move(json);
        return &*metadata_cache;
    };

    if (options.dump_path) {
        if (!EnsureParentDirExists(*options.dump_path)) {
            PrintStageError("frontend",
                            "ERROR: failed to create dump directory for " +
                                *options.dump_path);
            return static_cast<int>(ExitCode::kError);
        }

        std::ofstream out(*options.dump_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            PrintStageError("frontend",
                            "ERROR: failed to open dump file: " +
                                *options.dump_path);
            return static_cast<int>(ExitCode::kError);
        }
        tc::frontend::DumpGraph dumper(out);
        dumper.dump(graph_ir);
        std::cout << "Graph dump written to: " << *options.dump_path << '\n';
    }

    if (options.hash_path) {
        if (!EnsureParentDirExists(*options.hash_path)) {
            PrintStageError("frontend",
                            "ERROR: failed to create hash directory for " +
                                *options.hash_path);
            return static_cast<int>(ExitCode::kError);
        }

        std::ofstream out(*options.hash_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            PrintStageError("frontend",
                            "ERROR: failed to open hash file: " +
                                *options.hash_path);
            return static_cast<int>(ExitCode::kError);
        }

        const std::size_t graph_hash = tc::frontend::HashGraph(graph_ir);
        out << std::hex << std::setfill('0')
            << std::setw(static_cast<int>(sizeof(std::size_t) * 2))
            << graph_hash << '\n';
        std::cout << "Graph hash written to: " << *options.hash_path << '\n';
    }

    if (options.mlir_path) {
        const std::string* text = get_mlir_text();
        if (text == nullptr) {
            return static_cast<int>(ExitCode::kError);
        }
        if (!WriteTextFile(*options.mlir_path, *text, "frontend")) {
            return static_cast<int>(ExitCode::kError);
        }
        std::cout << "MLIR written to: " << *options.mlir_path << '\n';
    }

    if (options.metadata_path) {
        const std::string* json = get_metadata_json();
        if (json == nullptr) {
            return static_cast<int>(ExitCode::kError);
        }
        if (!WriteTextFile(*options.metadata_path, *json, "frontend")) {
            return static_cast<int>(ExitCode::kError);
        }
        std::cout << "Metadata written to: " << *options.metadata_path << '\n';
    }

    if (backend_output_requested) {
#if !TC_FRONTEND_HAS_BACKEND
        PrintStageError("backend",
                        "ERROR: frontend_driver was built without backend "
                        "codegen support");
        return static_cast<int>(ExitCode::kError);
#else
        const std::string* mlir_text_ptr = get_mlir_text();
        if (mlir_text_ptr == nullptr) {
            return static_cast<int>(ExitCode::kError);
        }
        const std::string& mlir_text = *mlir_text_ptr;

        tc::backend::BackendDiagnostic backend_diagnostic;
        if (options.llvm_path) {
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
                return static_cast<int>(ExitCode::kError);
            }
            if (!WriteTextFile(*options.llvm_path, llvm_ir, "backend")) {
                return static_cast<int>(ExitCode::kError);
            }
            std::cout << "LLVM IR written to: " << *options.llvm_path << '\n';
        }

        if (options.asm_path) {
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
                return static_cast<int>(ExitCode::kError);
            }
            if (!WriteTextFile(*options.asm_path, asm_text, "backend")) {
                return static_cast<int>(ExitCode::kError);
            }
            std::cout << "ASM written to: " << *options.asm_path << '\n';
        }

        if (options.object_path) {
            if (!tc::backend::EmitObjectFromMlirText(mlir_text,
                                                     options.input_path,
                                                     options.pass_pipeline,
                                                     options.target_triple,
                                                     *options.object_path,
                                                     backend_diagnostic)) {
                PrintStageError(
                    "backend",
                    tc::backend::FormatBackendDiagnostic(backend_diagnostic));
                return static_cast<int>(ExitCode::kError);
            }
            std::cout << "Object written to: " << *options.object_path << '\n';
        }

        if (options.exe_path) {
            const std::string* metadata_json = get_metadata_json();
            if (metadata_json == nullptr) {
                return static_cast<int>(ExitCode::kError);
            }
            if (!WriteTextFile(
                    *options.metadata_path, *metadata_json, "frontend")) {
                return static_cast<int>(ExitCode::kError);
            }
            std::cout << "Metadata written to: " << *options.metadata_path
                      << '\n';

            if (!tc::backend::LinkExecutableWithRuntime(*options.object_path,
                                                        *options.metadata_path,
                                                        *options.exe_path,
                                                        backend_diagnostic)) {
                PrintStageError(
                    "backend",
                    tc::backend::FormatBackendDiagnostic(backend_diagnostic));
                return static_cast<int>(ExitCode::kError);
            }
            std::cout << "Executable written to: " << *options.exe_path << '\n';
        }
#endif
    }

    if ((options.verify || options.verify_exec) && !verified) {
        return static_cast<int>(ExitCode::kVerifyFailed);
    }

    return static_cast<int>(ExitCode::kOk);
}
