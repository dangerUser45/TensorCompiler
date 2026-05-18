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

#include "backend_runner.hpp"
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

void FillDefaultPaths(Options& options)
{
    struct PathDefault
    {
        std::optional<std::string>& slot;
        std::string_view dir;
        std::string_view ext;
    };
    const std::array<PathDefault, 8> defaults{ {
        { options.dump_path, kDefaultDumpDir, ".dot" },
        { options.hash_path, kDefaultHashDir, ".hash" },
        { options.mlir_path, kDefaultMlirDir, ".mlir" },
        { options.metadata_path, kDefaultMetadataDir, ".json" },
        { options.llvm_path, kDefaultLlvmDir, ".ll" },
        { options.asm_path, kDefaultAsmDir, ".s" },
        { options.object_path, kDefaultObjectDir, ".o" },
        { options.exe_path, kDefaultExeDir, "" },
    } };
    for (const auto& d : defaults) {
        if (d.slot.has_value() && d.slot->empty()) {
            *d.slot = BuildPathByModelName(options.input_path, d.dir, d.ext);
        }
    }
    if (options.exe_path.has_value() && !options.object_path.has_value()) {
        options.object_path =
            BuildPathByModelName(options.input_path, kDefaultObjectDir, ".o");
    }
    if (options.exe_path.has_value() && !options.metadata_path.has_value()) {
        options.metadata_path = BuildPathByModelName(
            options.input_path, kDefaultMetadataDir, ".json");
    }
}

bool ResolvePositionalPath(Options& options,
                           int argc,
                           char** argv,
                           int& optind_io,
                           std::string& out_error)
{
    if (optind_io >= argc || (argc - optind_io) != 1) {
        return true;
    }
    std::optional<std::string>* unresolved_slot = nullptr;
    std::size_t unresolved_count = 0;
    auto mark = [&](std::optional<std::string>& slot) {
        if (slot.has_value() && slot->empty()) {
            unresolved_slot = &slot;
            ++unresolved_count;
        }
    };
    mark(options.dump_path);
    mark(options.hash_path);
    mark(options.mlir_path);
    mark(options.metadata_path);
    mark(options.llvm_path);
    mark(options.asm_path);
    mark(options.object_path);
    mark(options.exe_path);

    if (unresolved_count == 1) {
        *unresolved_slot = argv[optind_io];
        ++optind_io;
        return true;
    }
    if (unresolved_count > 1) {
        out_error = "ERROR: ambiguous output path; use explicit "
                    "--dump=, --hash=, --emit-mlir=, --emit-metadata=, "
                    "--emit-llvm=, --emit-asm=, --emit-object= or --emit-exe=";
        return false;
    }
    return true;
}

bool ParseRawFlags(int argc,
                   char** argv,
                   Options& out_options,
                   std::string& out_error,
                   int& optind_out)
{
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
    optind_out = optind;
    return true;
}

bool ParseArgs(int argc,
               char** argv,
               Options& out_options,
               std::string& out_error)
{
    out_options = Options{};
    out_error.clear();

    int optind_after = 0;
    if (!ParseRawFlags(argc, argv, out_options, out_error, optind_after)) {
        return false;
    }
    if (!ResolvePositionalPath(
            out_options, argc, argv, optind_after, out_error)) {
        return false;
    }
    if (optind_after < argc) {
        out_error = "ERROR: multiple input files are not supported";
        return false;
    }
    FillDefaultPaths(out_options);
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

ExitCode RunVerify(tc::frontend::Graph& graph,
                   bool strict,
                   bool report_always,
                   bool& out_verified)
{
    tc::frontend::verify::Report report;
    out_verified =
        strict ? tc::frontend::verify::VerifyGraphForExecutable(graph, report)
               : tc::frontend::verify::VerifyGraphForExecution(graph, report);
    if (report_always || !out_verified) {
        PrintVerifyReport(report);
    }
    return ExitCode::kOk;
}

ExitCode RunDump(tc::frontend::Graph& graph, const std::string& path)
{
    if (!EnsureParentDirExists(path)) {
        PrintStageError("frontend",
                        "ERROR: failed to create dump directory for " + path);
        return ExitCode::kError;
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        PrintStageError("frontend", "ERROR: failed to open dump file: " + path);
        return ExitCode::kError;
    }
    tc::frontend::DumpGraph dumper(out);
    dumper.dump(graph);
    std::cout << "Graph dump written to: " << path << '\n';
    return ExitCode::kOk;
}

ExitCode RunHash(tc::frontend::Graph& graph, const std::string& path)
{
    if (!EnsureParentDirExists(path)) {
        PrintStageError("frontend",
                        "ERROR: failed to create hash directory for " + path);
        return ExitCode::kError;
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        PrintStageError("frontend", "ERROR: failed to open hash file: " + path);
        return ExitCode::kError;
    }
    const std::size_t graph_hash = tc::frontend::HashGraph(graph);
    out << std::hex << std::setfill('0')
        << std::setw(static_cast<int>(sizeof(std::size_t) * 2)) << graph_hash
        << '\n';
    std::cout << "Graph hash written to: " << path << '\n';
    return ExitCode::kOk;
}

ExitCode RunEmitMlir(const std::string& path, const std::string& mlir_text)
{
    if (!WriteTextFile(path, mlir_text, "frontend")) {
        return ExitCode::kError;
    }
    std::cout << "MLIR written to: " << path << '\n';
    return ExitCode::kOk;
}

ExitCode RunEmitMetadata(const std::string& path, const std::string& json)
{
    if (!WriteTextFile(path, json, "frontend")) {
        return ExitCode::kError;
    }
    std::cout << "Metadata written to: " << path << '\n';
    return ExitCode::kOk;
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
        RunVerify(graph_ir,
                  /*strict=*/options.verify_exec || backend_output_requested,
                  /*report_always=*/options.verify || options.verify_exec,
                  verified);
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
        const ExitCode rc = RunDump(graph_ir, *options.dump_path);
        if (rc != ExitCode::kOk)
            return static_cast<int>(rc);
    }

    if (options.hash_path) {
        const ExitCode rc = RunHash(graph_ir, *options.hash_path);
        if (rc != ExitCode::kOk)
            return static_cast<int>(rc);
    }

    if (options.mlir_path) {
        const std::string* text = get_mlir_text();
        if (text == nullptr)
            return static_cast<int>(ExitCode::kError);
        const ExitCode rc = RunEmitMlir(*options.mlir_path, *text);
        if (rc != ExitCode::kOk)
            return static_cast<int>(rc);
    }

    if (options.metadata_path) {
        const std::string* json = get_metadata_json();
        if (json == nullptr)
            return static_cast<int>(ExitCode::kError);
        const ExitCode rc = RunEmitMetadata(*options.metadata_path, *json);
        if (rc != ExitCode::kOk)
            return static_cast<int>(rc);
    }

    if (backend_output_requested) {
        const std::string* mlir_text_ptr = get_mlir_text();
        if (mlir_text_ptr == nullptr) {
            return static_cast<int>(ExitCode::kError);
        }

        std::string metadata_path_str;
        std::string metadata_json_str;
        if (options.exe_path) {
            const std::string* json = get_metadata_json();
            if (json == nullptr) {
                return static_cast<int>(ExitCode::kError);
            }
            metadata_path_str = *options.metadata_path;
            metadata_json_str = *json;
        }

        const tc::frontend::driver::BackendRequest request{
            options.llvm_path, options.asm_path,      options.object_path,
            options.exe_path,  options.target_triple, options.pass_pipeline,
        };
        const int rc =
            tc::frontend::driver::RunBackendPipeline(request,
                                                     *mlir_text_ptr,
                                                     options.input_path,
                                                     metadata_json_str,
                                                     metadata_path_str);
        if (rc != 0) {
            return rc;
        }
    }

    if ((options.verify || options.verify_exec) && !verified) {
        return static_cast<int>(ExitCode::kVerifyFailed);
    }

    return static_cast<int>(ExitCode::kOk);
}
