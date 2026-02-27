#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>

#include "dump_graph.hpp"
#include "graph_hash.hpp"
#include "graph_verifier.hpp"
#include "onnx.pb.h"
#include "onnx_importer.hpp"

#ifndef TC_DEFAULT_DUMP_DIR
#define TC_DEFAULT_DUMP_DIR "build/dump"
#endif

#ifndef TC_DEFAULT_HASH_DIR
#define TC_DEFAULT_HASH_DIR "build/hash"
#endif

namespace {

struct Options final
{
    std::string input_path;
    std::string dump_path;
    std::string hash_path;
    bool verify = false;
    bool dump_requested = false;
    bool hash_requested = false;
};

inline std::string BuildUsage(const char* argv0)
{
    return std::string("Usage: ") + argv0 +
           " <model.onnx> [--verify] [--dump[=<output.dot>]] "
           "[--hash[=<output.hash>]]";
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
        { "dump", optional_argument, nullptr, 'd' },
        { "hash", optional_argument, nullptr, 'h' },
        { nullptr, 0, nullptr, 0 }
    };

    opterr = 0;
    optind = 1;

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "vd::h::", kLongOptions, nullptr)) !=
           -1) {
        switch (opt) {
            case 'v':
                out_options.verify = true;
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

    if (optind < argc) {
        if (out_options.dump_requested && out_options.dump_path.empty() &&
            (argc - optind) == 1) {
            out_options.dump_path = argv[optind];
        } else if (out_options.hash_requested && out_options.hash_path.empty() &&
                   (argc - optind) == 1) {
            out_options.hash_path = argv[optind];
        } else {
            out_error = "ERROR: multiple input files are not supported";
            return false;
        }
    }

    if (out_options.dump_requested && out_options.dump_path.empty()) {
        out_options.dump_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_DUMP_DIR, ".dot");
    }

    if (out_options.hash_requested && out_options.hash_path.empty()) {
        out_options.hash_path = BuildPathByModelName(
            out_options.input_path, TC_DEFAULT_HASH_DIR, ".hash");
    }

    return true;
}

void PrintVerifyReport(const tc::frontend::verify::Report& report)
{
    std::cout << "Verifier errors: " << report.error_count() << '\n';
    std::cout << "Verifier warnings: " << report.warning_count() << '\n';
    for (const auto& diagnostic : report.diagnostics()) {
        std::cout << tc::frontend::verify::SeverityToString(diagnostic.severity)
                  << ": " << diagnostic.message << '\n';
    }
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
        std::cerr << error << '\n';
        if (error.rfind("Usage:", 0) != 0) {
            std::cerr << BuildUsage(argv[0]) << '\n';
        }
        return 1;
    }

    ::onnx::ModelProto model;
    tc::frontend::Graph graph_ir;
    if (!tc::frontend::onnx::ImportOnnxToGraph(
            options.input_path, model, graph_ir, error)) {
        std::cerr << error << '\n';
        return 1;
    }

    bool verified = true;
    if (options.verify) {
        tc::frontend::verify::Report report;
        verified =
            tc::frontend::verify::VerifyGraphForExecution(graph_ir, report);
        PrintVerifyReport(report);
    }

    if (options.dump_requested) {
        if (!EnsureParentDirExists(options.dump_path)) {
            std::cerr << "ERROR: failed to create dump directory for "
                      << options.dump_path << '\n';
            return 1;
        }

        std::ofstream out(options.dump_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "ERROR: failed to open dump file: " << options.dump_path
                      << '\n';
            return 1;
        }
        tc::frontend::DumpGraph dumper(out);
        dumper.dump(graph_ir);
        std::cout << "Graph dump written to: " << options.dump_path << '\n';
    }

    if (options.hash_requested) {
        if (!EnsureParentDirExists(options.hash_path)) {
            std::cerr << "ERROR: failed to create hash directory for "
                      << options.hash_path << '\n';
            return 1;
        }

        std::ofstream out(options.hash_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "ERROR: failed to open hash file: " << options.hash_path
                      << '\n';
            return 1;
        }

        const std::size_t graph_hash = tc::frontend::HashGraph(graph_ir);
        out << std::hex << std::setfill('0')
            << std::setw(static_cast<int>(sizeof(std::size_t) * 2))
            << graph_hash << '\n';
        std::cout << "Graph hash written to: " << options.hash_path << '\n';
    }

    if (options.verify && !verified) {
        return 2;
    }

    return 0;
}
