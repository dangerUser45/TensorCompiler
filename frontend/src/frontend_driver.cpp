#include <cstddef>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

#include "dump_graph.hpp"
#include "graph_verifier.hpp"
#include "onnx.pb.h"
#include "onnx_importer.hpp"

#ifndef TC_DEFAULT_DUMP_DIR
#define TC_DEFAULT_DUMP_DIR "build/dump"
#endif
namespace {

struct Options final
{
    std::string input_path;
    std::string dump_path;
    bool verify = false;
};

std::string BuildUsage(const char* argv0)
{
    return std::string("Usage: ") + argv0 +
           " <model.onnx> [--verify] [--dump <output.dot>]";
}

std::string BuildDefaultDumpPath(const std::string& input_path)
{
    const std::filesystem::path input(input_path);

    std::string model_name = input.stem().string();

    if (model_name.empty()) {
        model_name = "model";
    }

    const std::filesystem::path out =
        std::filesystem::path(TC_DEFAULT_DUMP_DIR) / (model_name + ".dot");

    return out.string();
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
        { "dump", required_argument, nullptr, 'd' },
        { nullptr, 0, nullptr, 0 }
    };

    opterr = 0;
    optind = 1;

    int opt = 0;

    while ((opt = getopt_long(argc, argv, "vd:", kLongOptions, nullptr)) !=
           -1) {
        switch (opt) {
            case 'v':
                out_options.verify = true;
                break;
            case 'd':
                out_options.dump_path = optarg ? optarg : "";
                if (out_options.dump_path.empty()) {
                    out_error = "ERROR: --dump path is empty";
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

    out_options.input_path = argv[optind++];

    if (optind < argc) {
        out_error = "ERROR: multiple input files are not supported";
        return false;
    }

    if (out_options.dump_path.empty()) {
        out_options.dump_path = BuildDefaultDumpPath(out_options.input_path);
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

    const std::filesystem::path dump_path(options.dump_path);
    const std::filesystem::path parent_dir = dump_path.parent_path();

    if (!parent_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent_dir, ec);
        if (ec) {
            std::cerr << "ERROR: failed to create dump directory: "
                      << parent_dir << " (" << ec.message() << ")\n";
            return 1;
        }
    }

    std::ofstream out(options.dump_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "ERROR: failed to open dump file: " << options.dump_path
                  << '\n';
        return 1;
    }

    tc::frontend::DumpGraph dumper(out);
    dumper.dump(graph_ir);
    out.close();

    std::cout << "Graph dump written to: " << options.dump_path << '\n';

    if (options.verify && !verified) {
        return 2;
    }

    return 0;
}
