#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

#include "dump_graph.hpp"
#include "graph_verifier.hpp"
#include "onnx_importer.hpp"

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
    const std::size_t slash_pos = input_path.find_last_of("/\\");
    const std::size_t dot_pos = input_path.find_last_of('.');

    if (dot_pos == std::string::npos ||
        (slash_pos != std::string::npos && dot_pos < slash_pos)) {
        return input_path + ".dot";
    }

    return input_path.substr(0, dot_pos) + ".dot";
}

bool ParseArgs(int argc,
               char** argv,
               Options& out_options,
               std::string& out_error)
{
    out_options = Options{};
    out_error.clear();

    if (argc < 2) {
        out_error = BuildUsage(argv[0]);
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--verify") {
            out_options.verify = true;
            continue;
        }

        if (arg == "--dump") {
            if (i + 1 >= argc) {
                out_error = "ERROR: --dump requires output path";
                return false;
            }
            out_options.dump_path = argv[++i];
            if (out_options.dump_path.empty()) {
                out_error = "ERROR: --dump path is empty";
                return false;
            }
            continue;
        }

        if (arg.rfind("--", 0) == 0) {
            out_error = "ERROR: unknown option: " + arg;
            return false;
        }

        if (out_options.input_path.empty()) {
            out_options.input_path = arg;
        } else {
            out_error = "ERROR: multiple input files are not supported";
            return false;
        }
    }

    if (out_options.input_path.empty()) {
        out_error = "ERROR: model path is required";
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
