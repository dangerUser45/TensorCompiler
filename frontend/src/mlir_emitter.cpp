#include "mlir_emitter.hpp"

#include <sstream>

namespace {

std::string GraphNameOrFallback(const tc::frontend::Graph& graph)
{
    if (graph.get_name().empty()) {
        return "unnamed_graph";
    }
    return graph.get_name();
}

std::string NodeNameOrFallback(const tc::frontend::Node& node)
{
    if (node.get_name_node().empty()) {
        return "<unnamed>";
    }
    return node.get_name_node();
}

} // namespace

namespace tc::frontend::mlir {

bool EmitMlirModuleSkeleton(const Graph& graph,
                            std::string& out_mlir_text,
                            std::string& out_error)
{
    out_mlir_text.clear();
    out_error.clear();

    std::ostringstream out;
    out << "module {\n";
    out << "  // tc.graph: " << GraphNameOrFallback(graph) << '\n';
    out << "  // tc.node_count: " << graph.get_nodes().size() << '\n';
    out << "  func.func @main() {\n";
    out << "    // TODO(tc): Graph->MLIR lowering is not implemented yet.\n";

    for (std::size_t i = 0; i < graph.get_nodes().size(); ++i) {
        const auto& node = *graph.get_nodes()[i];
        out << "    // node[" << i << "]: op=" << ToString(node.get_op_kind())
            << ", name=" << NodeNameOrFallback(node) << '\n';
    }

    out << "    return\n";
    out << "  }\n";
    out << "}\n";

    out_mlir_text = out.str();
    return true;
}

} // namespace tc::frontend::mlir
