#include "mlir_emitter.hpp"

#include <optional>
#include <sstream>
#include <string_view>

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

std::optional<std::string_view> MlirElemType(tc::frontend::DataID id) noexcept
{
    switch (id) {
        case tc::frontend::DataID::FLOAT:
            return "f32";
        case tc::frontend::DataID::DOUBLE:
            return "f64";
        case tc::frontend::DataID::INT8:
            return "i8";
        case tc::frontend::DataID::INT16:
            return "i16";
        case tc::frontend::DataID::INT32:
            return "i32";
        case tc::frontend::DataID::INT64:
            return "i64";
        case tc::frontend::DataID::UNSIGNED_INT8:
            return "ui8";
        case tc::frontend::DataID::UNSIGNED_INT16:
            return "ui16";
        case tc::frontend::DataID::UNSIGNED_INT32:
            return "ui32";
        case tc::frontend::DataID::UNSIGNED_INT64:
            return "ui64";
        case tc::frontend::DataID::COMPLEX64:
            return "complex<f32>";
        case tc::frontend::DataID::COMPLEX128:
            return "complex<f64>";
        case tc::frontend::DataID::STRING:
        case tc::frontend::DataID::UNDEFINED:
            return std::nullopt;
    }
    return std::nullopt;
}

bool BuildMlirTensorType(const tc::frontend::TensorInfo& tensor,
                         std::string& out_type,
                         std::string& out_error,
                         std::string_view tensor_role)
{
    out_type.clear();
    out_error.clear();

    const auto elem_type = MlirElemType(tensor.get_data_type().id);
    if (!elem_type.has_value()) {
        out_error = "ERROR: unsupported MLIR element type for " +
                    std::string(tensor_role) + " tensor '" + tensor.get_name() +
                    "'";
        return false;
    }

    std::ostringstream type;
    type << "tensor<";
    const auto& shape = tensor.get_shape();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            type << "x";
        }
        if (shape[i] < 0) {
            type << "?";
        } else {
            type << shape[i];
        }
    }
    if (!shape.empty()) {
        type << "x";
    }
    type << *elem_type << ">";
    out_type = type.str();
    return true;
}

bool CanEmitSimpleReluEntry(const tc::frontend::Graph& graph) noexcept
{
    if (graph.get_input_tensors().size() != 1 ||
        graph.get_output_tensors().size() != 1 ||
        graph.get_nodes().size() != 1) {
        return false;
    }

    const auto& node = *graph.get_nodes().front();
    if (node.get_op_kind() != tc::frontend::OpKind::kRelu) {
        return false;
    }
    if (node.get_inputs().size() != 1 || node.get_outputs().size() != 1) {
        return false;
    }
    return true;
}

bool EmitSimpleReluMain(const tc::frontend::Graph& graph,
                        std::string& out_mlir_text,
                        std::string& out_error)
{
    out_mlir_text.clear();
    out_error.clear();

    const auto& input = *graph.get_input_tensors().front();
    const auto& output = *graph.get_output_tensors().front();
    const auto& node = *graph.get_nodes().front();

    std::string input_ty;
    std::string output_ty;
    if (!BuildMlirTensorType(input, input_ty, out_error, "input")) {
        return false;
    }
    if (!BuildMlirTensorType(output, output_ty, out_error, "output")) {
        return false;
    }

    if (input_ty != output_ty) {
        out_error =
            "ERROR: simple Relu MLIR emission requires matching input/output "
            "types";
        return false;
    }

    std::ostringstream out;
    out << "module {\n";
    out << "  // tc.graph: " << GraphNameOrFallback(graph) << '\n';
    out << "  // tc.node_count: " << graph.get_nodes().size() << '\n';
    out << "  func.func @main(%arg0: " << input_ty << ") -> " << output_ty
        << " {\n";
    out << "    // tc.node[0]: op=Relu, name=" << NodeNameOrFallback(node)
        << '\n';
    out << "    return %arg0 : " << output_ty << '\n';
    out << "  }\n";
    out << "}\n";

    out_mlir_text = out.str();
    return true;
}

} // namespace

namespace tc::frontend::mlir {

bool EmitMlirModuleSkeleton(const Graph& graph,
                            std::string& out_mlir_text,
                            std::string& out_error)
{
    out_mlir_text.clear();
    out_error.clear();

    if (CanEmitSimpleReluEntry(graph)) {
        return EmitSimpleReluMain(graph, out_mlir_text, out_error);
    }

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
