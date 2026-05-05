#include "mlir_emitter.hpp"

#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

bool CanEmitSimpleEntry(const tc::frontend::Graph& graph) noexcept
{
    if (graph.get_input_tensors().empty() ||
        graph.get_output_tensors().size() != 1 || graph.get_nodes().empty()) {
        return false;
    }

    for (const auto& node_ptr : graph.get_nodes()) {
        if (!node_ptr) {
            return false;
        }
        const auto kind = node_ptr->get_op_kind();
        if (kind != tc::frontend::OpKind::kRelu &&
            kind != tc::frontend::OpKind::kTranspose &&
            kind != tc::frontend::OpKind::kAdd &&
            kind != tc::frontend::OpKind::kMatMul) {
            return false;
        }
        std::size_t expected_inputs = 0;
        switch (kind) {
            case tc::frontend::OpKind::kRelu:
            case tc::frontend::OpKind::kTranspose:
                expected_inputs = 1;
                break;
            case tc::frontend::OpKind::kAdd:
            case tc::frontend::OpKind::kMatMul:
                expected_inputs = 2;
                break;
            case tc::frontend::OpKind::kUnknown:
                return false;
        }
        if (node_ptr->get_inputs().size() != expected_inputs ||
            node_ptr->get_outputs().size() != 1) {
            return false;
        }
    }
    return true;
}

bool EmitSimpleMain(const tc::frontend::Graph& graph,
                    std::string& out_mlir_text,
                    std::string& out_error)
{
    out_mlir_text.clear();
    out_error.clear();

    const auto& output = *graph.get_output_tensors().front();

    std::vector<std::string> input_types;
    std::unordered_map<std::string, std::string> tensor_types;
    input_types.reserve(graph.get_input_tensors().size());
    for (const auto& input_ptr : graph.get_input_tensors()) {
        if (!input_ptr) {
            out_error = "ERROR: null input tensor in graph";
            return false;
        }
        std::string input_ty;
        if (!BuildMlirTensorType(*input_ptr, input_ty, out_error, "input")) {
            return false;
        }
        tensor_types[input_ptr->get_name()] = input_ty;
        input_types.push_back(std::move(input_ty));
    }

    std::string output_ty;
    if (!BuildMlirTensorType(output, output_ty, out_error, "output")) {
        return false;
    }
    tensor_types[output.get_name()] = output_ty;
    for (const auto& output_ptr : graph.get_output_tensors()) {
        if (!output_ptr) {
            continue;
        }
        std::string out_ty;
        if (!BuildMlirTensorType(*output_ptr, out_ty, out_error, "output")) {
            return false;
        }
        tensor_types[output_ptr->get_name()] = out_ty;
    }

    std::ostringstream out;
    out << "module {\n";
    out << "  // tc.graph: " << GraphNameOrFallback(graph) << '\n';
    out << "  // tc.node_count: " << graph.get_nodes().size() << '\n';
    out << "  func.func @main(";
    for (std::size_t i = 0; i < input_types.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "%arg" << i << ": " << input_types[i];
    }
    out << ") -> " << output_ty << " {\n";

    for (std::size_t i = 0; i < graph.get_nodes().size(); ++i) {
        const auto& node = *graph.get_nodes()[i];
        out << "    // tc.node[" << i
            << "]: op=" << ToString(node.get_op_kind())
            << ", name=" << NodeNameOrFallback(node) << '\n';
    }

    struct MlirValue final
    {
        std::string name;
        std::string type;
    };

    std::unordered_map<std::string, MlirValue> values_by_tensor;
    for (std::size_t i = 0; i < graph.get_input_tensors().size(); ++i) {
        const auto& input_ptr = graph.get_input_tensors()[i];
        if (!input_ptr) {
            continue;
        }
        values_by_tensor[input_ptr->get_name()] = {
            "%arg" + std::to_string(i),
            input_types[i],
        };
    }

    std::size_t temp_id = 0;
    auto NextTemp = [&temp_id]() { return "%t" + std::to_string(temp_id++); };
    auto ResolveInputValue = [&values_by_tensor,
                              &input_types](const std::string& tensor_name) {
        const auto it = values_by_tensor.find(tensor_name);
        if (it != values_by_tensor.end()) {
            return it->second;
        }
        return MlirValue{ "%arg0", input_types.front() };
    };
    auto ResolveTensorType = [&tensor_types](const std::string& tensor_name,
                                             const std::string& fallback_type) {
        const auto it = tensor_types.find(tensor_name);
        if (it != tensor_types.end()) {
            return it->second;
        }
        return fallback_type;
    };
    auto BuildPermutation = [](const tc::frontend::Node& node) {
        std::string text;
        for (const auto& attr : node.get_attrs()) {
            if (!attr || attr->get_name() != "perm" ||
                attr->get_data_type().id != tc::frontend::DataID::INT64) {
                continue;
            }
            const auto& perm = attr->get_values<int64_t>();
            for (std::size_t i = 0; i < perm.size(); ++i) {
                if (i != 0) {
                    text += ", ";
                }
                text += std::to_string(perm[i]);
            }
            return text;
        }
        return std::string("0");
    };

    for (const auto& node_ptr : graph.get_nodes()) {
        if (!node_ptr) {
            continue;
        }
        const auto& node = *node_ptr;
        if (node.get_outputs().empty()) {
            continue;
        }

        const std::string& output_name = node.get_outputs()[0];
        MlirValue result{ NextTemp(), output_ty };

        switch (node.get_op_kind()) {
            case tc::frontend::OpKind::kRelu: {
                const MlirValue input = ResolveInputValue(node.get_inputs()[0]);
                result.type = ResolveTensorType(output_name, input.type);
                out << "    " << result.name << " = arith.maximumf "
                    << input.name << ", " << input.name << " : " << input.type
                    << '\n';
                break;
            }

            case tc::frontend::OpKind::kAdd: {
                const MlirValue lhs = ResolveInputValue(node.get_inputs()[0]);
                const MlirValue rhs = ResolveInputValue(node.get_inputs()[1]);
                result.type = ResolveTensorType(output_name, lhs.type);
                out << "    " << result.name << " = arith.addf " << lhs.name
                    << ", " << rhs.name << " : " << lhs.type << '\n';
                break;
            }

            case tc::frontend::OpKind::kMatMul: {
                const MlirValue lhs = ResolveInputValue(node.get_inputs()[0]);
                const MlirValue rhs = ResolveInputValue(node.get_inputs()[1]);
                result.type = ResolveTensorType(output_name, output_ty);
                const std::string init = NextTemp();
                out << "    " << init << " = tensor.empty() : " << result.type
                    << '\n';
                out << "    " << result.name << " = linalg.matmul ins("
                    << lhs.name << ", " << rhs.name << " : " << lhs.type << ", "
                    << rhs.type << ") outs(" << init << " : " << result.type
                    << ") -> " << result.type << '\n';
                break;
            }

            case tc::frontend::OpKind::kTranspose: {
                const MlirValue input = ResolveInputValue(node.get_inputs()[0]);
                result.type = ResolveTensorType(output_name, output_ty);
                const std::string init = NextTemp();
                out << "    " << init << " = tensor.empty() : " << result.type
                    << '\n';
                out << "    " << result.name << " = linalg.transpose ins("
                    << input.name << " : " << input.type << ") outs(" << init
                    << " : " << result.type << ") permutation = ["
                    << BuildPermutation(node) << "]\n";
                break;
            }

            case tc::frontend::OpKind::kUnknown:
                continue;
        }

        tensor_types[output_name] = result.type;
        values_by_tensor[output_name] = std::move(result);
    }

    std::string return_value = "%arg0";
    std::string return_type = input_types.front();
    const auto output_it = values_by_tensor.find(output.get_name());
    if (output_it != values_by_tensor.end()) {
        return_value = output_it->second.name;
        return_type = output_it->second.type;
    }

    if (return_type != output_ty) {
        const std::string cast_tmp = NextTemp();
        out << "    " << cast_tmp << " = builtin.unrealized_conversion_cast "
            << return_value << " : " << return_type << " to " << output_ty
            << '\n';
        return_value = cast_tmp;
        return_type = output_ty;
    }
    out << "    return " << return_value << " : " << output_ty << '\n';
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

    if (CanEmitSimpleEntry(graph)) {
        return EmitSimpleMain(graph, out_mlir_text, out_error);
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
