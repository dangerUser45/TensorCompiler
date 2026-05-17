#include "mlir_emitter.hpp"
#include "graph_utils.hpp"
#include "shape_inference.hpp"

#include <algorithm>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

std::string MlirTypeFromShape(const std::vector<int64_t>& shape,
                              tc::frontend::DataID dtype)
{
    const auto elem_type = MlirElemType(dtype);
    if (!elem_type.has_value()) {
        return std::string();
    }
    std::ostringstream out;
    out << "tensor<";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << "x";
        }
        if (shape[i] < 0) {
            out << "?";
        } else {
            out << shape[i];
        }
    }
    if (!shape.empty()) {
        out << "x";
    }
    out << *elem_type << ">";
    return out.str();
}

bool InferNodeOutputShape(
    const tc::frontend::Node& node,
    const std::unordered_map<std::string, std::vector<int64_t>>& shapes,
    const std::unordered_map<std::string, std::vector<int64_t>>& int64_inits,
    std::vector<int64_t>& out_shape)
{
    out_shape.clear();
    auto find_shape =
        [&shapes](const std::string& name) -> const std::vector<int64_t>* {
        const auto it = shapes.find(name);
        return it == shapes.end() ? nullptr : &it->second;
    };

    const auto& inputs = node.get_inputs();
    switch (node.get_op_kind()) {
        case tc::frontend::OpKind::kRelu: {
            if (inputs.empty()) {
                return false;
            }
            const auto* in_shape = find_shape(inputs[0]);
            if (!in_shape) {
                return false;
            }
            out_shape = *in_shape;
            return true;
        }
        case tc::frontend::OpKind::kAdd:
        case tc::frontend::OpKind::kMul: {
            if (inputs.size() < 2) {
                return false;
            }
            const auto* lhs = find_shape(inputs[0]);
            const auto* rhs = find_shape(inputs[1]);
            if (!lhs || !rhs) {
                return false;
            }
            return tc::frontend::ComputeBroadcastShape(*lhs, *rhs, out_shape);
        }
        case tc::frontend::OpKind::kMatMul: {
            if (inputs.size() < 2) {
                return false;
            }
            const auto* lhs = find_shape(inputs[0]);
            const auto* rhs = find_shape(inputs[1]);
            if (!lhs || !rhs || lhs->size() != 2 || rhs->size() != 2) {
                return false;
            }
            out_shape = { (*lhs)[0], (*rhs)[1] };
            return true;
        }
        case tc::frontend::OpKind::kReshape: {
            if (inputs.size() < 2) {
                return false;
            }
            const auto* in_shape = find_shape(inputs[0]);
            if (!in_shape) {
                return false;
            }
            const auto target_it = int64_inits.find(inputs[1]);
            if (target_it == int64_inits.end()) {
                return false;
            }
            std::string reshape_error;
            return tc::frontend::InferReshapeOutputShape(
                *in_shape, target_it->second, out_shape, reshape_error);
        }
        case tc::frontend::OpKind::kTranspose: {
            if (inputs.empty()) {
                return false;
            }
            const auto* in_shape = find_shape(inputs[0]);
            if (!in_shape) {
                return false;
            }
            std::vector<int64_t> perm;
            const auto* perm_attr = tc::frontend::FindAttr(node, "perm");
            if (perm_attr &&
                perm_attr->get_data_type().id == tc::frontend::DataID::INT64) {
                perm = perm_attr->get_values<int64_t>();
            }
            if (perm.empty()) {
                perm.reserve(in_shape->size());
                for (std::size_t i = 0; i < in_shape->size(); ++i) {
                    perm.push_back(
                        static_cast<int64_t>(in_shape->size() - 1 - i));
                }
            }
            if (perm.size() != in_shape->size()) {
                return false;
            }
            out_shape.reserve(perm.size());
            for (int64_t axis : perm) {
                if (axis < 0 ||
                    static_cast<std::size_t>(axis) >= in_shape->size()) {
                    return false;
                }
                out_shape.push_back((*in_shape)[axis]);
            }
            return true;
        }
        case tc::frontend::OpKind::kConv: {
            if (inputs.size() < 2) {
                return false;
            }
            const auto* in_shape = find_shape(inputs[0]);
            const auto* w_shape = find_shape(inputs[1]);
            if (!in_shape || !w_shape || in_shape->size() != 4 ||
                w_shape->size() != 4) {
                return false;
            }
            std::vector<int64_t> kernel{ 0, 0 };
            std::vector<int64_t> strides{ 1, 1 };
            std::vector<int64_t> pads{ 0, 0, 0, 0 };
            std::vector<int64_t> dilations{ 1, 1 };
            for (const auto& attr : node.get_attrs()) {
                if (!attr ||
                    attr->get_data_type().id != tc::frontend::DataID::INT64) {
                    continue;
                }
                const auto& v = attr->get_values<int64_t>();
                if (attr->get_name() == "kernel_shape" && v.size() == 2) {
                    kernel = v;
                } else if (attr->get_name() == "strides" && v.size() == 2) {
                    strides = v;
                } else if (attr->get_name() == "pads" && v.size() == 4) {
                    pads = v;
                } else if (attr->get_name() == "dilations" && v.size() == 2) {
                    dilations = v;
                }
            }
            if (kernel[0] == 0) {
                kernel[0] = (*w_shape)[2];
                kernel[1] = (*w_shape)[3];
            }
            const int64_t out_h =
                tc::frontend::ComputeSpatialOutputSize((*in_shape)[2],
                                                       kernel[0],
                                                       strides[0],
                                                       pads[0],
                                                       pads[2],
                                                       dilations[0]);
            const int64_t out_w =
                tc::frontend::ComputeSpatialOutputSize((*in_shape)[3],
                                                       kernel[1],
                                                       strides[1],
                                                       pads[1],
                                                       pads[3],
                                                       dilations[1]);
            out_shape = { (*in_shape)[0], (*w_shape)[0], out_h, out_w };
            return true;
        }
        case tc::frontend::OpKind::kMaxPool: {
            if (inputs.empty()) {
                return false;
            }
            const auto* in_shape = find_shape(inputs[0]);
            if (!in_shape || in_shape->size() != 4) {
                return false;
            }
            std::vector<int64_t> kernel{ 0, 0 };
            std::vector<int64_t> strides{ 1, 1 };
            std::vector<int64_t> pads{ 0, 0, 0, 0 };
            for (const auto& attr : node.get_attrs()) {
                if (!attr ||
                    attr->get_data_type().id != tc::frontend::DataID::INT64) {
                    continue;
                }
                const auto& v = attr->get_values<int64_t>();
                if (attr->get_name() == "kernel_shape" && v.size() == 2) {
                    kernel = v;
                } else if (attr->get_name() == "strides" && v.size() == 2) {
                    strides = v;
                } else if (attr->get_name() == "pads" && v.size() == 4) {
                    pads = v;
                }
            }
            if (kernel[0] == 0) {
                return false;
            }
            const int64_t out_h = tc::frontend::ComputeSpatialOutputSize(
                (*in_shape)[2], kernel[0], strides[0], pads[0], pads[2], 1);
            const int64_t out_w = tc::frontend::ComputeSpatialOutputSize(
                (*in_shape)[3], kernel[1], strides[1], pads[1], pads[3], 1);
            out_shape = { (*in_shape)[0], (*in_shape)[1], out_h, out_w };
            return true;
        }
        case tc::frontend::OpKind::kUnknown:
            return false;
    }
    return false;
}

void PrepopulateIntermediateTensorTypes(
    const tc::frontend::Graph& graph,
    std::unordered_map<std::string, std::string>& tensor_types)
{
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
    std::unordered_map<std::string, tc::frontend::DataID> dtypes;
    std::unordered_map<std::string, std::vector<int64_t>> int64_inits;

    auto register_tensor = [&](const std::string& name,
                               const std::vector<int64_t>& shape,
                               tc::frontend::DataID dtype) {
        if (name.empty()) {
            return;
        }
        shapes.emplace(name, shape);
        dtypes.emplace(name, dtype);
    };

    for (const auto& t : graph.get_input_tensors()) {
        if (t) {
            register_tensor(
                t->get_name(), t->get_shape(), t->get_data_type().id);
        }
    }
    for (const auto& t : graph.get_output_tensors()) {
        if (t) {
            register_tensor(
                t->get_name(), t->get_shape(), t->get_data_type().id);
        }
    }
    for (const auto& init : graph.get_inits()) {
        if (!init) {
            continue;
        }
        register_tensor(
            init->get_name(), init->get_shape(), init->get_data_type().id);
        if (init->get_data_type().id == tc::frontend::DataID::INT64 &&
            init->has_values()) {
            int64_inits[init->get_name()] = init->get_values<int64_t>();
        }
    }

    for (const auto& node_ptr : graph.get_nodes()) {
        if (!node_ptr) {
            continue;
        }
        std::vector<int64_t> out_shape;
        if (!InferNodeOutputShape(*node_ptr, shapes, int64_inits, out_shape)) {
            continue;
        }
        // Output dtype: use first input's dtype, or float32 fallback.
        tc::frontend::DataID out_dtype = tc::frontend::DataID::FLOAT;
        if (!node_ptr->get_inputs().empty()) {
            const auto it = dtypes.find(node_ptr->get_inputs().front());
            if (it != dtypes.end() &&
                it->second != tc::frontend::DataID::UNDEFINED) {
                out_dtype = it->second;
            }
        }
        for (const auto& output_name : node_ptr->get_outputs()) {
            if (output_name.empty()) {
                continue;
            }
            shapes[output_name] = out_shape;
            dtypes[output_name] = out_dtype;
            const std::string mlir_type =
                MlirTypeFromShape(out_shape, out_dtype);
            if (mlir_type.empty()) {
                continue;
            }
            // Do not overwrite types that were explicitly set from graph
            // input/output/initializer metadata.
            tensor_types.emplace(output_name, std::move(mlir_type));
        }
    }
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
            kind != tc::frontend::OpKind::kMul &&
            kind != tc::frontend::OpKind::kConv &&
            kind != tc::frontend::OpKind::kReshape &&
            kind != tc::frontend::OpKind::kMatMul &&
            kind != tc::frontend::OpKind::kMaxPool) {
            return false;
        }
        std::size_t expected_inputs = 0;
        switch (kind) {
            case tc::frontend::OpKind::kRelu:
            case tc::frontend::OpKind::kTranspose:
                expected_inputs = 1;
                break;
            case tc::frontend::OpKind::kAdd:
            case tc::frontend::OpKind::kMul:
            case tc::frontend::OpKind::kConv:
            case tc::frontend::OpKind::kReshape:
            case tc::frontend::OpKind::kMatMul:
                expected_inputs = 2;
                break;
            case tc::frontend::OpKind::kMaxPool:
                expected_inputs = 1;
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

bool IsTensorType(std::string_view type) noexcept
{
    return type.size() >= 8 && type.rfind("tensor<", 0) == 0 &&
           type.back() == '>';
}

bool ParseTensorTypeShape(std::string_view type,
                          std::vector<std::string>& out_dims)
{
    out_dims.clear();
    if (!IsTensorType(type)) {
        return false;
    }

    const std::string_view body = type.substr(7, type.size() - 8);
    const std::size_t last_x = body.rfind('x');
    if (last_x == std::string_view::npos) {
        return true;
    }

    const std::string_view dims_part = body.substr(0, last_x);
    std::size_t start = 0;
    while (start < dims_part.size()) {
        const std::size_t sep = dims_part.find('x', start);
        const std::size_t end =
            sep == std::string_view::npos ? dims_part.size() : sep;
        if (end == start) {
            return false;
        }
        out_dims.emplace_back(dims_part.substr(start, end - start));
        if (sep == std::string_view::npos) {
            break;
        }
        start = sep + 1;
    }
    return true;
}

bool IsBroadcastCompatibleDim(std::string_view in_dim,
                              std::string_view out_dim) noexcept
{
    if (in_dim == out_dim) {
        return true;
    }
    if (in_dim == "1" || in_dim == "?" || out_dim == "?") {
        return true;
    }
    return false;
}

bool BuildBroadcastDimensions(std::string_view input_type,
                              std::string_view output_type,
                              bool allow_channel_rank1,
                              std::string& out_dimensions)
{
    std::vector<std::string> input_dims;
    std::vector<std::string> output_dims;
    if (!ParseTensorTypeShape(input_type, input_dims) ||
        !ParseTensorTypeShape(output_type, output_dims)) {
        return false;
    }
    if (input_dims.size() > output_dims.size()) {
        return false;
    }
    if (allow_channel_rank1 && input_dims.size() == 1 &&
        output_dims.size() >= 2 &&
        IsBroadcastCompatibleDim(input_dims[0], output_dims[1])) {
        out_dimensions = "1";
        return true;
    }

    const std::size_t offset = output_dims.size() - input_dims.size();
    for (std::size_t axis = 0; axis < output_dims.size(); ++axis) {
        const std::string_view in_dim =
            axis < offset ? std::string_view("1")
                          : std::string_view(input_dims[axis - offset]);
        if (!IsBroadcastCompatibleDim(in_dim, output_dims[axis])) {
            return false;
        }
    }

    std::ostringstream dims;
    for (std::size_t i = 0; i < input_dims.size(); ++i) {
        if (i != 0) {
            dims << ", ";
        }
        dims << (offset + i);
    }
    out_dimensions = dims.str();
    return true;
}

std::string FormatFloatLiteral(float value)
{
    std::ostringstream out;
    out << std::setprecision(9) << value;
    std::string text = out.str();
    if (text.find('.') == std::string::npos &&
        text.find('e') == std::string::npos &&
        text.find('E') == std::string::npos) {
        text += ".0";
    }
    return text;
}

std::string BuildDenseFloatLiteral(const std::vector<float>& values)
{
    if (values.size() == 1) {
        return "dense<" + FormatFloatLiteral(values.front()) + ">";
    }

    std::ostringstream out;
    out << "dense<[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << FormatFloatLiteral(values[i]);
    }
    out << "]>";
    return out.str();
}

std::string EmitFloatInitializerConstant(std::ostream& out,
                                         const tc::frontend::Initializers& init,
                                         std::size_t& const_id)
{
    std::string init_type;
    std::string ignored_error;
    (void)BuildMlirTensorType(init, init_type, ignored_error, "initializer");

    const std::string const_name = "%cst" + std::to_string(const_id++);
    out << "    " << const_name << " = arith.constant "
        << BuildDenseFloatLiteral(init.get_values<float>()) << " : "
        << init_type << '\n';
    return const_name;
}

bool IntAttrEquals(const tc::frontend::Node& node,
                   std::string_view name,
                   const std::vector<int64_t>& expected)
{
    const auto* attr = tc::frontend::FindAttr(node, name);
    return attr && attr->get_data_type().id == tc::frontend::DataID::INT64 &&
           attr->get_values<int64_t>() == expected;
}

bool ConvAttrsSupportedForPlainLinalg(const tc::frontend::Node& node)
{
    return IntAttrEquals(node, "strides", { 1, 1 }) &&
           IntAttrEquals(node, "dilations", { 1, 1 });
}

bool ConvHasZeroPads(const tc::frontend::Node& node)
{
    return IntAttrEquals(node, "pads", { 0, 0, 0, 0 });
}

bool ReadIntAttr(const tc::frontend::Node& node,
                 std::string_view name,
                 std::size_t expected_size,
                 std::vector<int64_t>& out_values)
{
    out_values.clear();
    const auto* attr = tc::frontend::FindAttr(node, name);
    if (!attr || attr->get_data_type().id != tc::frontend::DataID::INT64) {
        return false;
    }
    const auto& values = attr->get_values<int64_t>();
    if (values.size() != expected_size) {
        return false;
    }
    out_values = values;
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
    out << "  func.func @tc_model(";
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

    std::unordered_map<std::string, const tc::frontend::Initializers*>
        initializers_by_name;
    for (const auto& init_ptr : graph.get_inits()) {
        if (!init_ptr) {
            continue;
        }
        if (init_ptr->get_data_type().id != tc::frontend::DataID::FLOAT &&
            init_ptr->get_data_type().id != tc::frontend::DataID::INT64) {
            out_error = "ERROR: unsupported initializer dtype for '" +
                        init_ptr->get_name() + "' for production MLIR";
            return false;
        }

        std::string init_type;
        if (!BuildMlirTensorType(
                *init_ptr, init_type, out_error, "initializer")) {
            return false;
        }
        tensor_types[init_ptr->get_name()] = init_type;
        if (init_ptr->get_data_type().id == tc::frontend::DataID::FLOAT) {
            initializers_by_name[init_ptr->get_name()] = init_ptr.get();
        }
    }

    // Pre-populate intermediate tensor types via op-specific shape inference
    // so that ResolveTensorType returns correct types for nodes whose output
    // is not the final graph output. Without this, multi-node graphs (e.g.
    // MNIST) silently fall back to the graph output type for intermediates.
    PrepopulateIntermediateTensorTypes(graph, tensor_types);

    std::size_t const_id = 0;
    std::size_t temp_id = 0;
    auto NextTemp = [&temp_id]() { return "%t" + std::to_string(temp_id++); };
    auto ResolveInputValue = [&values_by_tensor,
                              &initializers_by_name,
                              &tensor_types,
                              &const_id,
                              &out,
                              &input_types](const std::string& tensor_name) {
        const auto it = values_by_tensor.find(tensor_name);
        if (it != values_by_tensor.end()) {
            return it->second;
        }

        const auto init_it = initializers_by_name.find(tensor_name);
        if (init_it != initializers_by_name.end()) {
            const auto& init = *init_it->second;
            const std::string const_name =
                EmitFloatInitializerConstant(out, init, const_id);
            const std::string init_type = tensor_types.at(tensor_name);
            const MlirValue value{ const_name, init_type };
            values_by_tensor[tensor_name] = value;
            return value;
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
    auto MaterializeToType = [&out, &NextTemp](const MlirValue& value,
                                               const std::string& target_type,
                                               bool allow_channel_rank1) {
        if (value.type == target_type) {
            return value;
        }

        std::string dimensions;
        if (BuildBroadcastDimensions(
                value.type, target_type, allow_channel_rank1, dimensions)) {
            const std::string init = NextTemp();
            out << "    " << init << " = tensor.empty() : " << target_type
                << '\n';
            const std::string bcast = NextTemp();
            out << "    " << bcast << " = linalg.broadcast ins(" << value.name
                << " : " << value.type << ") outs(" << init << " : "
                << target_type << ") dimensions = [" << dimensions << "]\n";
            return MlirValue{ bcast, target_type };
        }

        const std::string cast_tmp = NextTemp();
        out << "    " << cast_tmp << " = builtin.unrealized_conversion_cast "
            << value.name << " : " << value.type << " to " << target_type
            << '\n';
        return MlirValue{ cast_tmp, target_type };
    };
    auto BuildZeroLiteral = [](const std::string& type) {
        if (IsTensorType(type)) {
            return std::string("dense<0.0>");
        }
        return std::string("0.0");
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
        std::optional<MlirValue> result;

        switch (node.get_op_kind()) {
            case tc::frontend::OpKind::kRelu: {
                const MlirValue input = ResolveInputValue(node.get_inputs()[0]);
                const std::string zero = NextTemp();
                out << "    " << zero << " = arith.constant "
                    << BuildZeroLiteral(input.type) << " : " << input.type
                    << '\n';
                result =
                    MlirValue{ NextTemp(),
                               ResolveTensorType(output_name, input.type) };
                out << "    " << result->name << " = arith.maximumf "
                    << input.name << ", " << zero << " : " << input.type
                    << '\n';
                break;
            }

            case tc::frontend::OpKind::kAdd: {
                MlirValue lhs = ResolveInputValue(node.get_inputs()[0]);
                MlirValue rhs = ResolveInputValue(node.get_inputs()[1]);
                const std::string result_type =
                    ResolveTensorType(output_name, lhs.type);
                const bool allow_channel_bias =
                    tc::frontend::IsSyntheticBiasAdd(node);

                lhs = MaterializeToType(lhs, result_type, allow_channel_bias);
                rhs = MaterializeToType(rhs, result_type, allow_channel_bias);

                result = MlirValue{ NextTemp(), result_type };
                out << "    " << result->name << " = arith.addf " << lhs.name
                    << ", " << rhs.name << " : " << result_type << '\n';
                break;
            }

            case tc::frontend::OpKind::kMul: {
                MlirValue lhs = ResolveInputValue(node.get_inputs()[0]);
                MlirValue rhs = ResolveInputValue(node.get_inputs()[1]);
                const std::string result_type =
                    ResolveTensorType(output_name, lhs.type);

                lhs = MaterializeToType(lhs, result_type, false);
                rhs = MaterializeToType(rhs, result_type, false);

                result = MlirValue{ NextTemp(), result_type };
                out << "    " << result->name << " = arith.mulf " << lhs.name
                    << ", " << rhs.name << " : " << result_type << '\n';
                break;
            }

            case tc::frontend::OpKind::kMatMul: {
                const MlirValue lhs = ResolveInputValue(node.get_inputs()[0]);
                const MlirValue rhs = ResolveInputValue(node.get_inputs()[1]);
                const std::string result_type =
                    ResolveTensorType(output_name, output_ty);
                const std::string init = NextTemp();
                out << "    " << init << " = tensor.empty() : " << result_type
                    << '\n';
                result = MlirValue{ NextTemp(), result_type };
                out << "    " << result->name << " = linalg.matmul ins("
                    << lhs.name << ", " << rhs.name << " : " << lhs.type << ", "
                    << rhs.type << ") outs(" << init << " : " << result_type
                    << ") -> " << result_type << '\n';
                break;
            }

            case tc::frontend::OpKind::kReshape: {
                const MlirValue input = ResolveInputValue(node.get_inputs()[0]);
                const std::string result_type =
                    ResolveTensorType(output_name, output_ty);
                if (input.type == result_type) {
                    result = input;
                } else {
                    result = MlirValue{ NextTemp(), result_type };
                    out << "    " << result->name << " = tensor.reshape "
                        << input.name << " : " << input.type << " to "
                        << result_type << '\n';
                }
                break;
            }

            case tc::frontend::OpKind::kMaxPool: {
                const MlirValue input = ResolveInputValue(node.get_inputs()[0]);
                const std::string result_type =
                    ResolveTensorType(output_name, output_ty);
                const std::string init = NextTemp();
                std::vector<int64_t> kernel_shape;
                std::vector<int64_t> strides;
                if (!ReadIntAttr(node, "kernel_shape", 2, kernel_shape) ||
                    !ReadIntAttr(node, "strides", 2, strides)) {
                    out_error = "ERROR: unsupported MaxPool attributes for "
                                "production MLIR; kernel_shape and strides "
                                "must be INT64[2]";
                    return false;
                }
                out << "    " << init << " = tensor.empty() : " << result_type
                    << '\n';
                result = MlirValue{ NextTemp(), result_type };
                out << "    " << result->name
                    << " = linalg.pooling_nchw_max {kernel = ["
                    << kernel_shape[0] << ", " << kernel_shape[1]
                    << "], strides = [" << strides[0] << ", " << strides[1]
                    << "]} ins(" << input.name << " : " << input.type
                    << ") outs(" << init << " : " << result_type << ") -> "
                    << result_type << '\n';
                break;
            }

            case tc::frontend::OpKind::kTranspose: {
                const MlirValue input = ResolveInputValue(node.get_inputs()[0]);
                const std::string result_type =
                    ResolveTensorType(output_name, output_ty);
                const std::string init = NextTemp();
                out << "    " << init << " = tensor.empty() : " << result_type
                    << '\n';
                result = MlirValue{ NextTemp(), result_type };
                out << "    " << result->name << " = linalg.transpose ins("
                    << input.name << " : " << input.type << ") outs(" << init
                    << " : " << result_type << ") permutation = ["
                    << BuildPermutation(node) << "]\n";
                break;
            }

            case tc::frontend::OpKind::kConv: {
                if (!ConvAttrsSupportedForPlainLinalg(node)) {
                    out_error =
                        "ERROR: unsupported Conv attributes for production "
                        "MLIR; only strides=[1,1] and dilations=[1,1] are "
                        "currently supported";
                    return false;
                }
                const MlirValue input = ResolveInputValue(node.get_inputs()[0]);
                const MlirValue weight =
                    ResolveInputValue(node.get_inputs()[1]);
                const std::string result_type =
                    ResolveTensorType(output_name, output_ty);
                const std::string init = NextTemp();
                out << "    " << init << " = tensor.empty() : " << result_type
                    << '\n';
                result = MlirValue{ NextTemp(), result_type };

                // Padded Conv stays a single op carrying an explicit
                // {pads = [...]} attribute. The backend's regex parser reads
                // this attribute on the same line; using tensor.pad with a
                // region body would require a multi-line MLIR parser.
                std::string pads_clause;
                if (!ConvHasZeroPads(node)) {
                    const auto* pads_attr =
                        tc::frontend::FindAttr(node, "pads");
                    const auto& pads = pads_attr->get_values<int64_t>();
                    std::ostringstream pc;
                    pc << " {pads = [" << pads[0] << ", " << pads[1] << ", "
                       << pads[2] << ", " << pads[3] << "]}";
                    pads_clause = pc.str();
                }

                out << "    " << result->name << " = linalg.conv_2d_nchw_fchw"
                    << pads_clause << " ins(" << input.name << ", "
                    << weight.name << " : " << input.type << ", " << weight.type
                    << ") outs(" << init << " : " << result_type << ") -> "
                    << result_type << '\n';
                break;
            }

            case tc::frontend::OpKind::kUnknown:
                continue;
        }

        if (result.has_value()) {
            tensor_types[output_name] = result->type;
            values_by_tensor[output_name] = std::move(*result);
        }
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

bool EmitMlirModule(const Graph& graph,
                    std::string& out_mlir_text,
                    std::string& out_error,
                    EmitMode mode)
{
    out_mlir_text.clear();
    out_error.clear();

    if (CanEmitSimpleEntry(graph)) {
        return EmitSimpleMain(graph, out_mlir_text, out_error);
    }

    if (mode == EmitMode::kStrict) {
        out_error = "ERROR: unsupported graph for production MLIR";
        return false;
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
