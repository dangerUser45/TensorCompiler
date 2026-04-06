#include "onnx_importer.hpp"
#include "graph.hpp"
#include "onnx.pb.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <google/protobuf/repeated_field.h>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
bool SetError(std::string& out_error, std::string msg)
{
    out_error = std::move(msg);
    return false;
}
} // namespace

namespace tc::frontend {

bool onnx::LoadOnnxModel(const std::string& path,
                         ::onnx::ModelProto& out_model,
                         std::string& out_error)
{
    out_model.Clear();
    out_error.clear();

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        out_error = "Failed to open file: " + path;
        return false;
    }

    if (!out_model.ParseFromIstream(&input)) {
        out_error = "Failed to parse ONNX protobuf: " + path;
        return false;
    }

    if (!out_model.has_graph()) {
        out_error = "ONNX model has no graph: " + path;
        return false;
    }

    return true;
}

namespace detail {

bool IsSupportedMvpOperator(const std::string& op_type)
{
    static const std::unordered_set<std::string> kSupportedOps = {
        "Relu",
        "Add",
        "MatMul",
        "Transpose",
    };

    return kSupportedOps.find(op_type) != kSupportedOps.end();
}

bool ParseDimsFromValueInfo(const ::onnx::ValueInfoProto& vi,
                            std::vector<int64_t>& out_shape,
                            std::string& /*out_error*/)
{
    out_shape.clear();

    if (!vi.has_type() || !vi.type().has_tensor_type() ||
        !vi.type().tensor_type().has_shape()) {
        return true; // it is possible
    }

    const auto& shape = vi.type().tensor_type().shape();
    out_shape.reserve(shape.dim_size());

    for (int i = 0; i < shape.dim_size(); ++i) {
        const auto& dim = shape.dim(i);
        if (dim.has_dim_value()) {
            out_shape.push_back(dim.dim_value());
        } else {
            out_shape.push_back(-1);
        }
    }

    return true;
}

template<typename T>
bool ParseRawVector(const std::string& raw,
                    std::vector<T>& out,
                    std::string& out_error,
                    const std::string& init_name)
{
    if (raw.size() % sizeof(T) != 0) {
        return SetError(out_error,
                        "ERROR: initializer '" + init_name +
                            "' has invalid raw_data size");
    }
    out.resize(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
}

template<typename OutT, typename InT>
bool FillNumericFromFieldOrRaw(
    const ::onnx::TensorProto& tp,
    const ::google::protobuf::RepeatedField<InT>& field,
    Initializers& out_init,
    std::string& out_error,
    const char* type_name)
{
    std::vector<OutT> values;
    if (!field.empty()) {
        values.reserve(field.size());
        for (const auto x : field) {
            values.push_back(static_cast<OutT>(x));
        }
    } else if (tp.has_raw_data()) {
        if (!ParseRawVector<OutT>(tp.raw_data(), values, out_error, tp.name()))
            return false;
    } else {
        return SetError(out_error,
                        "ERROR: initializer '" + tp.name() + "' has no " +
                            std::string(type_name) + " data");
    }

    out_init.set_values(std::move(values));
    return true;
}

bool FillInitializerValues(const ::onnx::TensorProto& tp,
                           Initializers& out_init,
                           std::string& out_error)
{
    const auto dt = static_cast<::onnx::TensorProto_DataType>(tp.data_type());

    switch (dt) {
        case ::onnx::TensorProto_DataType_FLOAT:
            return FillNumericFromFieldOrRaw<float>(
                tp, tp.float_data(), out_init, out_error, "FLOAT");

        case ::onnx::TensorProto_DataType_DOUBLE:
            return FillNumericFromFieldOrRaw<double>(
                tp, tp.double_data(), out_init, out_error, "DOUBLE");

        case ::onnx::TensorProto_DataType_INT32:
            return FillNumericFromFieldOrRaw<int32_t>(
                tp, tp.int32_data(), out_init, out_error, "INT32");

        case ::onnx::TensorProto_DataType_INT64:
            return FillNumericFromFieldOrRaw<int64_t>(
                tp, tp.int64_data(), out_init, out_error, "INT64");

        case ::onnx::TensorProto_DataType_UINT8:
            return FillNumericFromFieldOrRaw<uint8_t>(
                tp, tp.int32_data(), out_init, out_error, "UINT8");

        case ::onnx::TensorProto_DataType_INT8:
            return FillNumericFromFieldOrRaw<int8_t>(
                tp, tp.int32_data(), out_init, out_error, "INT8");

        case ::onnx::TensorProto_DataType_UINT16:
            return FillNumericFromFieldOrRaw<uint16_t>(
                tp, tp.int32_data(), out_init, out_error, "UINT16");

        case ::onnx::TensorProto_DataType_INT16:
            return FillNumericFromFieldOrRaw<int16_t>(
                tp, tp.int32_data(), out_init, out_error, "INT16");

        case ::onnx::TensorProto_DataType_UINT32:
            return FillNumericFromFieldOrRaw<uint32_t>(
                tp, tp.uint64_data(), out_init, out_error, "UINT32");

        case ::onnx::TensorProto_DataType_UINT64:
            return FillNumericFromFieldOrRaw<uint64_t>(
                tp, tp.uint64_data(), out_init, out_error, "UINT64");

        case ::onnx::TensorProto_DataType_STRING: {
            std::vector<std::string> v;
            v.reserve(tp.string_data_size());
            for (int i = 0; i < tp.string_data_size(); ++i)
                v.push_back(tp.string_data(i));
            out_init.set_values(std::move(v));
            return true;
        }

        default:
            return SetError(out_error,
                            "ERROR: unsupported initializer type for '" +
                                tp.name() +
                                "': " + ::onnx::TensorProto_DataType_Name(dt));
    }
}

bool BuildInitializers(const ::onnx::GraphProto& g,
                       Graph::InitVecT& out_inits,
                       std::string& out_error)
{
    out_inits.clear();

    out_inits.reserve(g.initializer_size());
    for (int i = 0; i < g.initializer_size(); ++i) {
        const auto& tp = g.initializer(i);

        if (tp.name().empty()) {
            return SetError(out_error, "ERROR: initializer has empty name");
        }
        if (!tp.has_data_type()) {
            return SetError(out_error,
                            "ERROR: initializer '" + tp.name() +
                                "' has no data_type");
        }

        auto init = std::make_unique<Initializers>();
        init->set_name(tp.name());

        std::vector<int64_t> shape;
        shape.reserve(tp.dims_size());
        for (int d = 0; d < tp.dims_size(); ++d)
            shape.push_back(tp.dims(d));
        init->set_shape(std::move(shape));

        if (!FillInitializerValues(tp, *init, out_error))
            return false;

        out_inits.push_back(std::move(init));
    }

    return true;
}

std::string BuildNodeContext(const ::onnx::NodeProto& node, int node_index)
{
    if (!node.name().empty()) {
        return "node[" + std::to_string(node_index) + "]('" + node.name() +
               "')";
    }
    return "node[" + std::to_string(node_index) + "]";
}

::onnx::AttributeProto_AttributeType ResolveAttributeType(
    const ::onnx::AttributeProto& attr)
{
    if (attr.has_type()) {
        return attr.type();
    }

    if (attr.has_f())
        return ::onnx::AttributeProto_AttributeType_FLOAT;
    if (attr.has_i())
        return ::onnx::AttributeProto_AttributeType_INT;
    if (attr.has_s())
        return ::onnx::AttributeProto_AttributeType_STRING;
    if (attr.floats_size() > 0)
        return ::onnx::AttributeProto_AttributeType_FLOATS;
    if (attr.ints_size() > 0)
        return ::onnx::AttributeProto_AttributeType_INTS;
    if (attr.strings_size() > 0)
        return ::onnx::AttributeProto_AttributeType_STRINGS;

    return ::onnx::AttributeProto_AttributeType_UNDEFINED;
}

bool FillAttributeValues(const ::onnx::AttributeProto& src,
                         Attribute& dst,
                         std::string& out_error,
                         const std::string& attr_context)
{
    const auto type = ResolveAttributeType(src);

    switch (type) {
        case ::onnx::AttributeProto_AttributeType_FLOAT: {
            if (!src.has_f()) {
                return SetError(out_error,
                                "ERROR: " + attr_context +
                                    " FLOAT has no value");
            }
            dst.set_values<float>({ src.f() });
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_INT: {
            if (!src.has_i()) {
                return SetError(out_error,
                                "ERROR: " + attr_context + " INT has no value");
            }
            dst.set_values<int64_t>({ src.i() });
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_STRING: {
            if (!src.has_s()) {
                return SetError(out_error,
                                "ERROR: " + attr_context +
                                    " STRING has no value");
            }
            dst.set_values<std::string>({ src.s() });
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_FLOATS: {
            std::vector<float> values;
            values.reserve(src.floats_size());
            for (int i = 0; i < src.floats_size(); ++i) {
                values.push_back(src.floats(i));
            }
            dst.set_values<float>(std::move(values));
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_INTS: {
            std::vector<int64_t> values;
            values.reserve(src.ints_size());
            for (int i = 0; i < src.ints_size(); ++i) {
                values.push_back(src.ints(i));
            }
            dst.set_values<int64_t>(std::move(values));
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_STRINGS: {
            std::vector<std::string> values;
            values.reserve(src.strings_size());
            for (int i = 0; i < src.strings_size(); ++i) {
                values.push_back(src.strings(i));
            }
            dst.set_values<std::string>(std::move(values));
            return true;
        }

        default:
            return SetError(
                out_error,
                "ERROR: " + attr_context + " unsupported type: " +
                    ::onnx::AttributeProto_AttributeType_Name(type));
    }
}

bool ParseAttribute(const ::onnx::AttributeProto& src,
                    std::unique_ptr<Attribute>& out_attr,
                    std::string& out_error,
                    const std::string& node_context,
                    int attr_index)
{
    const std::string attr_context =
        node_context + ".attribute[" + std::to_string(attr_index) + "]";

    auto attr = std::make_unique<Attribute>();
    attr->set_name(src.name());

    if (!FillAttributeValues(src, *attr, out_error, attr_context)) {
        return false;
    }

    out_attr = std::move(attr);
    return true;
}

bool BuildNodeAttributes(const ::onnx::NodeProto& src,
                         Node::AttrVecT& out_attrs,
                         std::string& out_error,
                         const std::string& node_context)
{
    out_attrs.clear();
    out_attrs.reserve(src.attribute_size());

    for (int i = 0; i < src.attribute_size(); ++i) {
        std::unique_ptr<Attribute> attr;
        if (!ParseAttribute(
                src.attribute(i), attr, out_error, node_context, i)) {
            return false;
        }
        out_attrs.push_back(std::move(attr));
    }

    return true;
}

bool ParseNode(const ::onnx::NodeProto& src,
               std::unique_ptr<Node>& out_node,
               std::string& out_error,
               int node_index)
{
    const std::string node_context = BuildNodeContext(src, node_index);
    const std::string& op_type = src.op_type();

    if (op_type.empty()) {
        return SetError(out_error, "ERROR: " + node_context + " has empty op");
    }

    if (!IsSupportedMvpOperator(op_type)) {
        return SetError(
            out_error,
            "ERROR: " + node_context + " unsupported operator: " + op_type);
    }

    auto node = std::make_unique<Node>();
    node->set_name_node(src.name());
    node->set_name_op(op_type);

    std::vector<std::string> inputs;
    inputs.reserve(src.input_size());
    for (int i = 0; i < src.input_size(); ++i) {
        inputs.push_back(src.input(i));
    }
    node->set_inputs(std::move(inputs));

    std::vector<std::string> outputs;
    outputs.reserve(src.output_size());
    for (int i = 0; i < src.output_size(); ++i) {
        outputs.push_back(src.output(i));
    }
    node->set_outputs(std::move(outputs));

    Node::AttrVecT attrs;
    if (!BuildNodeAttributes(src, attrs, out_error, node_context)) {
        return false;
    }
    node->set_attr(std::move(attrs));

    out_node = std::move(node);
    return true;
}

bool BuildNodes(const ::onnx::GraphProto& g,
                Graph::NodeVecT& out_nodes,
                std::string& out_error)
{
    out_nodes.clear();

    out_nodes.reserve(g.node_size());
    for (int i = 0; i < g.node_size(); ++i) {
        std::unique_ptr<Node> node;
        if (!ParseNode(g.node(i), node, out_error, i)) {
            return false;
        }
        out_nodes.push_back(std::move(node));
    }

    return true;
}

bool BuildInputTensors(const ::onnx::GraphProto& g,
                       Graph::TensVecT& out_inputs,
                       std::string& out_error)
{
    out_inputs.clear();
    out_error.clear();

    std::unordered_set<std::string> init_names;
    init_names.reserve(g.initializer_size());
    for (int i = 0; i < g.initializer_size(); ++i) {
        init_names.insert(g.initializer(i).name());
    }

    out_inputs.reserve(g.input_size());
    for (int i = 0; i < g.input_size(); ++i) {
        const auto& in = g.input(i);
        if (in.name().empty()) {
            out_error = "Input '" + in.name() + "' has empty name";
            return false;
        }

        if (init_names.find(in.name()) != init_names.end()) {
            continue;
        }

        std::vector<int64_t> shape;
        if (!ParseDimsFromValueInfo(in, shape, out_error)) {
            return false;
        }

        auto t = std::make_unique<TensorInfo>();
        t->set_name(in.name());
        t->set_shape(shape);
        out_inputs.push_back(std::move(t));
    }
    return true;
}

bool BuildOutputTensors(const ::onnx::GraphProto& g,
                        Graph::TensVecT& out_outputs,
                        std::string& out_error)
{
    out_outputs.clear();
    out_error.clear();

    out_outputs.reserve(g.output_size());

    for (int i = 0; i < g.output_size(); ++i) {
        const auto& out = g.output(i);
        if (out.name().empty()) {
            out_error = "Output has empty name";
            return false;
        }

        std::vector<int64_t> shape;
        if (!ParseDimsFromValueInfo(out, shape, out_error)) {
            return false;
        }

        auto t = std::make_unique<TensorInfo>();
        t->set_name(out.name());
        t->set_shape(shape);
        out_outputs.push_back(std::move(t));
    }

    return true;
}

bool BuildGraph(const ::onnx::GraphProto& g,
                Graph& out_graph,
                std::string& out_error)
{
    out_graph.set_name(g.name());

    Graph::TensVecT input_tensors;
    if (!BuildInputTensors(g, input_tensors, out_error)) {
        return false;
    }

    out_graph.set_input_tensors(std::move(input_tensors));

    Graph::TensVecT output_tensors;
    if (!BuildOutputTensors(g, output_tensors, out_error)) {
        return false;
    }

    out_graph.set_output_tensors(std::move(output_tensors));

    Graph::InitVecT initializers;
    if (!BuildInitializers(g, initializers, out_error)) {
        return false;
    }

    out_graph.set_inits(std::move(initializers));

    Graph::NodeVecT nodes;
    if (!BuildNodes(g, nodes, out_error)) {
        return false;
    }

    out_graph.set_nodes(std::move(nodes));

    return true;
}

} // namespace detail

bool onnx::ImportOnnxToGraph(const std::string& path,
                             ::onnx::ModelProto& input_model,
                             Graph& out_graph,
                             std::string& out_error)
{
    if (!onnx::LoadOnnxModel(path, input_model, out_error)) {
        return false;
    }

    return detail::BuildGraph(input_model.graph(), out_graph, out_error);
}
} // namespace tc::frontend
