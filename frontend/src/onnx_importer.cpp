#include "onnx_importer.hpp"
#include "graph.hpp"
#include "onnx.pb.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <google/protobuf/repeated_field.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>

namespace {
bool SetError(std::string& out_error, std::string msg) {
    out_error = std::move(msg);
    return false;
}

//TODO: добавить больше проверок на warnings 
bool ValidateInitializers(const ::onnx::GraphProto& graph, std::string& out_error)
{
    std::unordered_map<std::string, int> init_names;
    for (int i = 0; i < graph.initializer_size(); ++i) 
    {
        const auto& name = graph.initializer(i).name();
        if (name.empty()) {
            return SetError(out_error, "ERROR: ONNX graph has empty initializer name");
        }
        if (init_names.find(name) != init_names.end()) {
            return SetError(out_error, "ERROR: ONNX graph has duplicate initializer name");
        }
        init_names[name] = 1;
    }
    return true;
}

bool ValidateOnnxNode(const ::onnx::NodeProto& node, std::string& out_error) 
{
    if (node.output_size() == 0) {
        return SetError(out_error, "ERROR: ONNX node has no output");
    }

    if (node.op_type().empty()) {
        return SetError(out_error, "ERROR: ONNX node has no op type");
    }

    return true;
}

bool ValidateNodeOutputDups(const ::onnx::GraphProto& graph, std::string& out_error)
{
    std::unordered_set<std::string> global_outputs;
    for (int ni = 0; ni < graph.node_size(); ++ni) 
    {
        const auto& node = graph.node(ni);
        std::unordered_set<std::string> local_outputs;

        for (int oi = 0; oi < node.output_size(); ++oi) 
        {
            const std::string& out = node.output(oi);
            if (out.empty()) {
                return SetError(out_error, "ERROR: empty output name");
            }

            if(!local_outputs.insert(out).second) {
                return SetError(out_error, "ERROR: duplicate output name inside same node");
            }

            if(!global_outputs.insert(out).second) {
                return SetError(out_error, "ERROR: duplicate output name across nodes");
            }
        }
    }
    return true;
}

bool ValidateOnnxGraph(const ::onnx::GraphProto& graph, std::string& out_error) 
{
    out_error.clear();

    if (graph.name().empty()) {
        return SetError(out_error, "WARN: ONNX graph has no name");
    }

    if (graph.output_size() == 0) {
        return SetError(out_error, "ERROR: ONNX graph has no output");
    }

    if (!ValidateInitializers(graph, out_error)) return false;
    if (!ValidateNodeOutputDups(graph, out_error)) return false;

    for (int i = 0; i < graph.node_size(); ++i) 
    {
        const auto& node = graph.node(i);
        if (!ValidateOnnxNode(node, out_error)) {
            return false;
        }
    }

    return true;
}

bool ValidateOnnxModel(const ::onnx::ModelProto& model, std::string& out_error) 
{
    out_error.clear();

    if (!model.has_graph()) {
        return SetError(out_error, "ERROR: ONNX model has no graph");
    }

    if(!model.ir_version()) {
        return SetError(out_error, "WARN: ONNX model has no IR version");
    }

    if(model.graph().node_size() < 0) {
        return SetError(out_error, "ERROR: ONNX graph has invalid amount of nodes");
    }

    if(model.opset_import_size() == 0) {
        return SetError(out_error, "WARN: ONNX model has no opset import");
    }

    return ValidateOnnxGraph(model.graph(), out_error);
}


} //namespace

namespace tc::frontend {

bool onnx::LoadOnnxModel(const std::string& path,
                   ::onnx::ModelProto& out_model,
                   std::string& out_error) {
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

bool ParseDimsFromValueInfo(const ::onnx::ValueInfoProto& vi, 
        std::vector<int64_t>& out_shape,
        std::string& /*out_error*/)
{
    out_shape.clear();

    if  (!vi.has_type() || !vi.type().has_tensor_type() || 
        !vi.type().tensor_type().has_shape()) 
    {
        return true; //it is possible
    }

    const auto& shape = vi.type().tensor_type().shape();
    out_shape.reserve(shape.dim_size());

    for (int i = 0; i < shape.dim_size(); ++i) 
    {
        const auto& dim = shape.dim(i);
        if (dim.has_dim_value()) {
            out_shape.push_back(dim.dim_value());
        } else {
            out_shape.push_back(-1);
        }
    }

    return true;
}

template <typename T>
bool ParseRawVector(const std::string& raw,
                    std::vector<T>& out,
                    std::string& out_error,
                    const std::string& init_name) {
    if (raw.size() % sizeof(T) != 0) {
        return SetError(out_error, "ERROR: initializer '" + init_name 
                + "' has invalid raw_data size");
    }
    out.resize(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
}

template <typename OutT, typename InT>
bool FillNumericFromFieldOrRaw(const ::onnx::TensorProto& tp,
                               const ::google::protobuf::RepeatedField<InT>& field,
                               graph::Initializers& out_init,
                               std::string& out_error,
                               const char* type_name) {
    std::vector<OutT> values;
    if (!field.empty()) {
        values.reserve(field.size());
        for (const auto x : field) {
            values.push_back(static_cast<OutT>(x));
        }
    } else if (tp.has_raw_data()) {
        if (!ParseRawVector<OutT>(tp.raw_data(), values, out_error, tp.name())) return false;
    } else {
        return SetError(out_error, "ERROR: initializer '" + tp.name() +
                "' has no " + std::string(type_name) + " data");
    }

    out_init.set_values(std::move(values));
    return true;
}

bool FillInitializerValues(const ::onnx::TensorProto& tp,
                           graph::Initializers& out_init,
                           std::string& out_error) {
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
            for (int i = 0; i < tp.string_data_size(); ++i) v.push_back(tp.string_data(i));
            out_init.set_values(std::move(v));
            return true;
        }

        default:
            return SetError(
                out_error,
                "ERROR: unsupported initializer type for '" + tp.name() + "': " +
                ::onnx::TensorProto_DataType_Name(dt));
    }
}

bool BuildInitializers(const ::onnx::GraphProto& g,
                       Graph::InitVecT& out_inits,
                       std::string& out_error) {
    out_inits.clear();

    out_inits.reserve(g.initializer_size());
    for (int i = 0; i < g.initializer_size(); ++i) {
        const auto& tp = g.initializer(i);

        if (tp.name().empty()) {
            return SetError(out_error, "ERROR: initializer has empty name");
        }
        if (!tp.has_data_type()) {
            return SetError(out_error, "ERROR: initializer '" + tp.name() + "' has no data_type");
        }

        auto init = std::make_unique<graph::Initializers>();
        init->set_name(tp.name());

        std::vector<int64_t> shape;
        shape.reserve(tp.dims_size());
        for (int d = 0; d < tp.dims_size(); ++d) shape.push_back(tp.dims(d));
        init->set_shape(std::move(shape));

        if (!FillInitializerValues(tp, *init, out_error)) return false;

        out_inits.push_back(std::move(init));
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
    for (int i = 0; i < g.initializer_size(); ++i) 
    {
        init_names.insert(g.initializer(i).name());
    }

    out_inputs.reserve(g.input_size());
    for (int i = 0; i < g.input_size(); ++i) 
    {
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

        auto t = std::make_unique<graph::TensorInfo>();
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

    for (int i = 0; i < g.output_size(); ++i)
    {
        const auto& out = g.output(i);
        if (out.name().empty()) {
            out_error = "Output has empty name";
            return false;
        }

        std::vector<int64_t> shape;
        if (!ParseDimsFromValueInfo(out, shape, out_error)) {
            return false;
        }

        auto t  = std::make_unique<graph::TensorInfo>();
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

    if (!ValidateOnnxModel(input_model, out_error)) {
        return false;
    }

    return detail::BuildGraph(input_model.graph(), out_graph, out_error);
}
}  // namespace tc::frontend
