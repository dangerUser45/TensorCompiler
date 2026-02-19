#include "onnx_importer.hpp"
#include "graph.hpp"
#include "onnx.pb.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>
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
using TensorsT = std::vector<std::unique_ptr<graph::Tensor>>;

bool ParseDimsFromValueInfo(const ::onnx::ValueInfoProto& vi, 
        std::vector<graph::dims_map>& out_dims,
        std::string& out_error)
{
    if  (!vi.has_type() || !vi.type().has_tensor_type() || 
        !vi.type().tensor_type().has_shape()) 
    {
        return true;
    }

    const auto& shape = vi.type().tensor_type().shape();
    out_dims.reserve(shape.dim_size());

    for (int i = 0; i < shape.dim_size(); ++i) 
    {
        graph::dims_map d{};
        d.dims = i;

        const auto& dim = shape.dim(i);
        if (!dim.has_dim_value()) {
            d.elems_per_dim = -1;
        } else {
            const auto v = dim.dim_value();
            if (v < std::numeric_limits<int>::min() || 
                v > std::numeric_limits<int>::max()) {
                out_error = "Input '" + vi.name() + "': dim_value out of int range";
                return false;
            }
            d.elems_per_dim = static_cast<int>(v);
        }

        out_dims.push_back(d);
    }

    return true;
}

bool BuildInputTensors(const ::onnx::GraphProto& g,
                        TensorsT& out_inputs,
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

        std::vector<graph::dims_map> dims;
        if (!ParseDimsFromValueInfo(in, dims, out_error)) {
            return false;
        }

        auto t = std::make_unique<graph::Tensor>();
        t->set_name(in.name());
        t->set_dims_data(dims);
        out_inputs.push_back(std::move(t));
    }
    return true;
}

bool BuildOutputTensors(const ::onnx::GraphProto& g,
                        TensorsT& out_outputs,
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

        std::vector<graph::dims_map> dims;
        if (!ParseDimsFromValueInfo(out, dims, out_error)) {
            return false;
        }

        auto t  = std::make_unique<graph::Tensor>();
        t->set_name(out.name());
        t->set_dims_data(dims);
        out_outputs.push_back(std::move(t));
    }

    return true;
}

bool BuildGraph(const ::onnx::GraphProto& g,
                Graph& out_graph, 
                std::string& out_error) 
{
    out_graph.set_name(g.name());
    
    TensorsT input_tensors;
    if (!BuildInputTensors(g, input_tensors, out_error)) {
        return false;
    }

    out_graph.set_input_tensors(std::move(input_tensors));

    TensorsT output_tensors;
    if (!BuildOutputTensors(g, output_tensors, out_error)) {
        return false;
    }

    out_graph.set_output_tensors(std::move(output_tensors));

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
