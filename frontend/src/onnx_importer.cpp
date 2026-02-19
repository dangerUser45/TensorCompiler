#include "onnx_importer.hpp"

#include <fstream>
#include <stdexcept>

namespace {

bool ValidateOnnxModel(const ::onnx::ModelProto& model) {
    if (!model.has_graph()) {
        throw std::runtime_error("ERROR: ONNX model has no graph");
    }

    if(!model.ir_version()) {
        throw std::runtime_error("WARN: ONNX model has no IR version");
    }

    if(model.graph().node_size() < 0) {
        throw std::runtime_error("ERROR: ONNX graph has invalid amount of nodes");
    }
    return true;
}

bool ValidateOnnxGraph(const ::onnx::GraphProto& graph) {
    if (graph.name().empty()) {
        throw std::runtime_error("WARN: ONNX graph has no name");
    }

    if (graph.output_size() == 0) {
        throw std::runtime_error("ERROR: ONNX graph has no output");
    }

    


    return true;
}

bool ValidateOnnxNode(const ::onnx::NodeProto& node) {
    return true;
}
} //namespace

namespace tc::frontend::onnx {

bool LoadOnnxModel(const std::string& path,
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


}  // namespace tc::frontend::onnx
