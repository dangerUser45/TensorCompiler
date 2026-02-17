#include "onnx_importer.hpp"

#include <fstream>

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
