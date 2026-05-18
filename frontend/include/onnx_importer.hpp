#pragma once

#include <string>

namespace tc::frontend {
class Graph;
} // namespace tc::frontend

namespace tc::frontend::onnx {

bool ImportOnnxToGraph(const std::string& path,
                       Graph& out_graph,
                       std::string& out_error);

} // namespace tc::frontend::onnx
