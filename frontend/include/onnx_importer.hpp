#pragma once

#include <string>

#include "graph.hpp"

namespace tc::frontend {
namespace onnx {

bool ImportOnnxToGraph(const std::string& path,
                       Graph& out_graph,
                       std::string& out_error);

} // namespace onnx
} // namespace tc::frontend
