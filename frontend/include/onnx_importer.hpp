#pragma once

#include <string>

#include "graph.hpp"
#include "onnx.pb.h"

namespace tc::frontend {
namespace onnx {

bool LoadOnnxModel(const std::string& path,
                   ::onnx::ModelProto& out_model,
                   std::string& out_error);

bool ImportOnnxToGraph(const std::string& path,
                       ::onnx::ModelProto& input_model,
                       Graph& out_graph,
                       std::string& out_error);
} // namespace onnx
} // namespace tc::frontend