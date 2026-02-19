#pragma once

#include <string>

#include "onnx.pb.h"

namespace tc::frontend::onnx {

bool LoadOnnxModel(const std::string& path,
                   ::onnx::ModelProto& out_model,
                   std::string& out_error);

}  // namespace tc::frontend::onnx
