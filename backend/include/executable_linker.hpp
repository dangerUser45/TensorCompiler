#pragma once

#include <string>

#include "mlir_pipeline.hpp"

namespace tc::backend {

bool LinkExecutableWithRuntime(const std::string& model_object_path,
                               const std::string& metadata_json_path,
                               const std::string& output_executable_path,
                               BackendDiagnostic& out_diagnostic);

} // namespace tc::backend
