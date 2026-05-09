#pragma once

#include <string>

#include "mlir_pipeline.hpp"

namespace tc::backend {

bool EmitObjectFromMlirText(const std::string& mlir_text,
                            const std::string& source_name,
                            const std::string& pass_pipeline,
                            const std::string& target_triple,
                            const std::string& output_object_path,
                            BackendDiagnostic& out_diagnostic);

} // namespace tc::backend
