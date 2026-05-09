#pragma once

#include <string>

#include "mlir_pipeline.hpp"

namespace tc::backend {

bool EmitLlvmIrFromMlirText(const std::string& mlir_text,
                            const std::string& source_name,
                            const std::string& pass_pipeline,
                            const std::string& target_triple,
                            std::string& out_llvm_ir,
                            BackendDiagnostic& out_diagnostic);

bool EmitAsmFromMlirText(const std::string& mlir_text,
                         const std::string& source_name,
                         const std::string& pass_pipeline,
                         const std::string& target_triple,
                         std::string& out_asm,
                         BackendDiagnostic& out_diagnostic);

} // namespace tc::backend
