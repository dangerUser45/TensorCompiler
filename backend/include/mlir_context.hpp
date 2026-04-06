#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mlir {
class DialectRegistry;
class MLIRContext;
}

namespace tc::backend {

bool IsMlirAvailable() noexcept;
std::unique_ptr<mlir::MLIRContext> CreateMlirContext();
std::vector<std::string> GetRequiredMlirDialects();
void RegisterRequiredMlirDialects(mlir::DialectRegistry& registry);
std::string GetMlirSupportMessage();

} // namespace tc::backend
