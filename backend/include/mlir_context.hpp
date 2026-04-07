#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mlir {
class DialectRegistry;
class MLIRContext;
}

namespace tc::backend {

struct MlirContextDeleter final
{
    void operator()(mlir::MLIRContext* context) const;
};

using MlirContextPtr = std::unique_ptr<mlir::MLIRContext, MlirContextDeleter>;

bool IsMlirAvailable() noexcept;
MlirContextPtr CreateMlirContext();
std::vector<std::string> GetRequiredMlirDialects();
void RegisterRequiredMlirDialects(mlir::DialectRegistry& registry);
std::string GetMlirSupportMessage();

} // namespace tc::backend
