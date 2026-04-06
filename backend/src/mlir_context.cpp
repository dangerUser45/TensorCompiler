#include "tc/backend/mlir_context.hpp"

#if TC_BACKEND_HAS_MLIR
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#endif

namespace tc::backend {

bool IsMlirAvailable() noexcept
{
    return TC_BACKEND_HAS_MLIR != 0;
}

std::unique_ptr<mlir::MLIRContext> CreateMlirContext()
{
#if TC_BACKEND_HAS_MLIR
    mlir::DialectRegistry registry;
    RegisterRequiredMlirDialects(registry);

    auto context = std::make_unique<mlir::MLIRContext>(registry);
    context->allowUnregisteredDialects(false);
    context->loadDialect<mlir::func::FuncDialect,
                         mlir::arith::ArithDialect,
                         mlir::tensor::TensorDialect,
                         mlir::linalg::LinalgDialect>();
    return context;
#else
    return nullptr;
#endif
}

std::vector<std::string> GetRequiredMlirDialects()
{
    if (!IsMlirAvailable()) {
        return {};
    }

    return { "func", "arith", "tensor", "linalg" };
}

void RegisterRequiredMlirDialects(mlir::DialectRegistry& registry)
{
#if TC_BACKEND_HAS_MLIR
    registry.insert<mlir::func::FuncDialect,
                    mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect>();
#else
    (void)registry;
#endif
}

std::string GetMlirSupportMessage()
{
    if (IsMlirAvailable()) {
        return "MLIR support is enabled";
    }

    return "MLIR support is disabled: MLIR package was not found at configure time";
}

} // namespace tc::backend
