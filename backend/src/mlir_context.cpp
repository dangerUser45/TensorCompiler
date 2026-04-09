#include "mlir_context.hpp"

#if TC_BACKEND_HAS_MLIR
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#endif

namespace tc::backend {

void MlirContextDeleter::operator()(mlir::MLIRContext* context) const
{
#if TC_BACKEND_HAS_MLIR
    delete context;
#else
    (void)context;
#endif
}

bool IsMlirAvailable() noexcept
{
    return TC_BACKEND_HAS_MLIR != 0;
}

MlirContextPtr CreateMlirContext()
{
#if TC_BACKEND_HAS_MLIR
    mlir::DialectRegistry registry;
    RegisterRequiredMlirDialects(registry);

    MlirContextPtr context(new mlir::MLIRContext(registry));
    context->allowUnregisteredDialects(false);
    context->loadDialect<mlir::func::FuncDialect,
                         mlir::arith::ArithDialect,
                         mlir::cf::ControlFlowDialect,
                         mlir::index::IndexDialect,
                         mlir::tensor::TensorDialect,
                         mlir::linalg::LinalgDialect,
                         mlir::memref::MemRefDialect,
                         mlir::LLVM::LLVMDialect>();
    return context;
#else
    return MlirContextPtr(nullptr);
#endif
}

std::vector<std::string> GetRequiredMlirDialects()
{
    if (!IsMlirAvailable()) {
        return {};
    }

    return {
        "func",
        "arith",
        "cf",
        "index",
        "tensor",
        "linalg",
        "memref",
        "llvm",
    };
}

void RegisterRequiredMlirDialects(mlir::DialectRegistry& registry)
{
#if TC_BACKEND_HAS_MLIR
    mlir::registerAllExtensions(registry);
    mlir::registerConvertToLLVMDependentDialectLoading(registry);
    registry.insert<mlir::func::FuncDialect,
                    mlir::arith::ArithDialect,
                    mlir::cf::ControlFlowDialect,
                    mlir::index::IndexDialect,
                    mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect,
                    mlir::LLVM::LLVMDialect>();
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
