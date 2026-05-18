#include "llvm_target_init.hpp"

#include <llvm/Config/llvm-config.h>
#include <llvm/Support/TargetSelect.h>

namespace tc::backend {

bool InitializeNativeLlvmBackend(std::string& out_error)
{
    if (llvm::InitializeNativeTarget()) {
        out_error = "ERROR: failed to initialize native LLVM target";
        return false;
    }

    if (llvm::InitializeNativeTargetAsmPrinter()) {
        out_error = "ERROR: failed to initialize native LLVM asm printer";
        return false;
    }

    if (llvm::InitializeNativeTargetAsmParser()) {
        out_error = "ERROR: failed to initialize native LLVM asm parser";
        return false;
    }

    out_error.clear();
    return true;
}

std::string GetLlvmVersionString()
{
    return LLVM_VERSION_STRING;
}

} // namespace tc::backend
