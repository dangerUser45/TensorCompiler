#include <iostream>
#include <string>

#include "tc/backend/llvm_target_init.hpp"
#include "tc/backend/mlir_context.hpp"

int main()
{
    std::string error;
    if (!tc::backend::InitializeNativeLlvmBackend(error)) {
        std::cerr << error << '\n';
        return 1;
    }

    std::cout << "LLVM backend ready: "
              << tc::backend::GetLlvmVersionString() << '\n';
    std::cout << tc::backend::GetMlirSupportMessage() << '\n';

    return 0;
}
