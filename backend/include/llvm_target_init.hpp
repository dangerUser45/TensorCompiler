#pragma once

#include <string>

namespace tc::backend {

bool InitializeNativeLlvmBackend(std::string& out_error);
std::string GetLlvmVersionString();

} // namespace tc::backend
