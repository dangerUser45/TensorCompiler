#pragma once

#include <optional>
#include <string>

namespace tc::frontend::driver {

struct BackendRequest
{
    std::optional<std::string> llvm_path;
    std::optional<std::string> asm_path;
    std::optional<std::string> object_path;
    std::optional<std::string> exe_path;
    std::string target_triple;
    std::string pass_pipeline;
};

// Runs the codegen pipeline. Returns 0 on success, 1 on any backend failure.
// `metadata_json` and `metadata_path` are only consulted when
// `request.exe_path` is set (linker needs metadata sidecar). Prints diagnostics
// to stderr.
int RunBackendPipeline(const BackendRequest& request,
                       const std::string& mlir_text,
                       const std::string& input_path,
                       const std::string& metadata_json,
                       const std::string& metadata_path);

} // namespace tc::frontend::driver
