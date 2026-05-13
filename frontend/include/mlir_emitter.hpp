#pragma once

#include <string>

#include "graph.hpp"

namespace tc::frontend::mlir {

enum class EmitMode
{
    kStrict,  // returns false if graph is not EXEC-compatible
    kLenient, // always returns true; emits a stub for unsupported graphs
};

bool EmitMlirModule(const Graph& graph,
                    std::string& out_mlir_text,
                    std::string& out_error,
                    EmitMode mode = EmitMode::kStrict);

} // namespace tc::frontend::mlir
