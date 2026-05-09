#pragma once

#include <string>

#include "graph.hpp"

namespace tc::frontend::mlir {

bool EmitMlirModule(const Graph& graph,
                    std::string& out_mlir_text,
                    std::string& out_error);

bool EmitMlirModuleSkeleton(const Graph& graph,
                            std::string& out_mlir_text,
                            std::string& out_error);

} // namespace tc::frontend::mlir
