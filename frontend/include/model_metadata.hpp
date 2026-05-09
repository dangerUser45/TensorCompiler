#pragma once

#include <string>

#include "graph.hpp"

namespace tc::frontend::metadata {

bool BuildMetadataJson(const Graph& graph,
                       std::string& out_json,
                       std::string& out_error);

} // namespace tc::frontend::metadata
