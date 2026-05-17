#pragma once

#include <string>

namespace tc::frontend {
class Graph;
} // namespace tc::frontend

namespace tc::frontend::metadata {

bool BuildMetadataJson(const Graph& graph,
                       std::string& out_json,
                       std::string& out_error);

} // namespace tc::frontend::metadata
