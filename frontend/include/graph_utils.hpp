#pragma once

#include "type_info.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace tc::frontend {

class Attribute;
class Node;

bool IsSupportedNumericDtype(DataID id) noexcept;

std::string BuildNodeContext(std::string_view node_name,
                             std::size_t node_index);

std::string BuildNodeContext(const Node& node, std::size_t node_index);

const Attribute* FindAttr(const Node& node, std::string_view name);

bool EndsWith(std::string_view text, std::string_view suffix) noexcept;

bool IsSyntheticBiasAdd(const Node& node) noexcept;

} // namespace tc::frontend
