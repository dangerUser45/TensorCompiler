#include "graph_utils.hpp"

#include "frontend_constants.hpp"
#include "graph.hpp"

#include <string>

namespace tc::frontend {

bool IsSupportedNumericDtype(DataID id) noexcept
{
    switch (id) {
        case DataID::INT8:
        case DataID::INT16:
        case DataID::INT32:
        case DataID::INT64:
        case DataID::UNSIGNED_INT8:
        case DataID::UNSIGNED_INT16:
        case DataID::UNSIGNED_INT32:
        case DataID::UNSIGNED_INT64:
        case DataID::FLOAT:
        case DataID::DOUBLE:
            return true;
        case DataID::COMPLEX64:
        case DataID::COMPLEX128:
        case DataID::STRING:
        case DataID::UNDEFINED:
            return false;
    }
    return false;
}

std::string BuildNodeContext(std::string_view node_name, std::size_t node_index)
{
    const std::string prefix = "node[" + std::to_string(node_index) + "]";
    if (!node_name.empty()) {
        return prefix + "('" + std::string(node_name) + "')";
    }
    return prefix;
}

std::string BuildNodeContext(const Node& node, std::size_t node_index)
{
    return BuildNodeContext(node.get_name_node(), node_index);
}

const Attribute* FindAttr(const Node& node, std::string_view name)
{
    for (const auto& attr : node.get_attrs()) {
        if (attr && attr->get_name() == name) {
            return attr.get();
        }
    }
    return nullptr;
}

bool EndsWith(std::string_view text, std::string_view suffix) noexcept
{
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) ==
               0;
}

bool IsSyntheticBiasAdd(const Node& node) noexcept
{
    return node.get_op_kind() == OpKind::kAdd &&
           EndsWith(node.get_name_node(), kSyntheticAddSuffix);
}

} // namespace tc::frontend
