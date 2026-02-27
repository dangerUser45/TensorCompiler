#include <functional>

#include "graph.hpp"
#include "graph_hash.hpp"

namespace tc::frontend {
namespace {
void HashCombine(std::size_t& seed, const std::size_t& value);

void HashNameGraph(std::size_t& seed, const std::string& name);

template<class VecT>
void HashTensors(std::size_t& seed, const VecT& vec);

void HashNodes(std::size_t& seed, const Graph::NodeVecT& node_vec);
} // namespace

std::size_t HashGraph(const Graph& graph)
{
    std::size_t seed = 0;

    HashNameGraph(seed, graph.get_name());

    HashTensors(seed, graph.get_inits());
    HashTensors(seed, graph.get_input_tensors());
    HashTensors(seed, graph.get_output_tensors());

    HashNodes(seed, graph.get_nodes());

    return seed;
}

namespace {

void HashCombine(std::size_t& seed, const std::size_t& value)
{
    // Boost method
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

void HashNameGraph(std::size_t& seed, const std::string& name)
{
    std::hash<std::string> Hasher;
    const std::size_t val = Hasher(name);
    HashCombine(seed, val);
}

template<class VecT>
void HashTensors(std::size_t& seed, const VecT& vec)
{
    for (auto&& elem : vec) {
        if (!elem)
            continue;

        const std::string& name = elem->get_name();
        std::hash<std::string> string_hasher;
        const std::size_t name_hash = string_hasher(name);
        HashCombine(seed, name_hash);

        const std::string data_type_str = elem->get_data_type().data_type_str;
        const std::size_t dtype_hash = string_hasher(data_type_str);
        HashCombine(seed, dtype_hash);

        const std::vector<int64_t>& shape = elem->get_shape();
        std::hash<int64_t> data_hasher;
        for (auto&& data_elem : shape) {
            const std::size_t dim_hash = data_hasher(data_elem);
            HashCombine(seed, dim_hash);
        }
    }
}

void HashNodes(std::size_t& seed, const tc::frontend::Graph::NodeVecT& node_vec)
{
    for (auto&& elem : node_vec) {
        if (!elem)
            continue;

        std::hash<std::string> string_hasher;

        const std::string& name_node = elem->get_name_node();
        const std::string& name_op = elem->get_name_op();

        const std::size_t name_node_hash = string_hasher(name_node);
        const std::size_t name_op_hash = string_hasher(name_op);

        HashCombine(seed, name_node_hash);
        HashCombine(seed, name_op_hash);

        for (auto&& input : elem->get_inputs()) {
            const std::size_t input_elem_hash = string_hasher(input);
            HashCombine(seed, input_elem_hash);
        }

        for (auto&& output : elem->get_outputs()) {
            const std::size_t output_elem_hash = string_hasher(output);
            HashCombine(seed, output_elem_hash);
        }

        for (auto&& attr : elem->get_attrs()) {
            const std::string& name = attr->get_name();
            const std::size_t name_hash = string_hasher(name);
            HashCombine(seed, name_hash);

            const std::string& data_type_str =
                attr->get_data_type().data_type_str;
            const std::size_t dtype_hash = string_hasher(data_type_str);
            HashCombine(seed, dtype_hash);
        }
    }
}

} // namespace
} // namespace tc::frontend
