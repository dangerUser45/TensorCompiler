#include "dump_graph.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include "dump_style.hpp"

namespace tc::frontend {

namespace {

template<typename T>
std::string print_vector(const std::vector<T>& values)
{
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i != values.size(); ++i) {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, int32_t> ||
                      std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, uint8_t> ||
                      std::is_same_v<T, uint16_t> ||
                      std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, uint64_t>) {
            oss << static_cast<long long>(values[i]);
        } else {
            oss << values[i];
        }

        if (i + 1 != values.size())
            oss << ", ";
    }

    oss << "]";
    return oss.str();
}

template<typename T>
std::string value_to_string(const T& value)
{
    std::ostringstream oss;
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                  std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
                  std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                  std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
        oss << static_cast<long long>(value);
    } else {
        oss << value;
    }
    return oss.str();
}

template<typename T>
std::string print_vector_multiline_html(const std::vector<T>& values)
{
    if (values.empty())
        return "[]";

    std::ostringstream oss;
    for (size_t i = 0; i != values.size(); ++i) {
        oss << value_to_string(values[i]);
        if (i + 1 != values.size())
            oss << "<BR ALIGN=\"LEFT\"/>";
    }
    return oss.str();
}

template<typename ValSource, typename Fn>
std::string dispatch_typed(const ValSource& src, Fn fn, std::string fallback)
{
    try {
        switch (src.get_data_type().id) {
            case DataID::INT8:
                return fn(src.template get_values<int8_t>());
            case DataID::INT16:
                return fn(src.template get_values<int16_t>());
            case DataID::INT32:
                return fn(src.template get_values<int32_t>());
            case DataID::INT64:
                return fn(src.template get_values<int64_t>());
            case DataID::UNSIGNED_INT8:
                return fn(src.template get_values<uint8_t>());
            case DataID::UNSIGNED_INT16:
                return fn(src.template get_values<uint16_t>());
            case DataID::UNSIGNED_INT32:
                return fn(src.template get_values<uint32_t>());
            case DataID::UNSIGNED_INT64:
                return fn(src.template get_values<uint64_t>());
            case DataID::FLOAT:
                return fn(src.template get_values<float>());
            case DataID::DOUBLE:
                return fn(src.template get_values<double>());
            case DataID::STRING:
                return fn(src.template get_values<std::string>());
            default:
                return fallback;
        }
    } catch (...) {
        return fallback;
    }
}

std::string normalize_name(std::string name)
{
    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
            c = '_';
        }
    }
    return name;
}

std::string print_shape(const std::vector<int64_t>& shape)
{
    return print_vector(shape);
}

std::string print_names_multiline_html(const std::vector<std::string>& names)
{
    if (names.empty())
        return "(none)";

    std::ostringstream oss;
    for (size_t i = 0; i != names.size(); ++i) {
        oss << names[i];
        if (i + 1 != names.size())
            oss << "<BR ALIGN=\"LEFT\"/>";
    }
    return oss.str();
}

std::string to_upper(std::string value)
{
    for (char& c : value) {
        if (c >= 'a' && c <= 'z')
            c = static_cast<char>(c - ('a' - 'A'));
    }
    return value;
}

std::string tensor_id(const TensorInfo& tensor,
                      const std::string& prefix,
                      size_t index)
{
    if (!tensor.get_name().empty())
        return normalize_name(tensor.get_name());
    return normalize_name(prefix) + "_" + std::to_string(index);
}

std::string node_id(const Node& node, size_t index)
{
    if (!node.get_name_node().empty())
        return normalize_name(node.get_name_node());
    return "op_" + std::to_string(index);
}

int64_t element_count(const std::vector<int64_t>& shape)
{
    if (shape.empty())
        return 0;

    int64_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0)
            return -1;
        if (count > std::numeric_limits<int64_t>::max() / dim)
            return -1;
        count *= dim;
    }
    return count;
}

std::string init_values_as_multiline_html(const Initializers& init)
{
    return dispatch_typed(
        init,
        [](const auto& v) { return print_vector_multiline_html(v); },
        "(unknown)");
}

std::string attr_values_as_string(const Attribute& attr)
{
    return dispatch_typed(
        attr,
        [](const auto& v) { return print_vector(v); },
        "(" + std::string(DataIDToString(attr.get_data_type().id)) + ")");
}

int edge_minlen_from_label(const std::string& label)
{
    constexpr size_t kCharsPerRank = 12;
    constexpr int kMaxMinLen = 12;
    const int minlen = 1 + static_cast<int>(label.size() / kCharsPerRank);
    return std::min(minlen, kMaxMinLen);
}

} // anonymous namespace

DumpGraph::DumpGraph(std::ofstream& out)
  : out_(out)
{
}

void DumpGraph::dump(Graph& graph)
{
    begin_graph(graph);
    print_graph(graph);
    end_graph();
}

void DumpGraph::begin_graph(Graph& graph)
{
    out_ << "digraph " << normalize_name(graph.get_name()) << " {\n\t";
    out_ << "rankdir=TB\n\t"
            "splines=ortho;\n\t"
            "nodesep="
         << NODESEP
         << ";\n\t"
            "ranksep="
         << RANKSEP
         << ";\n\t"
            "bgcolor=\""
         << BACKGROUND_COLOR
         << "\";\n\n\t"

            "labelloc=\"t\";\n\t"
            "label=\""
         << graph.get_name()
         << "\";\n\t"
            "fontname=\""
         << FONT_NAME
         << "\";\n\n\t"
            "fontsize="
         << TITLE_FONT_SIZE
         << ";\n\n\t"

            "node [fontname=\""
         << FONT_NAME
         << "\", shape=plaintext, style=\"rounded,filled\"];\n\t"
            "edge [fontname=\""
         << FONT_NAME << "\", color=\"" << EDGE_COLOR
         << "\", penwidth=" << PEN_WIDTH << ", arrowsize=" << ARROW_SIZE
         << "];\n\n\t";
}

void DumpGraph::end_graph()
{
    out_ << "}\n";
}

void DumpGraph::begin_table()
{
    out_ << "style=\"rounded\",\n\t\t"
            "label=<\n\t\t\t"
            "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
            "CELLPADDING=\"6\">\n\t\t\t\t";
}

void DumpGraph::end_table()
{
    out_ << "</TABLE>\n\t\t>\n\t];\n\n\t";
}

void DumpGraph::print_io_header()
{
    out_ << "// ===== INPUTS/OUTPUTS =====\n\t";
}

void DumpGraph::print_io_tensors(const Graph::TensVecT& tensors,
                                 const std::string& title)
{
    for (size_t i = 0; i != tensors.size(); ++i) {
        if (!tensors[i])
            continue;

        const std::string id = tensor_id(*tensors[i], title, i);
        out_ << id << " [\n\t\t";
        begin_table();
        out_ << "<TR><TD BGCOLOR=\"" << IO_HEADER_COLOR << "\"><B>"
             << to_upper(title)
             << "</B></TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << IO_BASE_COLOR
             << "\"><B>name</B>: " << normalize_name(tensors[i]->get_name())
             << "</TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << IO_BASE_COLOR << "\"><B>dtype</B>: "
             << DataIDToString(tensors[i]->get_data_type().id)
             << "</TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << IO_BASE_COLOR
             << "\"><B>shape</B>: " << print_shape(tensors[i]->get_shape())
             << "</TD></TR>\n\t\t\t";
        end_table();
    }
}

void DumpGraph::print_inits(const Graph::InitVecT& inits)
{
    out_ << "// ===== INITIALIZERS =====\n\t";

    for (size_t i = 0; i != inits.size(); ++i) {
        if (!inits[i])
            continue;

        const std::string id = tensor_id(*inits[i], "init", i);
        out_ << id << " [\n\t\t";
        begin_table();
        out_ << "<TR><TD BGCOLOR=\"" << INITS_HEADER_COLOR
             << "\"><B>INITIALIZER</B></TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << INITS_BASE_COLOR << "\"><B>name</B>: " << inits[i]->get_name()
             << "</TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << INITS_BASE_COLOR << "\"><B>dtype</B>: "
             << DataIDToString(inits[i]->get_data_type().id)
             << "</TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << INITS_BASE_COLOR
             << "\"><B>shape</B>: " << print_shape(inits[i]->get_shape())
             << "</TD></TR>\n\t\t\t\t";

        const int64_t n_elements = element_count(inits[i]->get_shape());
        if (n_elements > 16) {
            out_ << "<TR><TD BGCOLOR=\"" << INITS_BASE_COLOR
                 << "\"><B>storage</B>: raw_data</TD></TR>\n\t\t\t";
        } else {
            out_ << "<TR><TD BGCOLOR=\"" << INITS_BASE_COLOR
                 << "\" ALIGN=\"LEFT\"><B>values</B>:<BR ALIGN=\"LEFT\"/>"
                 << init_values_as_multiline_html(*inits[i])
                 << "</TD></TR>\n\t\t\t";
        }
        end_table();
    }
}

void DumpGraph::print_node_attrs(const Node& node)
{
    const auto& attrs = node.get_attrs();
    if (attrs.empty()) {
        out_ << "<TR><TD BGCOLOR=\"" << OPERAT_BASE_COLOR
             << "\"><B>attributes</B>: (none)</TD></TR>\n\t\t\t";
        return;
    }

    out_ << "<TR><TD BGCOLOR=\"" << OPERAT_BASE_COLOR
         << "\"><TABLE BORDER=\"0\" CELLBORDER=\"1\" "
            "CELLSPACING=\"0\" CELLPADDING=\"5\">\n\t\t\t\t\t"
            "<TR><TD BGCOLOR=\""
         << ATTR_HEADER_COLOR << "\"><B>ATTRIBUTES</B></TD></TR>\n\t\t\t\t\t";

    for (size_t j = 0; j != attrs.size(); ++j) {
        if (!attrs[j])
            continue;
        out_ << "<TR><TD BGCOLOR=\"" << ATTR_BASE_COLOR << "\"><B>"
             << attrs[j]->get_name()
             << "</B>: " << attr_values_as_string(*attrs[j])
             << "</TD></TR>\n\t\t\t\t";
        if (j != attrs.size() - 1)
            out_ << "\t";
    }

    out_ << "</TABLE></TD></TR>\n\t\t\t";
}

void DumpGraph::print_op_nodes(const Graph::NodeVecT& nodes)
{
    out_ << "// ===== NODES =====\n\t";

    for (size_t i = 0; i != nodes.size(); ++i) {
        if (!nodes[i])
            continue;

        const std::string id = node_id(*nodes[i], i);
        out_ << id << " [\n\t\t";
        begin_table();
        out_ << "<TR><TD BGCOLOR=\"" << OPERAT_HEADER_COLOR
             << "\"><B>OPERATION</B></TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << OPERAT_BASE_COLOR << "\"><B>name</B>: " << id
             << "</TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << OPERAT_BASE_COLOR
             << "\" ALIGN=\"LEFT\"><B>inputs</B>:<BR ALIGN=\"LEFT\"/>"
             << print_names_multiline_html(nodes[i]->get_inputs())
             << "</TD></TR>\n\t\t\t\t"
                "<TR><TD BGCOLOR=\""
             << OPERAT_BASE_COLOR
             << "\" ALIGN=\"LEFT\"><B>outputs</B>:<BR ALIGN=\"LEFT\"/>"
             << print_names_multiline_html(nodes[i]->get_outputs())
             << "</TD></TR>\n\t\t\t\t";

        print_node_attrs(*nodes[i]);
        end_table();
    }
}

void DumpGraph::emit_rank_inputs(const Graph::TensVecT& inputs)
{
    out_ << "{ rank=same; ";
    for (size_t i = 0; i != inputs.size(); ++i) {
        if (!inputs[i])
            continue;
        out_ << tensor_id(*inputs[i], "input", i) << "; ";
    }
    out_ << "}\n\t";
}

void DumpGraph::emit_rank_nodes(const Graph::NodeVecT& nodes,
                                const Graph::InitVecT& inits)
{
    std::unordered_map<std::string, std::string> init_id_map;
    for (size_t i = 0; i != inits.size(); ++i) {
        if (!inits[i])
            continue;
        init_id_map[normalize_name(inits[i]->get_name())] =
            tensor_id(*inits[i], "init", i);
    }

    std::unordered_set<std::string> ranked_inits;
    for (size_t i = 0; i != nodes.size(); ++i) {
        if (!nodes[i])
            continue;

        out_ << "{ rank=same; ";
        for (const std::string& raw : nodes[i]->get_inputs()) {
            const auto it = init_id_map.find(normalize_name(raw));
            if (it == init_id_map.end())
                continue;
            if (ranked_inits.insert(it->second).second)
                out_ << it->second << "; ";
        }
        out_ << node_id(*nodes[i], i) << "; }\n\t";
    }
}

void DumpGraph::emit_rank_outputs(const Graph::TensVecT& outputs)
{
    out_ << "{ rank=same; ";
    for (size_t i = 0; i != outputs.size(); ++i) {
        if (!outputs[i])
            continue;
        out_ << tensor_id(*outputs[i], "output", i) << "; ";
    }
    out_ << "}\n\n\t";
}

void DumpGraph::set_rank(const Graph::TensVecT& inputs,
                         const Graph::InitVecT& inits,
                         const Graph::NodeVecT& nodes,
                         const Graph::TensVecT& outputs)
{
    out_ << "// ===== LAYERED ALIGNMENT (rank=same) =====\n\t";
    emit_rank_inputs(inputs);
    emit_rank_nodes(nodes, inits);
    emit_rank_outputs(outputs);
}

void DumpGraph::emit_node_edges(size_t idx,
                                const Node& node,
                                ProducerMap& producers,
                                std::unordered_set<std::string>& printed)
{
    const std::string nid = node_id(node, idx);

    for (const std::string& raw : node.get_inputs()) {
        const std::string name = normalize_name(raw);
        const auto prod_it = producers.find(name);
        if (prod_it == producers.end())
            continue;

        const std::string edge_key = prod_it->second + "->" + nid + ":" + name;
        if (!printed.insert(edge_key).second)
            continue;

        out_ << prod_it->second << " -> " << nid << " [xlabel=\"" << name
             << "\", minlen=" << edge_minlen_from_label(name) << "];\n\t";
    }

    for (const std::string& raw : node.get_outputs())
        producers[normalize_name(raw)] = nid;
}

void DumpGraph::emit_graph_output_edges(
    const Graph::TensVecT& outputs,
    const ProducerMap& producers,
    std::unordered_set<std::string>& printed)
{
    for (size_t i = 0; i != outputs.size(); ++i) {
        if (!outputs[i])
            continue;

        const std::string out_name = normalize_name(outputs[i]->get_name());
        const auto prod_it = producers.find(out_name);
        if (prod_it == producers.end())
            continue;

        const std::string out_id = tensor_id(*outputs[i], "output", i);
        const std::string edge_key =
            prod_it->second + "->" + out_id + ":" + out_name;
        if (!printed.insert(edge_key).second)
            continue;

        out_ << prod_it->second << " -> " << out_id << " [xlabel=\"" << out_name
             << "\", minlen=" << edge_minlen_from_label(out_name) << "];\n";
        if (i != outputs.size() - 1)
            out_ << "\t";
    }
}

void DumpGraph::set_links(const Graph::TensVecT& inputs,
                          const Graph::InitVecT& inits,
                          const Graph::NodeVecT& nodes,
                          const Graph::TensVecT& outputs)
{
    ProducerMap producers;
    for (size_t i = 0; i != inputs.size(); ++i) {
        if (!inputs[i])
            continue;
        producers[normalize_name(inputs[i]->get_name())] =
            tensor_id(*inputs[i], "input", i);
    }
    for (size_t i = 0; i != inits.size(); ++i) {
        if (!inits[i])
            continue;
        producers[normalize_name(inits[i]->get_name())] =
            tensor_id(*inits[i], "init", i);
    }

    std::unordered_set<std::string> printed;
    out_ << "// ===== DATA FLOW (orthogonal edges) =====\n\t";
    for (size_t i = 0; i != nodes.size(); ++i) {
        if (!nodes[i])
            continue;
        emit_node_edges(i, *nodes[i], producers, printed);
    }
    emit_graph_output_edges(outputs, producers, printed);
}

void DumpGraph::print_graph(Graph& graph)
{
    print_io_header();
    print_io_tensors(graph.get_input_tensors(), "input");
    print_io_tensors(graph.get_output_tensors(), "output");
    print_inits(graph.get_inits());
    print_op_nodes(graph.get_nodes());
    set_rank(graph.get_input_tensors(),
             graph.get_inits(),
             graph.get_nodes(),
             graph.get_output_tensors());
    set_links(graph.get_input_tensors(),
              graph.get_inits(),
              graph.get_nodes(),
              graph.get_output_tensors());
}

} // namespace tc::frontend
