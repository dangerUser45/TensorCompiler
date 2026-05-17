#pragma once

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "graph.hpp"

namespace tc::frontend {

class DumpGraph final
{
public:
    DumpGraph(std::ofstream& out);
    ~DumpGraph() = default;

    void dump(Graph& graph);

private:
    using ProducerMap = std::unordered_map<std::string, std::string>;

    std::ofstream& out_;

    void begin_graph(Graph& graph);
    void end_graph();

    void begin_table();
    void end_table();

    void print_io_header();
    void print_io_tensors(const Graph::TensVecT& tensors,
                          const std::string& title);
    void print_inits(const Graph::InitVecT& inits);
    void print_node_attrs(const Node& node);
    void print_op_nodes(const Graph::NodeVecT& nodes);

    void emit_rank_inputs(const Graph::TensVecT& inputs);
    void emit_rank_nodes(const Graph::NodeVecT& nodes,
                         const Graph::InitVecT& inits);
    void emit_rank_outputs(const Graph::TensVecT& outputs);
    void set_rank(const Graph::TensVecT& inputs,
                  const Graph::InitVecT& inits,
                  const Graph::NodeVecT& nodes,
                  const Graph::TensVecT& outputs);

    void emit_node_edges(size_t idx,
                         const Node& node,
                         ProducerMap& producers,
                         std::unordered_set<std::string>& printed);
    void emit_graph_output_edges(const Graph::TensVecT& outputs,
                                 const ProducerMap& producers,
                                 std::unordered_set<std::string>& printed);
    void set_links(const Graph::TensVecT& inputs,
                   const Graph::InitVecT& inits,
                   const Graph::NodeVecT& nodes,
                   const Graph::TensVecT& outputs);

    void print_graph(Graph& graph);
};
} // namespace tc::frontend
