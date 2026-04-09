#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "graph.hpp"
#include "mlir_emitter.hpp"
#include "op_kind.hpp"

namespace {

tc::frontend::Graph MakeSingleReluGraph()
{
    tc::frontend::Graph graph;
    graph.set_name("demo_graph");

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("relu_0");
    node->set_name_op("Relu");
    node->set_op_kind(tc::frontend::OpKind::kRelu);
    node->set_inputs({ "x" });
    node->set_outputs({ "y" });

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));
    return graph;
}

} // namespace

TEST(MlirEmitter, EmitsDeterministicSkeletonForSingleNodeGraph)
{
    const auto graph = MakeSingleReluGraph();
    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(
        tc::frontend::mlir::EmitMlirModuleSkeleton(graph, mlir_text, error))
        << error;
    EXPECT_TRUE(error.empty());

    const std::string expected =
        "module {\n"
        "  // tc.graph: demo_graph\n"
        "  // tc.node_count: 1\n"
        "  func.func @main() {\n"
        "    // TODO(tc): Graph->MLIR lowering is not implemented yet.\n"
        "    // node[0]: op=Relu, name=relu_0\n"
        "    return\n"
        "  }\n"
        "}\n";
    EXPECT_EQ(mlir_text, expected);
}

TEST(MlirEmitter, UsesFallbackGraphNameForUnnamedGraph)
{
    tc::frontend::Graph graph;
    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(
        tc::frontend::mlir::EmitMlirModuleSkeleton(graph, mlir_text, error))
        << error;
    EXPECT_TRUE(error.empty());

    const std::string expected =
        "module {\n"
        "  // tc.graph: unnamed_graph\n"
        "  // tc.node_count: 0\n"
        "  func.func @main() {\n"
        "    // TODO(tc): Graph->MLIR lowering is not implemented yet.\n"
        "    return\n"
        "  }\n"
        "}\n";
    EXPECT_EQ(mlir_text, expected);
}
