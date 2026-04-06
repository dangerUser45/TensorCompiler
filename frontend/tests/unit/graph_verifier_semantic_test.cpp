#include "graph_verifier.hpp"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

bool HasDiagnosticContaining(const tc::frontend::verify::Report& report,
                             const std::string& fragment)
{
    for (const auto& diagnostic : report.diagnostics()) {
        if (diagnostic.message.find(fragment) != std::string::npos) {
            return true;
        }
    }
    return false;
}

tc::frontend::Graph MakeSingleNodeGraph(tc::frontend::OpKind op_kind,
                                        std::string op_name,
                                        std::vector<std::string> inputs,
                                        std::vector<std::string> outputs)
{
    tc::frontend::Graph graph;
    graph.set_name("semantic_graph");

    tc::frontend::Graph::NodeVecT nodes;
    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("n0");
    node->set_name_op(std::move(op_name));
    node->set_op_kind(op_kind);
    node->set_inputs(std::move(inputs));
    node->set_outputs(outputs);
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    tc::frontend::Graph::TensVecT output_tensors;
    output_tensors.reserve(outputs.size());
    for (const auto& output_name : outputs) {
        auto tensor = std::make_unique<tc::frontend::TensorInfo>();
        tensor->set_name(output_name);
        tensor->set_shape({ 1 });
        output_tensors.push_back(std::move(tensor));
    }
    graph.set_output_tensors(std::move(output_tensors));

    return graph;
}

TEST(GraphVerifierSemantic, ValidReluGraphPasses)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kRelu, "Relu", { "x" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << "expected valid relu graph to pass semantic checks";
}

TEST(GraphVerifierSemantic, ReluWithTwoInputsFails)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kRelu, "Relu", { "x", "b" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "expects 1 input"))
        << "missing relu arity diagnostic";
}

TEST(GraphVerifierSemantic, UnknownOpKindFails)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kUnknown, "Relu", { "x" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "unknown op kind"))
        << "missing unknown op kind diagnostic";
}

} // namespace
