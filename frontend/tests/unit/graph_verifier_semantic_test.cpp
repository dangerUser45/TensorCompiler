#include "graph_verifier.hpp"

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
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

std::string DiagnosticsAsString(const tc::frontend::verify::Report& report)
{
    std::ostringstream oss;
    for (const auto& diagnostic : report.diagnostics()) {
        oss << diagnostic.message << '\n';
    }
    return oss.str();
}

tc::frontend::Graph MakeSingleNodeGraph(
    tc::frontend::OpKind op_kind,
    std::string op_name,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    tc::frontend::Node::AttrVecT attrs = {},
    const std::unordered_map<std::string, tc::frontend::DataT>&
        tensor_types = {})
{
    tc::frontend::Graph graph;
    graph.set_name("semantic_graph");

    tc::frontend::Graph::NodeVecT nodes;
    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("n0");
    node->set_name_op(std::move(op_name));
    node->set_op_kind(op_kind);
    node->set_inputs(inputs);
    node->set_outputs(outputs);
    node->set_attr(std::move(attrs));
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    std::set<std::string> input_names(inputs.begin(), inputs.end());
    tc::frontend::Graph::TensVecT input_tensors;
    input_tensors.reserve(input_names.size());
    for (const auto& input_name : input_names) {
        auto tensor = std::make_unique<tc::frontend::TensorInfo>();
        tensor->set_name(input_name);
        tensor->set_shape({ 1 });
        const auto it = tensor_types.find(input_name);
        tensor->set_data_type(it == tensor_types.end()
                                  ? tc::frontend::TypeInfo<float>::type
                                  : it->second);
        input_tensors.push_back(std::move(tensor));
    }
    graph.set_input_tensors(std::move(input_tensors));

    tc::frontend::Graph::TensVecT output_tensors;
    output_tensors.reserve(outputs.size());
    for (const auto& output_name : outputs) {
        auto tensor = std::make_unique<tc::frontend::TensorInfo>();
        tensor->set_name(output_name);
        tensor->set_shape({ 1 });
        const auto it = tensor_types.find(output_name);
        tensor->set_data_type(it == tensor_types.end()
                                  ? tc::frontend::TypeInfo<float>::type
                                  : it->second);
        output_tensors.push_back(std::move(tensor));
    }
    graph.set_output_tensors(std::move(output_tensors));

    return graph;
}

std::unique_ptr<tc::frontend::Attribute> MakeIntsAttr(
    std::string name,
    std::vector<int64_t> values)
{
    auto attr = std::make_unique<tc::frontend::Attribute>();
    attr->set_name(std::move(name));
    attr->set_values<int64_t>(std::move(values));
    return attr;
}

std::unique_ptr<tc::frontend::Attribute> MakeFloatsAttr(
    std::string name,
    std::vector<float> values)
{
    auto attr = std::make_unique<tc::frontend::Attribute>();
    attr->set_name(std::move(name));
    attr->set_values<float>(std::move(values));
    return attr;
}

TEST(GraphVerifierSemantic, ValidReluGraphPasses)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kRelu, "Relu", { "x" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << "expected valid relu graph to pass semantic checks\n"
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ReluWithTwoInputsFails)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kRelu, "Relu", { "x", "b" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(HasDiagnosticContaining(report, "expects 1 input"))
        << "missing relu arity diagnostic";
}

TEST(GraphVerifierSemantic, UnknownOpKindFails)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kUnknown, "Relu", { "x" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(HasDiagnosticContaining(report, "unknown op kind"))
        << "missing unknown op kind diagnostic";
}

TEST(GraphVerifierSemantic, ReluWithUnexpectedAttributeFails)
{
    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(MakeFloatsAttr("alpha", { 1.0f }));

    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                                     "Relu",
                                     { "x" },
                                     { "y" },
                                     std::move(attrs));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(HasDiagnosticContaining(report, "does not support attributes"))
        << "missing unexpected attribute diagnostic";
}

TEST(GraphVerifierSemantic, TransposeWithNegativePermFails)
{
    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(MakeIntsAttr("perm", { 0, -1 }));

    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kTranspose,
                                     "Transpose",
                                     { "x" },
                                     { "y" },
                                     std::move(attrs));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "contains negative axis"))
        << "missing negative perm diagnostic";
}

TEST(GraphVerifierSemantic, TransposeWithDuplicatePermFails)
{
    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(MakeIntsAttr("perm", { 1, 1 }));

    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kTranspose,
                                     "Transpose",
                                     { "x" },
                                     { "y" },
                                     std::move(attrs));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "has duplicate axis"))
        << "missing duplicate perm diagnostic";
}

TEST(GraphVerifierSemantic, AddWithMismatchedInputDtypesFails)
{
    auto graph =
        MakeSingleNodeGraph(tc::frontend::OpKind::kAdd,
                            "Add",
                            { "x", "b" },
                            { "y" },
                            {},
                            {
                                { "x", tc::frontend::TypeInfo<float>::type },
                                { "b", tc::frontend::TypeInfo<int32_t>::type },
                                { "y", tc::frontend::TypeInfo<float>::type },
                            });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "input dtypes mismatch"))
        << "missing add dtype mismatch diagnostic";
}

TEST(GraphVerifierSemantic, ReluWithUnsupportedStringInputDtypeFails)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kRelu,
        "Relu",
        { "x" },
        { "y" },
        {},
        {
            { "x", tc::frontend::TypeInfo<std::string>::type },
            { "y", tc::frontend::TypeInfo<std::string>::type },
        });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "expects numeric dtype"))
        << "missing relu numeric dtype diagnostic";
}

TEST(GraphVerifierSemantic, TransposeWithOutputDtypeMismatchFails)
{
    auto graph =
        MakeSingleNodeGraph(tc::frontend::OpKind::kTranspose,
                            "Transpose",
                            { "x" },
                            { "y" },
                            {},
                            {
                                { "x", tc::frontend::TypeInfo<float>::type },
                                { "y", tc::frontend::TypeInfo<int32_t>::type },
                            });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "output dtype mismatch"))
        << "missing transpose output dtype mismatch diagnostic";
}

} // namespace
