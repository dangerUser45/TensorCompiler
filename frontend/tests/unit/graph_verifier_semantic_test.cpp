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
        tensor_types = {},
    const std::unordered_map<std::string, std::vector<int64_t>>&
        tensor_shapes = {})
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
        const auto shape_it = tensor_shapes.find(input_name);
        tensor->set_shape(shape_it == tensor_shapes.end()
                              ? std::vector<int64_t>{ 1 }
                              : shape_it->second);
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
        const auto shape_it = tensor_shapes.find(output_name);
        tensor->set_shape(shape_it == tensor_shapes.end()
                              ? std::vector<int64_t>{ 1 }
                              : shape_it->second);
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

std::unique_ptr<tc::frontend::Attribute> MakeStringAttr(std::string name,
                                                        std::string value)
{
    auto attr = std::make_unique<tc::frontend::Attribute>();
    attr->set_name(std::move(name));
    attr->set_values<std::string>({ std::move(value) });
    return attr;
}

std::unique_ptr<tc::frontend::Initializers> MakeInitializer(
    std::string name,
    std::vector<int64_t> shape,
    tc::frontend::DataT dtype)
{
    auto init = std::make_unique<tc::frontend::Initializers>();
    init->set_name(std::move(name));
    init->set_shape(std::move(shape));
    init->set_data_type(dtype);
    return init;
}

tc::frontend::Node::AttrVecT MakeConvAttrs(
    std::vector<int64_t> kernel_shape = { 2, 2 },
    std::vector<int64_t> strides = { 1, 1 },
    std::vector<int64_t> pads = { 0, 0, 0, 0 },
    std::vector<int64_t> dilations = { 1, 1 },
    int64_t group = 1,
    std::string auto_pad = "NOTSET")
{
    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(MakeIntsAttr("kernel_shape", std::move(kernel_shape)));
    attrs.push_back(MakeIntsAttr("strides", std::move(strides)));
    attrs.push_back(MakeIntsAttr("pads", std::move(pads)));
    attrs.push_back(MakeIntsAttr("dilations", std::move(dilations)));
    attrs.push_back(MakeIntsAttr("group", { group }));
    attrs.push_back(MakeStringAttr("auto_pad", std::move(auto_pad)));
    return attrs;
}

tc::frontend::Graph MakeConvGraph(
    tc::frontend::Node::AttrVecT attrs,
    std::unordered_map<std::string, tc::frontend::DataT> tensor_types = {},
    std::unordered_map<std::string, std::vector<int64_t>> tensor_shapes = {})
{
    if (tensor_shapes.empty()) {
        tensor_shapes = {
            { "x", { 1, 2, 8, 8 } },
            { "w", { 2, 2, 2, 2 } },
            { "y", { 1, 2, 7, 7 } },
        };
    }
    if (tensor_types.empty()) {
        tensor_types = {
            { "x", tc::frontend::TypeInfo<float>::type },
            { "w", tc::frontend::TypeInfo<float>::type },
            { "y", tc::frontend::TypeInfo<float>::type },
        };
    }

    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kConv,
                                     "Conv",
                                     { "x", "w" },
                                     { "y" },
                                     std::move(attrs),
                                     tensor_types,
                                     tensor_shapes);

    tc::frontend::Graph::InitVecT inits;
    auto weight =
        MakeInitializer("w", tensor_shapes.at("w"), tensor_types.at("w"));
    if (tensor_types.at("w").id == tc::frontend::DataID::FLOAT) {
        weight->set_values<float>(std::vector<float>(16, 1.0f));
    } else if (tensor_types.at("w").id == tc::frontend::DataID::INT32) {
        weight->set_values<int32_t>(std::vector<int32_t>(16, 1));
    }
    inits.push_back(std::move(weight));
    graph.set_inits(std::move(inits));
    return graph;
}

tc::frontend::Graph MakeReshapeGraph(bool include_shape_as_runtime_input)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kReshape,
        "Reshape",
        { "x", "shape" },
        { "y" },
        {},
        {
            { "x", tc::frontend::TypeInfo<float>::type },
            { "shape", tc::frontend::TypeInfo<int64_t>::type },
            { "y", tc::frontend::TypeInfo<float>::type },
        },
        {
            { "x", { 1, 2, 2 } },
            { "shape", { 2 } },
            { "y", { 1, 4 } },
        });

    tc::frontend::Graph::InitVecT inits;
    auto shape_init =
        MakeInitializer("shape", { 2 }, tc::frontend::TypeInfo<int64_t>::type);
    shape_init->set_values<int64_t>({ 1, 4 });
    inits.push_back(std::move(shape_init));
    graph.set_inits(std::move(inits));

    if (!include_shape_as_runtime_input) {
        tc::frontend::Graph::TensVecT input_tensors;
        auto x = std::make_unique<tc::frontend::TensorInfo>();
        x->set_name("x");
        x->set_shape({ 1, 2, 2 });
        x->set_data_type(tc::frontend::TypeInfo<float>::type);
        input_tensors.push_back(std::move(x));
        graph.set_input_tensors(std::move(input_tensors));
    }

    return graph;
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

TEST(GraphVerifierSemantic, ExecutableVerifierRejectsMultipleRuntimeInputs)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kAdd, "Add", { "x", "b" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(
        HasDiagnosticContaining(report, "expects exactly 1 runtime input"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ExecutableVerifierAcceptsSingleStaticFloat32Relu)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                                     "Relu",
                                     { "x" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 1, 2 } },
                                         { "y", { 1, 2 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ExecutableVerifierRejectsNonFloat32RuntimeInput)
{
    auto graph =
        MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                            "Relu",
                            { "x" },
                            { "y" },
                            {},
                            {
                                { "x", tc::frontend::TypeInfo<int32_t>::type },
                                { "y", tc::frontend::TypeInfo<int32_t>::type },
                            },
                            {
                                { "x", { 1, 2 } },
                                { "y", { 1, 2 } },
                            });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(
        HasDiagnosticContaining(report, "runtime input 'x' must be float32"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ExecutableVerifierRejectsDynamicRuntimeInputShape)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                                     "Relu",
                                     { "x" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 1, -1 } },
                                         { "y", { 1, -1 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(HasDiagnosticContaining(
        report, "runtime input 'x' must have static shape"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ExecutableVerifierRejectsMissingRuntimeOutputShape)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                                     "Relu",
                                     { "x" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 1, 2 } },
                                         { "y", {} },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(
        HasDiagnosticContaining(report, "runtime output 'y' must have shape"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ExecutableVerifierRejectsNonFloat32Initializer)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                                     "Relu",
                                     { "x" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 1, 2 } },
                                         { "y", { 1, 2 } },
                                     });
    tc::frontend::Graph::InitVecT inits;
    auto init =
        MakeInitializer("w", { 2 }, tc::frontend::TypeInfo<int32_t>::type);
    init->set_values<int32_t>({ 1, 2 });
    inits.push_back(std::move(init));
    graph.set_inits(std::move(inits));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(
        HasDiagnosticContaining(report, "initializer 'w' must be float32"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ExecutableVerifierRejectsDynamicInitializerShape)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                                     "Relu",
                                     { "x" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 1, 2 } },
                                         { "y", { 1, 2 } },
                                     });
    tc::frontend::Graph::InitVecT inits;
    auto init =
        MakeInitializer("w", { -1 }, tc::frontend::TypeInfo<float>::type);
    init->set_values<float>({ 1.0f, 2.0f });
    inits.push_back(std::move(init));
    graph.set_inits(std::move(inits));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(HasDiagnosticContaining(
        report, "initializer 'w' must have static shape"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ExecutableVerifierRejectsInitializerWithoutData)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                                     "Relu",
                                     { "x" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 1, 2 } },
                                         { "y", { 1, 2 } },
                                     });
    tc::frontend::Graph::InitVecT inits;
    inits.push_back(
        MakeInitializer("w", { 2 }, tc::frontend::TypeInfo<float>::type));
    graph.set_inits(std::move(inits));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(
        HasDiagnosticContaining(report, "initializer 'w' must have data"))
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

TEST(GraphVerifierSemantic, ValidMulGraphPasses)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kMul, "Mul", { "x", "b" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << "expected valid mul graph to pass semantic checks\n"
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ValidReshapeGraphPasses)
{
    auto graph = MakeReshapeGraph(true);

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ReshapeWithoutInitializerShapeFails)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kReshape,
        "Reshape",
        { "x", "shape" },
        { "y" },
        {},
        {
            { "x", tc::frontend::TypeInfo<float>::type },
            { "shape", tc::frontend::TypeInfo<int64_t>::type },
            { "y", tc::frontend::TypeInfo<float>::type },
        },
        {
            { "x", { 1, 2, 2 } },
            { "shape", { 2 } },
            { "y", { 1, 4 } },
        });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "must be an INT64 initializer"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ExecutableVerifierAllowsReshapeInt64Initializer)
{
    auto graph = MakeReshapeGraph(false);

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecutable(graph, report))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, MulWithOneInputFails)
{
    auto graph = MakeSingleNodeGraph(
        tc::frontend::OpKind::kMul, "Mul", { "x" }, { "y" });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << DiagnosticsAsString(report);
    EXPECT_TRUE(HasDiagnosticContaining(report, "expects 2 input"))
        << "missing mul arity diagnostic";
}

TEST(GraphVerifierSemantic, MulWithMismatchedInputDtypesFails)
{
    auto graph =
        MakeSingleNodeGraph(tc::frontend::OpKind::kMul,
                            "Mul",
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
        << "missing mul dtype mismatch diagnostic";
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

TEST(GraphVerifierSemantic, ReluWithMismatchedInputOutputShapeFails)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kRelu,
                                     "Relu",
                                     { "x" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 2, 3 } },
                                         { "y", { 2, 4 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "output shape mismatch"))
        << "missing relu output shape mismatch diagnostic";
}

TEST(GraphVerifierSemantic, AddWithIncompatibleInputShapesFails)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kAdd,
                                     "Add",
                                     { "x", "b" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 2, 3 } },
                                         { "b", { 2, 4 } },
                                         { "y", { 2, 3 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "input shapes mismatch"))
        << "missing add input shape mismatch diagnostic";
}

TEST(GraphVerifierSemantic, AddWithBroadcastBiasShapePasses)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kAdd,
                                     "Add",
                                     { "x", "b" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 2, 4 } },
                                         { "b", { 4 } },
                                         { "y", { 2, 4 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, AddWithRankOneChannelLikeShapeFails)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kAdd,
                                     "Add",
                                     { "x", "b" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 1, 3, 4, 4 } },
                                         { "b", { 3 } },
                                         { "y", { 1, 3, 4, 4 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "input shapes mismatch"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, MulWithIncompatibleInputShapesFails)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kMul,
                                     "Mul",
                                     { "x", "b" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 2, 3 } },
                                         { "b", { 2, 4 } },
                                         { "y", { 2, 3 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "input shapes mismatch"))
        << "missing mul input shape mismatch diagnostic";
}

TEST(GraphVerifierSemantic, MulWithBroadcastBiasShapePasses)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kMul,
                                     "Mul",
                                     { "x", "b" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 2, 4 } },
                                         { "b", { 4 } },
                                         { "y", { 2, 4 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, MulWithRankOneChannelLikeShapeFails)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kMul,
                                     "Mul",
                                     { "x", "b" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 1, 3, 4, 4 } },
                                         { "b", { 3 } },
                                         { "y", { 1, 3, 4, 4 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "input shapes mismatch"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, MatMulWithIncompatibleInnerDimensionsFails)
{
    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kMatMul,
                                     "MatMul",
                                     { "x", "w" },
                                     { "y" },
                                     {},
                                     {},
                                     {
                                         { "x", { 2, 3 } },
                                         { "w", { 4, 5 } },
                                         { "y", { 2, 5 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "inner dimensions mismatch"))
        << "missing matmul inner dimensions mismatch diagnostic";
}

TEST(GraphVerifierSemantic, TransposeWithPermRankMismatchFails)
{
    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(MakeIntsAttr("perm", { 0, 1 }));

    auto graph = MakeSingleNodeGraph(tc::frontend::OpKind::kTranspose,
                                     "Transpose",
                                     { "x" },
                                     { "y" },
                                     std::move(attrs),
                                     {},
                                     {
                                         { "x", { 2, 3, 4 } },
                                         { "y", { 2, 3 } },
                                     });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "perm rank mismatch"))
        << "missing transpose perm rank mismatch diagnostic";
}

TEST(GraphVerifierSemantic, ValidConvGraphPasses)
{
    auto graph = MakeConvGraph(MakeConvAttrs());

    tc::frontend::verify::Report report;
    EXPECT_TRUE(tc::frontend::verify::VerifyGraphForExecution(graph, report))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvInputRankMismatchFails)
{
    auto graph = MakeConvGraph(MakeConvAttrs(),
                               {},
                               {
                                   { "x", { 1, 2, 8 } },
                                   { "w", { 2, 2, 2, 2 } },
                                   { "y", { 1, 2, 7, 7 } },
                               });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "input rank"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvWeightRankMismatchFails)
{
    auto graph = MakeConvGraph(MakeConvAttrs(),
                               {},
                               {
                                   { "x", { 1, 2, 8, 8 } },
                                   { "w", { 2, 2, 2 } },
                                   { "y", { 1, 2, 7, 7 } },
                               });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "weight rank"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvOutputRankMismatchFails)
{
    auto graph = MakeConvGraph(MakeConvAttrs(),
                               {},
                               {
                                   { "x", { 1, 2, 8, 8 } },
                                   { "w", { 2, 2, 2, 2 } },
                                   { "y", { 1, 2, 7 } },
                               });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "output rank"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvDtypeMismatchFails)
{
    auto graph =
        MakeConvGraph(MakeConvAttrs(),
                      {
                          { "x", tc::frontend::TypeInfo<float>::type },
                          { "w", tc::frontend::TypeInfo<int32_t>::type },
                          { "y", tc::frontend::TypeInfo<float>::type },
                      });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "input dtypes mismatch"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvChannelMismatchFails)
{
    auto graph = MakeConvGraph(MakeConvAttrs(),
                               {},
                               {
                                   { "x", { 1, 3, 8, 8 } },
                                   { "w", { 2, 2, 2, 2 } },
                                   { "y", { 1, 2, 7, 7 } },
                               });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "channel mismatch"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvRejectsGroupNotOne)
{
    auto graph = MakeConvGraph(
        MakeConvAttrs({ 2, 2 }, { 1, 1 }, { 0, 0, 0, 0 }, { 1, 1 }, 2));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "group must be 1"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvRejectsInvalidPadsLength)
{
    auto graph =
        MakeConvGraph(MakeConvAttrs({ 2, 2 }, { 1, 1 }, { 0, 0 }, { 1, 1 }));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "pads"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvRejectsInvalidStridesLength)
{
    auto graph =
        MakeConvGraph(MakeConvAttrs({ 2, 2 }, { 1 }, { 0, 0, 0, 0 }, { 1, 1 }));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "strides"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvRejectsInvalidDilationsLength)
{
    auto graph =
        MakeConvGraph(MakeConvAttrs({ 2, 2 }, { 1, 1 }, { 0, 0, 0, 0 }, { 1 }));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "dilations"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvKernelShapeMismatchFails)
{
    auto graph =
        MakeConvGraph(MakeConvAttrs({ 3, 3 }, { 1, 1 }, { 0, 0, 0, 0 }));

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "kernel_shape mismatch"))
        << DiagnosticsAsString(report);
}

TEST(GraphVerifierSemantic, ConvNonPositiveOutputShapeFails)
{
    auto graph = MakeConvGraph(
        MakeConvAttrs({ 2, 2 }, { 1, 1 }, { 0, 0, 0, 0 }, { 1, 1 }),
        {},
        {
            { "x", { 1, 2, 1, 1 } },
            { "w", { 2, 2, 2, 2 } },
            { "y", { 1, 2, 0, 0 } },
        });

    tc::frontend::verify::Report report;
    EXPECT_FALSE(tc::frontend::verify::VerifyGraphForExecution(graph, report));
    EXPECT_TRUE(HasDiagnosticContaining(report, "non-positive output shape"))
        << DiagnosticsAsString(report);
}

} // namespace
