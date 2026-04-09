#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "graph.hpp"
#include "mlir_emitter.hpp"
#include "onnx_importer.hpp"
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

std::unique_ptr<tc::frontend::TensorInfo> MakeTensor(std::string name,
                                                     std::vector<int64_t> shape,
                                                     tc::frontend::DataID id)
{
    auto tensor = std::make_unique<tc::frontend::TensorInfo>();
    tensor->set_name(std::move(name));
    tensor->set_shape(std::move(shape));
    tensor->set_data_type({ id, "" });
    return tensor;
}

tc::frontend::Graph MakeSingleAddGraph()
{
    tc::frontend::Graph graph;
    graph.set_name("add_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(MakeTensor("x", { 1, 2 }, tc::frontend::DataID::FLOAT));
    inputs.push_back(MakeTensor("y", { 1, 2 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(MakeTensor("z", { 1, 2 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("add_0");
    node->set_name_op("Add");
    node->set_op_kind(tc::frontend::OpKind::kAdd);
    node->set_inputs({ "x", "y" });
    node->set_outputs({ "z" });

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    return graph;
}

tc::frontend::Graph MakeSingleMatMulGraph()
{
    tc::frontend::Graph graph;
    graph.set_name("matmul_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(MakeTensor("a", { 1, 2 }, tc::frontend::DataID::FLOAT));
    inputs.push_back(MakeTensor("b", { 2, 4 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(MakeTensor("c", { 1, 4 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("matmul_0");
    node->set_name_op("MatMul");
    node->set_op_kind(tc::frontend::OpKind::kMatMul);
    node->set_inputs({ "a", "b" });
    node->set_outputs({ "c" });

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

TEST(MlirEmitter, SingleReluModelRequiresLoweredMlirWithoutSkeletonTodo)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/single_relu.onnx";
    ASSERT_TRUE(
        tc::frontend::onnx::ImportOnnxToGraph(model_path, model, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(
        tc::frontend::mlir::EmitMlirModuleSkeleton(graph, mlir_text, error))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @main(%arg0: tensor<1x2xf32>) -> "
                             "tensor<1x2xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, TwoTransposesModelRequiresLoweredMlirWithoutSkeletonTodo)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/two_transposes.onnx";
    ASSERT_TRUE(
        tc::frontend::onnx::ImportOnnxToGraph(model_path, model, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(
        tc::frontend::mlir::EmitMlirModuleSkeleton(graph, mlir_text, error))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @main(%arg0: tensor<2x3x4xf32>) -> "
                             "tensor<3x2x4xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("op=Transpose"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, AddGraphEmitsTypedMainWithTwoArguments)
{
    const auto graph = MakeSingleAddGraph();
    ASSERT_EQ(graph.get_input_tensors().size(), 2u);
    ASSERT_EQ(graph.get_output_tensors().size(), 1u);
    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_EQ(graph.get_nodes()[0]->get_op_kind(), tc::frontend::OpKind::kAdd);
    ASSERT_EQ(graph.get_nodes()[0]->get_inputs().size(), 2u);
    ASSERT_EQ(graph.get_nodes()[0]->get_outputs().size(), 1u);

    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(
        tc::frontend::mlir::EmitMlirModuleSkeleton(graph, mlir_text, error))
        << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @main(%arg0: tensor<1x2xf32>, %arg1: "
                             "tensor<1x2xf32>) -> tensor<1x2xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("op=Add"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("return %arg0 : tensor<1x2xf32>"),
              std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, MatMulGraphEmitsTypedMainWithTwoArguments)
{
    const auto graph = MakeSingleMatMulGraph();
    ASSERT_EQ(graph.get_input_tensors().size(), 2u);
    ASSERT_EQ(graph.get_output_tensors().size(), 1u);
    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_EQ(graph.get_nodes()[0]->get_op_kind(),
              tc::frontend::OpKind::kMatMul);
    ASSERT_EQ(graph.get_nodes()[0]->get_inputs().size(), 2u);
    ASSERT_EQ(graph.get_nodes()[0]->get_outputs().size(), 1u);

    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(
        tc::frontend::mlir::EmitMlirModuleSkeleton(graph, mlir_text, error))
        << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @main(%arg0: tensor<1x2xf32>, %arg1: "
                             "tensor<2x4xf32>) -> tensor<1x4xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("op=MatMul"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("builtin.unrealized_conversion_cast"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("return %0 : tensor<1x4xf32>"), std::string::npos)
        << mlir_text;
}
