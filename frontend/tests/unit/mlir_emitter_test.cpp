#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "graph.hpp"
#include "mlir_emitter.hpp"
#include "onnx.pb.h"
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

std::unique_ptr<tc::frontend::Attribute> MakeIntAttr(
    std::string name,
    std::vector<int64_t> values)
{
    auto attr = std::make_unique<tc::frontend::Attribute>();
    attr->set_name(std::move(name));
    attr->set_values<int64_t>(std::move(values));
    return attr;
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

tc::frontend::Graph MakeConvGraphForMlir(tc::frontend::Node::AttrVecT attrs)
{
    tc::frontend::Graph graph;
    graph.set_name("conv_attr_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(
        MakeTensor("x", { 1, 2, 8, 8 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(
        MakeTensor("y", { 1, 2, 4, 4 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    tc::frontend::Graph::InitVecT inits;
    auto weight = std::make_unique<tc::frontend::Initializers>();
    weight->set_name("w");
    weight->set_shape({ 2, 2, 2, 2 });
    weight->set_values<float>(std::vector<float>(16, 1.0f));
    inits.push_back(std::move(weight));
    graph.set_inits(std::move(inits));

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("conv_0");
    node->set_name_op("Conv");
    node->set_op_kind(tc::frontend::OpKind::kConv);
    node->set_inputs({ "x", "w" });
    node->set_outputs({ "y" });
    node->set_attr(std::move(attrs));

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    return graph;
}

tc::frontend::Graph MakeAddWithInitializerGraph()
{
    tc::frontend::Graph graph;
    graph.set_name("add_initializer_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(MakeTensor("x", { 1, 2 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(MakeTensor("z", { 1, 2 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    tc::frontend::Graph::InitVecT inits;
    auto bias = std::make_unique<tc::frontend::Initializers>();
    bias->set_name("bias");
    bias->set_shape({ 2 });
    bias->set_values<float>({ 1.0f, 2.0f });
    inits.push_back(std::move(bias));
    graph.set_inits(std::move(inits));

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("add_bias");
    node->set_name_op("Add");
    node->set_op_kind(tc::frontend::OpKind::kAdd);
    node->set_inputs({ "x", "bias" });
    node->set_outputs({ "z" });

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    return graph;
}

tc::frontend::Graph MakeAddWithUsedAndUnusedInitializerGraph()
{
    tc::frontend::Graph graph = MakeAddWithInitializerGraph();

    tc::frontend::Graph::InitVecT inits;

    auto bias = std::make_unique<tc::frontend::Initializers>();
    bias->set_name("bias");
    bias->set_shape({ 2 });
    bias->set_values<float>({ 1.0f, 2.0f });
    inits.push_back(std::move(bias));

    auto unused = std::make_unique<tc::frontend::Initializers>();
    unused->set_name("unused");
    unused->set_shape({ 2 });
    unused->set_values<float>({ 3.0f, 4.0f });
    inits.push_back(std::move(unused));

    graph.set_inits(std::move(inits));
    return graph;
}

tc::frontend::Graph MakeSingleMulGraph()
{
    tc::frontend::Graph graph;
    graph.set_name("mul_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(MakeTensor("x", { 1, 2 }, tc::frontend::DataID::FLOAT));
    inputs.push_back(MakeTensor("y", { 1, 2 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(MakeTensor("z", { 1, 2 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("mul_0");
    node->set_name_op("Mul");
    node->set_op_kind(tc::frontend::OpKind::kMul);
    node->set_inputs({ "x", "y" });
    node->set_outputs({ "z" });

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    return graph;
}

tc::frontend::Graph MakeSingleReshapeGraph()
{
    tc::frontend::Graph graph;
    graph.set_name("reshape_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(
        MakeTensor("input", { 2, 3, 4 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(
        MakeTensor("output", { 6, 4 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("reshape_0");
    node->set_name_op("Reshape");
    node->set_op_kind(tc::frontend::OpKind::kReshape);
    node->set_inputs({ "input", "shape" });
    node->set_outputs({ "output" });

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    tc::frontend::Graph::InitVecT inits;
    auto shape_init = std::make_unique<tc::frontend::Initializers>();
    shape_init->set_name("shape");
    shape_init->set_shape({ 2 });
    shape_init->set_values<int64_t>({ 6, 4 });
    inits.push_back(std::move(shape_init));
    graph.set_inits(std::move(inits));

    return graph;
}

tc::frontend::Graph MakeSingleMaxPoolGraph()
{
    tc::frontend::Graph graph;
    graph.set_name("maxpool_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(
        MakeTensor("input", { 1, 1, 4, 4 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(
        MakeTensor("output", { 1, 1, 2, 2 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("maxpool_0");
    node->set_name_op("MaxPool");
    node->set_op_kind(tc::frontend::OpKind::kMaxPool);
    node->set_inputs({ "input" });
    node->set_outputs({ "output" });

    // Add MaxPool attributes
    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(MakeIntAttr("kernel_shape", { 2, 2 }));
    attrs.push_back(MakeIntAttr("strides", { 2, 2 }));
    attrs.push_back(MakeIntAttr("pads", { 0, 0, 0, 0 }));
    node->set_attr(std::move(attrs));

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    return graph;
}

void FillTensorValueInfo(::onnx::ValueInfoProto* value_info,
                         const std::string& name,
                         const std::vector<int64_t>& shape_dims)
{
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(::onnx::TensorProto_DataType_FLOAT);
    auto* shape = tensor_type->mutable_shape();
    for (int64_t dim : shape_dims) {
        shape->add_dim()->set_dim_value(dim);
    }
}

::onnx::ModelProto BuildBinaryOnnxModel(const std::string& op_type)
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("mlir_emitter_binary_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 1, 2 });
    FillTensorValueInfo(graph->add_input(), "rhs", { 1, 2 });
    FillTensorValueInfo(graph->add_output(), "y", { 1, 2 });

    auto* node = graph->add_node();
    node->set_name("mul_0");
    node->set_op_type(op_type);
    node->add_input("x");
    node->add_input("rhs");
    node->add_output("y");

    return model;
}

class TempModelFile final
{
public:
    explicit TempModelFile(const ::onnx::ModelProto& model)
    {
        static std::atomic_uint64_t seq{ 0 };
        const auto now = std::chrono::high_resolution_clock::now()
                             .time_since_epoch()
                             .count();
        path_ = std::filesystem::temp_directory_path() /
                ("tc_mlir_emitter_" + std::to_string(now) + "_" +
                 std::to_string(seq.fetch_add(1, std::memory_order_relaxed)) +
                 ".onnx");

        std::ofstream output(path_, std::ios::binary);
        if (!output.is_open() || !model.SerializeToOstream(&output)) {
            throw std::runtime_error("failed to write temp ONNX model");
        }
    }

    ~TempModelFile()
    {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

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

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
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

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
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

TEST(MlirEmitter, ProductionMlirFailsInsteadOfReturningTodoFallback)
{
    tc::frontend::Graph graph;
    graph.set_name("unsupported_for_production");

    std::string mlir_text;
    std::string error;

    EXPECT_FALSE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error));
    EXPECT_TRUE(mlir_text.empty()) << mlir_text;
    EXPECT_NE(error.find("unsupported graph for production MLIR"),
              std::string::npos)
        << error;
}

TEST(MlirEmitter, SingleReluModelRequiresLoweredMlirWithoutSkeletonTodo)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/single_relu.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @tc_model(%arg0: tensor<1x2xf32>) -> "
                             "tensor<1x2xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("arith.maximumf"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, ReluLoweringUsesZeroConstantInsteadOfIdentityMax)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/single_relu.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;

    EXPECT_NE(mlir_text.find("arith.constant dense<0.0> : tensor<1x2xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_EQ(mlir_text.find("arith.maximumf %arg0, %arg0 : tensor<1x2xf32>"),
              std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, TwoTransposesModelRequiresLoweredMlirWithoutSkeletonTodo)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/two_transposes.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @tc_model(%arg0: tensor<2x3x4xf32>) -> "
                             "tensor<3x2x4xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("linalg.transpose"), std::string::npos)
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

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(
        mlir_text.find("func.func @tc_model(%arg0: tensor<1x2xf32>, %arg1: "
                       "tensor<1x2xf32>) -> tensor<1x2xf32>"),
        std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("arith.addf"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, AddModelRequiresArithmeticAddLowering)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/add.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;

    EXPECT_NE(mlir_text.find("arith.addf"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, InitializerOperandEmitsConstantAndNotFunctionArgument)
{
    const auto graph = MakeAddWithInitializerGraph();

    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error))
        << error;

    EXPECT_NE(mlir_text.find("func.func @tc_model(%arg0: tensor<1x2xf32>) -> "
                             "tensor<1x2xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_EQ(mlir_text.find("%arg1"), std::string::npos) << mlir_text;
    EXPECT_NE(
        mlir_text.find("arith.constant dense<[1.0, 2.0]> : tensor<2xf32>"),
        std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("linalg.broadcast"), std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("arith.addf"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, UnusedInitializerDoesNotEmitConstant)
{
    const auto graph = MakeAddWithUsedAndUnusedInitializerGraph();

    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error))
        << error;

    EXPECT_NE(mlir_text.find("dense<[1.0, 2.0]>"), std::string::npos)
        << mlir_text;
    EXPECT_EQ(mlir_text.find("dense<[3.0, 4.0]>"), std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, MulGraphEmitsTypedMainWithMulf)
{
    const auto graph = MakeSingleMulGraph();
    ASSERT_EQ(graph.get_input_tensors().size(), 2u);
    ASSERT_EQ(graph.get_output_tensors().size(), 1u);
    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_EQ(graph.get_nodes()[0]->get_op_kind(), tc::frontend::OpKind::kMul);
    ASSERT_EQ(graph.get_nodes()[0]->get_inputs().size(), 2u);
    ASSERT_EQ(graph.get_nodes()[0]->get_outputs().size(), 1u);

    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(
        mlir_text.find("func.func @tc_model(%arg0: tensor<1x2xf32>, %arg1: "
                       "tensor<1x2xf32>) -> tensor<1x2xf32>"),
        std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("op=Mul"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("arith.mulf"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, MulOnnxModelRequiresArithmeticMulLowering)
{
    const TempModelFile temp_model(BuildBinaryOnnxModel("Mul"));

    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("op=Mul"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("arith.mulf"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
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

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(
        mlir_text.find("func.func @tc_model(%arg0: tensor<1x2xf32>, %arg1: "
                       "tensor<2x4xf32>) -> tensor<1x4xf32>"),
        std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("linalg.matmul"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("tensor.empty() : tensor<1x4xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, MatMulModelRequiresLinalgMatmulLowering)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/matmul_2d.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;

    EXPECT_NE(mlir_text.find("linalg.matmul"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("tensor.empty() : tensor<3x3xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, GemmBiasAddLowersWithExplicitBroadcastBeforeAdd)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/matmul_addmm.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;

    EXPECT_NE(mlir_text.find("linalg.broadcast ins(%arg2 : tensor<4xf32>)"),
              std::string::npos)
        << mlir_text;
    EXPECT_EQ(mlir_text.find(", %arg2 : tensor<2x4xf32>"), std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, ConvModelLowersToLinalgConvWithInitializerConstants)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/conv_2d_nchw_fchw.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc)"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @tc_model(%arg0: "
                             "tensor<1x2x8x8xf32>) -> "
                             "tensor<1x2x7x7xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("op=Conv"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("arith.constant dense<[-1.5"), std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("tensor.empty() : tensor<1x2x7x7xf32>"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("linalg.conv_2d_nchw_fchw"), std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("linalg.broadcast ins(%cst1 : tensor<2xf32>)"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("dimensions = [1]"), std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, ConvWithNonUnitStrideFailsProductionEmission)
{
    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(MakeIntAttr("kernel_shape", { 2, 2 }));
    attrs.push_back(MakeIntAttr("strides", { 2, 2 }));
    attrs.push_back(MakeIntAttr("pads", { 0, 0, 0, 0 }));
    attrs.push_back(MakeIntAttr("dilations", { 1, 1 }));
    attrs.push_back(MakeIntAttr("group", { 1 }));
    auto graph = MakeConvGraphForMlir(std::move(attrs));

    std::string mlir_text;
    std::string error;
    EXPECT_FALSE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error));
    EXPECT_TRUE(mlir_text.empty()) << mlir_text;
    EXPECT_NE(error.find("unsupported Conv attributes"), std::string::npos)
        << error;
}

TEST(MlirEmitter, ConvWithSameUpperPadsEmitsConvWithPadsAttribute)
{
    // Build a single-Conv graph mirroring MNIST Convolution28: input
    // tensor<1x1x28x28xf32>, weight tensor<8x1x5x5xf32>, output
    // tensor<1x8x28x28xf32>, pads=[2,2,2,2], strides=[1,1], dilations=[1,1].
    // The emitter must keep this as a single linalg.conv_2d_nchw_fchw op
    // carrying an explicit {pads = [...]} attribute, so the backend's
    // single-line regex parser can read pads without needing region-body
    // (tensor.pad) support.
    tc::frontend::Graph graph;
    graph.set_name("conv_padded_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(
        MakeTensor("x", { 1, 1, 28, 28 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(
        MakeTensor("y", { 1, 8, 28, 28 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    tc::frontend::Graph::InitVecT inits;
    auto weight = std::make_unique<tc::frontend::Initializers>();
    weight->set_name("w");
    weight->set_shape({ 8, 1, 5, 5 });
    weight->set_values<float>(std::vector<float>(8 * 1 * 5 * 5, 1.0f));
    inits.push_back(std::move(weight));
    graph.set_inits(std::move(inits));

    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(MakeIntAttr("kernel_shape", { 5, 5 }));
    attrs.push_back(MakeIntAttr("strides", { 1, 1 }));
    attrs.push_back(MakeIntAttr("pads", { 2, 2, 2, 2 }));
    attrs.push_back(MakeIntAttr("dilations", { 1, 1 }));
    attrs.push_back(MakeIntAttr("group", { 1 }));

    auto node = std::make_unique<tc::frontend::Node>();
    node->set_name_node("conv_0");
    node->set_name_op("Conv");
    node->set_op_kind(tc::frontend::OpKind::kConv);
    node->set_inputs({ "x", "w" });
    node->set_outputs({ "y" });
    node->set_attr(std::move(attrs));

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(node));
    graph.set_nodes(std::move(nodes));

    std::string mlir_text;
    std::string error;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc)"), std::string::npos) << mlir_text;
    // Padded Conv must NOT be decomposed into tensor.pad anymore.
    EXPECT_EQ(mlir_text.find("tensor.pad"), std::string::npos) << mlir_text;
    EXPECT_EQ(mlir_text.find("tensor.yield"), std::string::npos) << mlir_text;
    // Padded shape (1x1x32x32) must not appear because we no longer materialize
    // a padded tensor on the frontend side.
    EXPECT_EQ(mlir_text.find("tensor<1x1x32x32xf32>"), std::string::npos)
        << mlir_text;
    // Conv carries an explicit {pads = [2, 2, 2, 2]} attribute on the same
    // line as the op.
    EXPECT_NE(mlir_text.find("linalg.conv_2d_nchw_fchw {pads = [2, 2, 2, 2]} "
                             "ins(%arg0, %cst0 : tensor<1x1x28x28xf32>, "
                             "tensor<8x1x5x5xf32>)"),
              std::string::npos)
        << mlir_text;
    // Output type unchanged (1x8x28x28).
    EXPECT_NE(mlir_text.find("tensor<1x8x28x28xf32>"), std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, ConvWithZeroPadsOmitsPadsAttribute)
{
    // Default unpadded Conv must keep the existing one-liner spelling without
    // any {pads = ...} clause, so the backend's existing regex (which has the
    // attribute group as optional) keeps matching it byte-for-byte.
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/conv_2d_nchw_fchw.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error))
        << error;

    EXPECT_EQ(mlir_text.find("{pads ="), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("linalg.conv_2d_nchw_fchw ins("),
              std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, MatMulLoweringUsesStableTemporaryOrdering)
{
    const auto graph = MakeSingleMatMulGraph();
    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;

    const auto empty_pos = mlir_text.find("%t0 = tensor.empty()");
    const auto matmul_pos = mlir_text.find("%t1 = linalg.matmul");

    ASSERT_NE(empty_pos, std::string::npos) << mlir_text;
    ASSERT_NE(matmul_pos, std::string::npos) << mlir_text;
    EXPECT_LT(empty_pos, matmul_pos) << mlir_text;
}

TEST(MlirEmitter, MaxPoolGraphEmitsLinalgPoolingNchw)
{
    const auto graph = MakeSingleMaxPoolGraph();
    ASSERT_EQ(graph.get_input_tensors().size(), 1u);
    ASSERT_EQ(graph.get_output_tensors().size(), 1u);
    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_EQ(graph.get_nodes()[0]->get_op_kind(),
              tc::frontend::OpKind::kMaxPool);
    ASSERT_EQ(graph.get_nodes()[0]->get_inputs().size(), 1u);
    ASSERT_EQ(graph.get_nodes()[0]->get_outputs().size(), 1u);

    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;

    // Check function signature
    EXPECT_NE(mlir_text.find("func.func @tc_model(%arg0: tensor<1x1x4x4xf32>) "
                             "-> tensor<1x1x2x2xf32>"),
              std::string::npos)
        << mlir_text;

    // Check MaxPool operation uses linalg.pooling_nchw_max with explicit
    // kernel/strides attributes.
    EXPECT_NE(
        mlir_text.find(
            "linalg.pooling_nchw_max {kernel = [2, 2], strides = [2, 2]}"),
        std::string::npos)
        << mlir_text;

    // Check tensor.empty for output initialization
    EXPECT_NE(mlir_text.find("tensor.empty() : tensor<1x1x2x2xf32>"),
              std::string::npos)
        << mlir_text;

    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}

TEST(MlirEmitter, MultiNodeGraphPropagatesIntermediateTensorTypes)
{
    // Conv (1x1x8x8 -> 1x2x6x6) -> MaxPool (k=2,s=2 -> 1x2x3x3). Final output
    // type differs from intermediate Conv output. The emitter must lower
    // each op with the correct intermediate type, not fall back to the graph
    // output type.
    tc::frontend::Graph graph;
    graph.set_name("multi_node_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(
        MakeTensor("x", { 1, 1, 8, 8 }, tc::frontend::DataID::FLOAT));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(
        MakeTensor("y", { 1, 2, 3, 3 }, tc::frontend::DataID::FLOAT));
    graph.set_output_tensors(std::move(outputs));

    tc::frontend::Graph::InitVecT inits;
    auto weight = std::make_unique<tc::frontend::Initializers>();
    weight->set_name("w");
    weight->set_shape({ 2, 1, 3, 3 });
    weight->set_values<float>(std::vector<float>(2 * 1 * 3 * 3, 1.0f));
    inits.push_back(std::move(weight));
    graph.set_inits(std::move(inits));

    tc::frontend::Node::AttrVecT conv_attrs;
    conv_attrs.push_back(MakeIntAttr("kernel_shape", { 3, 3 }));
    conv_attrs.push_back(MakeIntAttr("strides", { 1, 1 }));
    conv_attrs.push_back(MakeIntAttr("pads", { 0, 0, 0, 0 }));
    conv_attrs.push_back(MakeIntAttr("dilations", { 1, 1 }));
    conv_attrs.push_back(MakeIntAttr("group", { 1 }));

    auto conv_node = std::make_unique<tc::frontend::Node>();
    conv_node->set_name_node("conv_0");
    conv_node->set_name_op("Conv");
    conv_node->set_op_kind(tc::frontend::OpKind::kConv);
    conv_node->set_inputs({ "x", "w" });
    conv_node->set_outputs({ "conv_out" });
    conv_node->set_attr(std::move(conv_attrs));

    tc::frontend::Node::AttrVecT pool_attrs;
    pool_attrs.push_back(MakeIntAttr("kernel_shape", { 2, 2 }));
    pool_attrs.push_back(MakeIntAttr("strides", { 2, 2 }));
    pool_attrs.push_back(MakeIntAttr("pads", { 0, 0, 0, 0 }));

    auto pool_node = std::make_unique<tc::frontend::Node>();
    pool_node->set_name_node("pool_0");
    pool_node->set_name_op("MaxPool");
    pool_node->set_op_kind(tc::frontend::OpKind::kMaxPool);
    pool_node->set_inputs({ "conv_out" });
    pool_node->set_outputs({ "y" });
    pool_node->set_attr(std::move(pool_attrs));

    tc::frontend::Graph::NodeVecT nodes;
    nodes.push_back(std::move(conv_node));
    nodes.push_back(std::move(pool_node));
    graph.set_nodes(std::move(nodes));

    std::string mlir_text;
    std::string error;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc)"), std::string::npos) << mlir_text;
    // Intermediate Conv output type must appear in the lowered MLIR.
    EXPECT_NE(mlir_text.find("tensor<1x2x6x6xf32>"), std::string::npos)
        << mlir_text;
    // Final output type must also appear (function signature + return).
    EXPECT_NE(mlir_text.find("tensor<1x2x3x3xf32>"), std::string::npos)
        << mlir_text;
    // No type mismatch fallback should leak in:
    // builtin.unrealized_conversion_cast should NOT be present because both
    // intermediate and output types are properly propagated.
    EXPECT_EQ(mlir_text.find("builtin.unrealized_conversion_cast"),
              std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, MnistFullGraphEmitsSuccessfully)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    const std::string model_path =
        std::string(TC_TEST_MODELS_DIR) + "/mnist-8.onnx";
    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(model_path, graph, error))
        << "import failed: " << error;

    std::string mlir_text;
    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(graph, mlir_text, error))
        << error;

    EXPECT_EQ(mlir_text.find("TODO(tc)"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @tc_model(%arg0: "
                             "tensor<1x1x28x28xf32>) -> tensor<1x10xf32>"),
              std::string::npos)
        << mlir_text;
    // No tensor.pad: padded Conv is emitted as a single op with {pads = [...]}
    // so the backend's regex parser does not need region-body support.
    EXPECT_EQ(mlir_text.find("tensor.pad"), std::string::npos) << mlir_text;
    EXPECT_EQ(mlir_text.find("tensor.yield"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("linalg.conv_2d_nchw_fchw {pads = [2, 2, 2, 2]}"),
              std::string::npos)
        << mlir_text;
    EXPECT_NE(
        mlir_text.find(
            "linalg.pooling_nchw_max {kernel = [2, 2], strides = [2, 2]}"),
        std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("linalg.matmul"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("tensor.reshape"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("arith.maximumf"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("arith.addf"), std::string::npos) << mlir_text;
    // Final output type appears in signature and return.
    EXPECT_NE(mlir_text.find("tensor<1x10xf32>"), std::string::npos)
        << mlir_text;
}

TEST(MlirEmitter, ReshapeGraphEmitsTensorReshape)
{
    const auto graph = MakeSingleReshapeGraph();
    ASSERT_EQ(graph.get_input_tensors().size(), 1u);
    ASSERT_EQ(graph.get_output_tensors().size(), 1u);
    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_EQ(graph.get_nodes()[0]->get_op_kind(),
              tc::frontend::OpKind::kReshape);
    ASSERT_EQ(graph.get_nodes()[0]->get_inputs().size(), 2u);
    ASSERT_EQ(graph.get_nodes()[0]->get_outputs().size(), 1u);

    std::string mlir_text;
    std::string error;

    ASSERT_TRUE(tc::frontend::mlir::EmitMlirModule(
        graph, mlir_text, error, tc::frontend::mlir::EmitMode::kLenient))
        << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(mlir_text.find("TODO(tc): Graph->MLIR lowering is not "
                             "implemented yet."),
              std::string::npos)
        << mlir_text;

    // Check function signature
    EXPECT_NE(mlir_text.find("func.func @tc_model(%arg0: tensor<2x3x4xf32>) "
                             "-> tensor<6x4xf32>"),
              std::string::npos)
        << mlir_text;

    // Check Reshape operation uses tensor.reshape
    EXPECT_NE(mlir_text.find("tensor.reshape"), std::string::npos) << mlir_text;

    // No unrealized_conversion_cast should be present
    EXPECT_EQ(mlir_text.find("builtin.unrealized_conversion_cast"),
              std::string::npos)
        << mlir_text;

    EXPECT_NE(mlir_text.find("return %"), std::string::npos) << mlir_text;
}
