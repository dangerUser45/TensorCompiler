#include "onnx_importer.hpp"
#include "op_kind.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::filesystem::path ModelsDir()
{
    return std::filesystem::path(TC_TEST_MODELS_DIR);
}

std::filesystem::path ModelPath(const std::string& file_name)
{
    return ModelsDir() / file_name;
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

::onnx::ModelProto BuildBinaryNodeModel(const std::string& op_type)
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("importer_op_kind_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 2, 2 });
    FillTensorValueInfo(graph->add_input(), "rhs", { 2, 2 });
    FillTensorValueInfo(graph->add_output(), "y", { 2, 2 });

    auto* node = graph->add_node();
    node->set_name("n0");
    node->set_op_type(op_type);
    node->add_input("x");
    node->add_input("rhs");
    node->add_output("y");

    return model;
}

::onnx::ModelProto BuildReshapeModel()
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("importer_reshape_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 1, 2, 2 });
    FillTensorValueInfo(graph->add_output(), "y", { 1, 4 });

    auto* shape_init = graph->add_initializer();
    shape_init->set_name("shape");
    shape_init->set_data_type(::onnx::TensorProto_DataType_INT64);
    shape_init->add_dims(2);
    shape_init->add_int64_data(1);
    shape_init->add_int64_data(4);

    auto* node = graph->add_node();
    node->set_name("reshape_0");
    node->set_op_type("Reshape");
    node->add_input("x");
    node->add_input("shape");
    node->add_output("y");

    return model;
}

::onnx::ModelProto BuildMaxPoolModel()
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("importer_maxpool_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 1, 1, 4, 4 });
    FillTensorValueInfo(graph->add_output(), "y", { 1, 1, 2, 2 });

    auto* node = graph->add_node();
    node->set_name("maxpool_0");
    node->set_op_type("MaxPool");
    node->add_input("x");
    node->add_output("y");

    // Add MaxPool attributes
    auto* kernel_shape = node->add_attribute();
    kernel_shape->set_name("kernel_shape");
    kernel_shape->set_type(::onnx::AttributeProto::INTS);
    kernel_shape->add_ints(2);
    kernel_shape->add_ints(2);

    auto* strides = node->add_attribute();
    strides->set_name("strides");
    strides->set_type(::onnx::AttributeProto::INTS);
    strides->add_ints(2);
    strides->add_ints(2);

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
                ("tc_importer_op_kind_" + std::to_string(now) + "_" +
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

} // namespace

TEST(ImporterOpKind, SingleReluSetsReluKind)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("single_relu.onnx").string(), model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    EXPECT_EQ(graph.get_nodes()[0]->get_op_kind(), tc::frontend::OpKind::kRelu);
}

TEST(ImporterOpKind, TwoTransposesSetsTransposeKindForAllNodes)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("two_transposes.onnx").string(), model, graph, error))
        << error;

    ASSERT_FALSE(graph.get_nodes().empty());
    EXPECT_TRUE(std::all_of(graph.get_nodes().begin(),
                            graph.get_nodes().end(),
                            [](const auto& node_ptr) {
                                return node_ptr != nullptr &&
                                       node_ptr->get_op_kind() ==
                                           tc::frontend::OpKind::kTranspose;
                            }));
}

TEST(ImporterOpKind, MulSetsMulKind)
{
    const TempModelFile temp_model(BuildBinaryNodeModel("Mul"));

    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    EXPECT_EQ(graph.get_nodes()[0]->get_op_kind(), tc::frontend::OpKind::kMul);
}

TEST(ImporterOpKind, ConvFixtureSetsConvAndBiasAddKinds)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("conv_2d_nchw_fchw.onnx").string(), model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 2u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    ASSERT_NE(graph.get_nodes()[1], nullptr);
    EXPECT_EQ(graph.get_nodes()[0]->get_op_kind(), tc::frontend::OpKind::kConv);
    EXPECT_EQ(graph.get_nodes()[1]->get_op_kind(), tc::frontend::OpKind::kAdd);
}

TEST(ImporterOpKind, ReshapeSetsReshapeKind)
{
    const TempModelFile temp_model(BuildReshapeModel());

    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    EXPECT_EQ(graph.get_nodes()[0]->get_op_kind(),
              tc::frontend::OpKind::kReshape);
}

TEST(ImporterOpKind, MaxPoolSetsMaxPoolKind)
{
    const TempModelFile temp_model(BuildMaxPoolModel());

    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    EXPECT_EQ(graph.get_nodes()[0]->get_op_kind(),
              tc::frontend::OpKind::kMaxPool);
}

TEST(ImporterOpKind, MnistModelImportsAllSupportedOperators)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("mnist-8.onnx").string(), model, graph, error))
        << error;

    // The MNIST graph contains Reshape, Conv (with auto_pad=SAME_UPPER
    // normalized to explicit pads), Add, Relu, MaxPool, MatMul.
    bool has_conv = false;
    bool has_maxpool = false;
    bool has_reshape = false;
    bool has_matmul = false;
    for (const auto& node_ptr : graph.get_nodes()) {
        ASSERT_NE(node_ptr, nullptr);
        switch (node_ptr->get_op_kind()) {
            case tc::frontend::OpKind::kConv:
                has_conv = true;
                break;
            case tc::frontend::OpKind::kMaxPool:
                has_maxpool = true;
                break;
            case tc::frontend::OpKind::kReshape:
                has_reshape = true;
                break;
            case tc::frontend::OpKind::kMatMul:
                has_matmul = true;
                break;
            default:
                break;
        }
    }
    EXPECT_TRUE(has_conv);
    EXPECT_TRUE(has_maxpool);
    EXPECT_TRUE(has_reshape);
    EXPECT_TRUE(has_matmul);
}

TEST(ImporterOpKind, MnistConvNodesNormalizeSameUpperToExplicitPads)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("mnist-8.onnx").string(), model, graph, error))
        << error;

    // Both MNIST Convs (kernel=5x5, stride=1, dilation=1, SAME_UPPER) must
    // normalize to pads=[2,2,2,2] with auto_pad="NOTSET".
    int conv_count = 0;
    for (const auto& node_ptr : graph.get_nodes()) {
        ASSERT_NE(node_ptr, nullptr);
        if (node_ptr->get_op_kind() != tc::frontend::OpKind::kConv) {
            continue;
        }
        ++conv_count;
        const tc::frontend::Attribute* pads_attr = nullptr;
        const tc::frontend::Attribute* auto_pad_attr = nullptr;
        for (const auto& attr : node_ptr->get_attrs()) {
            if (!attr) {
                continue;
            }
            if (attr->get_name() == "pads") {
                pads_attr = attr.get();
            } else if (attr->get_name() == "auto_pad") {
                auto_pad_attr = attr.get();
            }
        }
        ASSERT_NE(pads_attr, nullptr);
        EXPECT_EQ(pads_attr->get_values<int64_t>(),
                  (std::vector<int64_t>{ 2, 2, 2, 2 }));
        ASSERT_NE(auto_pad_attr, nullptr);
        EXPECT_EQ(auto_pad_attr->get_values<std::string>(),
                  (std::vector<std::string>{ "NOTSET" }));
    }
    EXPECT_EQ(conv_count, 2);
}

TEST(ImporterOpKind, UnsupportedModelStillFails)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("candy-8.onnx").string(), model, graph, error));
    EXPECT_NE(error.find("unsupported operator"), std::string::npos) << error;
}
