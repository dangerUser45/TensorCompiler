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

TEST(ImporterOpKind, UnsupportedModelStillFails)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("candy-8.onnx").string(), model, graph, error));
    EXPECT_NE(error.find("unsupported operator"), std::string::npos) << error;
}
