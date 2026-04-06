#include "onnx_importer.hpp"
#include "op_kind.hpp"

#include <algorithm>
#include <filesystem>
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

TEST(ImporterOpKind, UnsupportedModelStillFails)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("candy-8.onnx").string(), model, graph, error));
    EXPECT_NE(error.find("unsupported operator"), std::string::npos) << error;
}
