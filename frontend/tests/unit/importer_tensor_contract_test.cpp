#include "onnx_importer.hpp"

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

namespace {

std::filesystem::path ModelPath(const std::string& file_name)
{
    return std::filesystem::path(TC_TEST_MODELS_DIR) / file_name;
}

TEST(ImporterTensorContract, SingleReluFillsInputAndOutputTensorDtypes)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("single_relu.onnx").string(), model, graph, error))
        << error;

    ASSERT_EQ(graph.get_input_tensors().size(), 1u);
    ASSERT_EQ(graph.get_output_tensors().size(), 1u);

    const auto& input = graph.get_input_tensors()[0];
    const auto& output = graph.get_output_tensors()[0];
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    EXPECT_EQ(input->get_data_type().id, tc::frontend::DataID::FLOAT);
    EXPECT_EQ(output->get_data_type().id, tc::frontend::DataID::FLOAT);

    ASSERT_EQ(input->get_shape().size(), 2u);
    EXPECT_EQ(input->get_shape()[0], 1);
    EXPECT_EQ(input->get_shape()[1], 2);
}

TEST(ImporterTensorContract, TwoTransposesFillsAllIoDtypes)
{
    ::onnx::ModelProto model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        ModelPath("two_transposes.onnx").string(), model, graph, error))
        << error;

    ASSERT_EQ(graph.get_input_tensors().size(), 1u);
    ASSERT_EQ(graph.get_output_tensors().size(), 1u);

    EXPECT_EQ(graph.get_input_tensors()[0]->get_data_type().id,
              tc::frontend::DataID::FLOAT);
    EXPECT_EQ(graph.get_output_tensors()[0]->get_data_type().id,
              tc::frontend::DataID::FLOAT);
}

} // namespace
