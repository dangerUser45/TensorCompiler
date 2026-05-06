#include "onnx_importer.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::atomic_uint64_t g_temp_counter{ 0 };

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

::onnx::ModelProto BuildSingleNodeModel(
    const std::string& op_type,
    const std::function<void(::onnx::NodeProto&)>& configure_node)
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("importer_attr_norm_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 2, 2 });
    FillTensorValueInfo(graph->add_output(), "y", { 2, 2 });

    auto* node = graph->add_node();
    node->set_name("n0");
    node->set_op_type(op_type);
    node->add_input("x");
    node->add_output("y");

    configure_node(*node);
    return model;
}

class TempModelFile final
{
public:
    explicit TempModelFile(const ::onnx::ModelProto& model)
    {
        const auto now = std::chrono::high_resolution_clock::now()
                             .time_since_epoch()
                             .count();
        const auto seq = g_temp_counter.fetch_add(1, std::memory_order_relaxed);
        path_ = std::filesystem::temp_directory_path() /
                ("tc_importer_attr_" + std::to_string(now) + "_" +
                 std::to_string(seq) + ".onnx");

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

TEST(ImporterAttrNormalization, ReluUnexpectedAttributeFails)
{
    const auto model =
        BuildSingleNodeModel("Relu", [](::onnx::NodeProto& node) {
            auto* attr = node.add_attribute();
            attr->set_name("alpha");
            attr->set_type(::onnx::AttributeProto_AttributeType_FLOAT);
            attr->set_f(1.0f);
        });

    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("does not support attribute"), std::string::npos)
        << error;
}

TEST(ImporterAttrNormalization, MulUnexpectedAttributeFails)
{
    const auto model = BuildSingleNodeModel("Mul", [](::onnx::NodeProto& node) {
        node.add_input("rhs");

        auto* attr = node.add_attribute();
        attr->set_name("axis");
        attr->set_type(::onnx::AttributeProto_AttributeType_INT);
        attr->set_i(0);
    });

    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("does not support attribute"), std::string::npos)
        << error;
}

TEST(ImporterAttrNormalization, TransposeUnknownAttributeFails)
{
    const auto model =
        BuildSingleNodeModel("Transpose", [](::onnx::NodeProto& node) {
            auto* attr = node.add_attribute();
            attr->set_name("axes");
            attr->set_type(::onnx::AttributeProto_AttributeType_INTS);
            attr->add_ints(1);
            attr->add_ints(0);
        });

    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("unsupported attribute"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, TransposePermIsImportedAsInt64Vector)
{
    const auto model =
        BuildSingleNodeModel("Transpose", [](::onnx::NodeProto& node) {
            auto* attr = node.add_attribute();
            attr->set_name("perm");
            attr->set_type(::onnx::AttributeProto_AttributeType_INTS);
            attr->add_ints(1);
            attr->add_ints(0);
        });

    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    const auto& attrs = graph.get_nodes()[0]->get_attrs();
    ASSERT_EQ(attrs.size(), 1u);
    ASSERT_NE(attrs[0], nullptr);

    EXPECT_EQ(attrs[0]->get_name(), "perm");
    EXPECT_EQ(attrs[0]->get_data_type().id, tc::frontend::DataID::INT64);
    EXPECT_EQ(attrs[0]->get_values<int64_t>(), (std::vector<int64_t>{ 1, 0 }));
}
