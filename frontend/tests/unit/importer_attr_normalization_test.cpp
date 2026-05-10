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

::onnx::ModelProto BuildMaxPoolModel(
    const std::function<void(::onnx::NodeProto&)>& configure_node)
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("importer_maxpool_attr_norm_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 1, 1, 4, 4 });
    FillTensorValueInfo(graph->add_output(), "y", { 1, 1, 2, 2 });

    auto* node = graph->add_node();
    node->set_name("maxpool0");
    node->set_op_type("MaxPool");
    node->add_input("x");
    node->add_output("y");

    auto* kernel = node->add_attribute();
    kernel->set_name("kernel_shape");
    kernel->set_type(::onnx::AttributeProto_AttributeType_INTS);
    kernel->add_ints(2);
    kernel->add_ints(2);

    configure_node(*node);
    return model;
}

::onnx::ModelProto BuildConvModel(
    const std::function<void(::onnx::NodeProto&)>& configure_node)
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("importer_conv_attr_norm_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 1, 2, 8, 8 });
    FillTensorValueInfo(graph->add_output(), "y", { 1, 2, 7, 7 });
    auto* weight = graph->add_initializer();
    weight->set_name("w");
    weight->set_data_type(::onnx::TensorProto_DataType_FLOAT);
    weight->add_dims(2);
    weight->add_dims(2);
    weight->add_dims(2);
    weight->add_dims(2);
    for (int i = 0; i < 16; ++i) {
        weight->add_float_data(1.0f);
    }

    auto* node = graph->add_node();
    node->set_name("conv0");
    node->set_op_type("Conv");
    node->add_input("x");
    node->add_input("w");
    node->add_output("y");

    configure_node(*node);
    return model;
}

const tc::frontend::Attribute* FindAttr(const tc::frontend::Node& node,
                                        const std::string& name)
{
    for (const auto& attr : node.get_attrs()) {
        if (attr && attr->get_name() == name) {
            return attr.get();
        }
    }
    return nullptr;
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

TEST(ImporterAttrNormalization, ConvImportsDefaultAttributes)
{
    const TempModelFile temp_model(BuildConvModel([](::onnx::NodeProto&) {}));

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    const auto& node = *graph.get_nodes()[0];
    EXPECT_EQ(node.get_op_kind(), tc::frontend::OpKind::kConv);

    ASSERT_NE(FindAttr(node, "kernel_shape"), nullptr);
    EXPECT_EQ(FindAttr(node, "kernel_shape")->get_values<int64_t>(),
              (std::vector<int64_t>{ 2, 2 }));
    ASSERT_NE(FindAttr(node, "strides"), nullptr);
    EXPECT_EQ(FindAttr(node, "strides")->get_values<int64_t>(),
              (std::vector<int64_t>{ 1, 1 }));
    ASSERT_NE(FindAttr(node, "pads"), nullptr);
    EXPECT_EQ(FindAttr(node, "pads")->get_values<int64_t>(),
              (std::vector<int64_t>{ 0, 0, 0, 0 }));
    ASSERT_NE(FindAttr(node, "dilations"), nullptr);
    EXPECT_EQ(FindAttr(node, "dilations")->get_values<int64_t>(),
              (std::vector<int64_t>{ 1, 1 }));
    ASSERT_NE(FindAttr(node, "group"), nullptr);
    EXPECT_EQ(FindAttr(node, "group")->get_values<int64_t>(),
              (std::vector<int64_t>{ 1 }));
    ASSERT_NE(FindAttr(node, "auto_pad"), nullptr);
    EXPECT_EQ(FindAttr(node, "auto_pad")->get_values<std::string>(),
              (std::vector<std::string>{ "NOTSET" }));
}

TEST(ImporterAttrNormalization, ConvPreservesExplicitAttributes)
{
    const auto model = BuildConvModel([](::onnx::NodeProto& node) {
        auto* kernel = node.add_attribute();
        kernel->set_name("kernel_shape");
        kernel->set_type(::onnx::AttributeProto_AttributeType_INTS);
        kernel->add_ints(3);
        kernel->add_ints(3);

        auto* pads = node.add_attribute();
        pads->set_name("pads");
        pads->set_type(::onnx::AttributeProto_AttributeType_INTS);
        pads->add_ints(1);
        pads->add_ints(1);
        pads->add_ints(1);
        pads->add_ints(1);
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
    const auto& node = *graph.get_nodes()[0];
    ASSERT_NE(FindAttr(node, "kernel_shape"), nullptr);
    EXPECT_EQ(FindAttr(node, "kernel_shape")->get_values<int64_t>(),
              (std::vector<int64_t>{ 3, 3 }));
    ASSERT_NE(FindAttr(node, "pads"), nullptr);
    EXPECT_EQ(FindAttr(node, "pads")->get_values<int64_t>(),
              (std::vector<int64_t>{ 1, 1, 1, 1 }));
}

TEST(ImporterAttrNormalization, ConvUnsupportedAutoPadFails)
{
    const auto model = BuildConvModel([](::onnx::NodeProto& node) {
        auto* attr = node.add_attribute();
        attr->set_name("auto_pad");
        attr->set_type(::onnx::AttributeProto_AttributeType_STRING);
        attr->set_s("SAME_LOWER");
    });
    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("auto_pad"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, ConvAutoPadValidRejected)
{
    const auto model = BuildConvModel([](::onnx::NodeProto& node) {
        auto* attr = node.add_attribute();
        attr->set_name("auto_pad");
        attr->set_type(::onnx::AttributeProto_AttributeType_STRING);
        attr->set_s("VALID");
    });
    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("auto_pad"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, ConvAutoPadSameUpperNormalizesToExplicitPads)
{
    // Build a Conv equivalent to MNIST Convolution28: kernel=5x5, stride=1,
    // dilations=1, auto_pad=SAME_UPPER. Expected normalized pads = [2,2,2,2].
    const auto model = BuildConvModel([](::onnx::NodeProto& node) {
        node.clear_attribute();

        auto* kernel = node.add_attribute();
        kernel->set_name("kernel_shape");
        kernel->set_type(::onnx::AttributeProto_AttributeType_INTS);
        kernel->add_ints(5);
        kernel->add_ints(5);

        auto* strides = node.add_attribute();
        strides->set_name("strides");
        strides->set_type(::onnx::AttributeProto_AttributeType_INTS);
        strides->add_ints(1);
        strides->add_ints(1);

        auto* dilations = node.add_attribute();
        dilations->set_name("dilations");
        dilations->set_type(::onnx::AttributeProto_AttributeType_INTS);
        dilations->add_ints(1);
        dilations->add_ints(1);

        auto* auto_pad = node.add_attribute();
        auto_pad->set_name("auto_pad");
        auto_pad->set_type(::onnx::AttributeProto_AttributeType_STRING);
        auto_pad->set_s("SAME_UPPER");
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
    const auto& node = *graph.get_nodes()[0];
    EXPECT_EQ(node.get_op_kind(), tc::frontend::OpKind::kConv);

    ASSERT_NE(FindAttr(node, "pads"), nullptr);
    EXPECT_EQ(FindAttr(node, "pads")->get_values<int64_t>(),
              (std::vector<int64_t>{ 2, 2, 2, 2 }));
    ASSERT_NE(FindAttr(node, "auto_pad"), nullptr);
    EXPECT_EQ(FindAttr(node, "auto_pad")->get_values<std::string>(),
              (std::vector<std::string>{ "NOTSET" }));
}

TEST(ImporterAttrNormalization, ConvAutoPadSameUpperWithExplicitPadsRejected)
{
    // ONNX semantics: if auto_pad != NOTSET, pads MUST NOT be set.
    const auto model = BuildConvModel([](::onnx::NodeProto& node) {
        node.clear_attribute();

        auto* kernel = node.add_attribute();
        kernel->set_name("kernel_shape");
        kernel->set_type(::onnx::AttributeProto_AttributeType_INTS);
        kernel->add_ints(3);
        kernel->add_ints(3);

        auto* pads = node.add_attribute();
        pads->set_name("pads");
        pads->set_type(::onnx::AttributeProto_AttributeType_INTS);
        pads->add_ints(1);
        pads->add_ints(1);
        pads->add_ints(1);
        pads->add_ints(1);

        auto* auto_pad = node.add_attribute();
        auto_pad->set_name("auto_pad");
        auto_pad->set_type(::onnx::AttributeProto_AttributeType_STRING);
        auto_pad->set_s("SAME_UPPER");
    });
    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("auto_pad"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, MaxPoolImportsDefaultAttributes)
{
    const TempModelFile temp_model(
        BuildMaxPoolModel([](::onnx::NodeProto&) {}));

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 1u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    const auto& node = *graph.get_nodes()[0];
    EXPECT_EQ(node.get_op_kind(), tc::frontend::OpKind::kMaxPool);

    ASSERT_NE(FindAttr(node, "kernel_shape"), nullptr);
    EXPECT_EQ(FindAttr(node, "kernel_shape")->get_values<int64_t>(),
              (std::vector<int64_t>{ 2, 2 }));
    ASSERT_NE(FindAttr(node, "strides"), nullptr);
    EXPECT_EQ(FindAttr(node, "strides")->get_values<int64_t>(),
              (std::vector<int64_t>{ 1, 1 }));
    ASSERT_NE(FindAttr(node, "pads"), nullptr);
    EXPECT_EQ(FindAttr(node, "pads")->get_values<int64_t>(),
              (std::vector<int64_t>{ 0, 0, 0, 0 }));
    ASSERT_NE(FindAttr(node, "auto_pad"), nullptr);
    EXPECT_EQ(FindAttr(node, "auto_pad")->get_values<std::string>(),
              (std::vector<std::string>{ "NOTSET" }));
}

TEST(ImporterAttrNormalization, MaxPoolPreservesExplicitStridesAndPads)
{
    const auto model = BuildMaxPoolModel([](::onnx::NodeProto& node) {
        auto* strides = node.add_attribute();
        strides->set_name("strides");
        strides->set_type(::onnx::AttributeProto_AttributeType_INTS);
        strides->add_ints(2);
        strides->add_ints(2);

        auto* pads = node.add_attribute();
        pads->set_name("pads");
        pads->set_type(::onnx::AttributeProto_AttributeType_INTS);
        pads->add_ints(0);
        pads->add_ints(0);
        pads->add_ints(1);
        pads->add_ints(1);
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
    const auto& node = *graph.get_nodes()[0];
    ASSERT_NE(FindAttr(node, "strides"), nullptr);
    EXPECT_EQ(FindAttr(node, "strides")->get_values<int64_t>(),
              (std::vector<int64_t>{ 2, 2 }));
    ASSERT_NE(FindAttr(node, "pads"), nullptr);
    EXPECT_EQ(FindAttr(node, "pads")->get_values<int64_t>(),
              (std::vector<int64_t>{ 0, 0, 1, 1 }));
}

TEST(ImporterAttrNormalization, MaxPoolMissingKernelShapeFails)
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("importer_maxpool_no_kernel_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 1, 1, 4, 4 });
    FillTensorValueInfo(graph->add_output(), "y", { 1, 1, 2, 2 });

    auto* node = graph->add_node();
    node->set_name("maxpool0");
    node->set_op_type("MaxPool");
    node->add_input("x");
    node->add_output("y");

    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph imported_graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, imported_graph, error));
    EXPECT_NE(error.find("kernel_shape"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, MaxPoolUnsupportedAutoPadFails)
{
    const auto model = BuildMaxPoolModel([](::onnx::NodeProto& node) {
        auto* attr = node.add_attribute();
        attr->set_name("auto_pad");
        attr->set_type(::onnx::AttributeProto_AttributeType_STRING);
        attr->set_s("SAME_UPPER");
    });
    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("auto_pad"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, MaxPoolCeilModeRejected)
{
    const auto model = BuildMaxPoolModel([](::onnx::NodeProto& node) {
        auto* attr = node.add_attribute();
        attr->set_name("ceil_mode");
        attr->set_type(::onnx::AttributeProto_AttributeType_INT);
        attr->set_i(1);
    });
    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("ceil_mode"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, MaxPoolNonDefaultDilationsRejected)
{
    const auto model = BuildMaxPoolModel([](::onnx::NodeProto& node) {
        auto* attr = node.add_attribute();
        attr->set_name("dilations");
        attr->set_type(::onnx::AttributeProto_AttributeType_INTS);
        attr->add_ints(2);
        attr->add_ints(2);
    });
    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("dilations"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, MaxPoolUnknownAttributeRejected)
{
    const auto model = BuildMaxPoolModel([](::onnx::NodeProto& node) {
        auto* attr = node.add_attribute();
        attr->set_name("storage_order");
        attr->set_type(::onnx::AttributeProto_AttributeType_INT);
        attr->set_i(1);
    });
    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error));
    EXPECT_NE(error.find("storage_order"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, MaxPoolIndicesOutputRejected)
{
    ::onnx::ModelProto model;
    model.set_ir_version(8);

    auto* opset = model.add_opset_import();
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("importer_maxpool_indices_graph");

    FillTensorValueInfo(graph->add_input(), "x", { 1, 1, 4, 4 });
    FillTensorValueInfo(graph->add_output(), "y", { 1, 1, 2, 2 });
    FillTensorValueInfo(graph->add_output(), "indices", { 1, 1, 2, 2 });

    auto* node = graph->add_node();
    node->set_name("maxpool0");
    node->set_op_type("MaxPool");
    node->add_input("x");
    node->add_output("y");
    node->add_output("indices");

    auto* kernel = node->add_attribute();
    kernel->set_name("kernel_shape");
    kernel->set_type(::onnx::AttributeProto_AttributeType_INTS);
    kernel->add_ints(2);
    kernel->add_ints(2);

    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph imported_graph;
    std::string error;

    EXPECT_FALSE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, imported_graph, error));
    EXPECT_NE(error.find("expects 1 output"), std::string::npos) << error;
}

TEST(ImporterAttrNormalization, ConvWithBiasNormalizesToConvAdd)
{
    const auto model =
        BuildConvModel([](::onnx::NodeProto& node) { node.add_input("bias"); });
    const TempModelFile temp_model(model);

    ::onnx::ModelProto loaded_model;
    tc::frontend::Graph graph;
    std::string error;

    ASSERT_TRUE(tc::frontend::onnx::ImportOnnxToGraph(
        temp_model.path().string(), loaded_model, graph, error))
        << error;

    ASSERT_EQ(graph.get_nodes().size(), 2u);
    ASSERT_NE(graph.get_nodes()[0], nullptr);
    ASSERT_NE(graph.get_nodes()[1], nullptr);
    EXPECT_EQ(graph.get_nodes()[0]->get_op_kind(), tc::frontend::OpKind::kConv);
    EXPECT_EQ(graph.get_nodes()[1]->get_op_kind(), tc::frontend::OpKind::kAdd);
    EXPECT_EQ(graph.get_nodes()[0]->get_inputs(),
              (std::vector<std::string>{ "x", "w" }));
    EXPECT_EQ(graph.get_nodes()[1]->get_inputs(),
              (std::vector<std::string>{ "y__conv_0", "bias" }));
    EXPECT_EQ(graph.get_nodes()[1]->get_outputs(),
              (std::vector<std::string>{ "y" }));
}
