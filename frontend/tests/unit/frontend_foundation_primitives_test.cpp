#include "frontend_constants.hpp"
#include "graph.hpp"
#include "graph_utils.hpp"
#include "shape_inference.hpp"
#include "type_info.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::unique_ptr<tc::frontend::Attribute> MakeIntAttr(
    std::string name,
    std::vector<int64_t> values)
{
    auto attr = std::make_unique<tc::frontend::Attribute>();
    attr->set_name(std::move(name));
    attr->set_values<int64_t>(std::move(values));
    return attr;
}

tc::frontend::Node MakeNamedNode(std::string name, tc::frontend::OpKind kind)
{
    tc::frontend::Node node;
    node.set_name_node(std::move(name));
    node.set_op_kind(kind);
    return node;
}

TEST(ShapeInference, ComputesBroadcastShapeWithRankExpansion)
{
    std::vector<int64_t> out_shape;

    EXPECT_TRUE(
        tc::frontend::ComputeBroadcastShape({ 2, 1, 4 }, { 3, 4 }, out_shape));

    EXPECT_EQ(out_shape, (std::vector<int64_t>{ 2, 3, 4 }));
}

TEST(ShapeInference, ComputesBroadcastShapeWithDynamicDimensions)
{
    std::vector<int64_t> out_shape;

    EXPECT_TRUE(tc::frontend::ComputeBroadcastShape(
        { -1, 1, 8 }, { 1, 3, 8 }, out_shape));

    EXPECT_EQ(out_shape, (std::vector<int64_t>{ -1, 3, 8 }));
}

TEST(ShapeInference, RejectsIncompatibleBroadcastDimensions)
{
    std::vector<int64_t> out_shape;

    EXPECT_FALSE(
        tc::frontend::ComputeBroadcastShape({ 2, 3 }, { 2, 4 }, out_shape));
}

TEST(ShapeInference, ComputesChannelBiasBroadcastShape)
{
    std::vector<int64_t> out_shape;

    EXPECT_TRUE(tc::frontend::ComputeChannelBiasBroadcastShape(
        { 1, 4, 8, 8 }, { 4 }, out_shape));

    EXPECT_EQ(out_shape, (std::vector<int64_t>{ 1, 4, 8, 8 }));
}

TEST(ShapeInference, InfersReshapeAxisAndCopiesZeroDimensions)
{
    std::vector<int64_t> out_shape;
    std::string error;

    EXPECT_TRUE(tc::frontend::InferReshapeOutputShape(
        { 2, 3, 4 }, { 0, -1 }, out_shape, error));

    EXPECT_TRUE(error.empty());
    EXPECT_EQ(out_shape, (std::vector<int64_t>{ 2, 12 }));
}

TEST(ShapeInference, RejectsMultipleReshapeInferenceAxes)
{
    std::vector<int64_t> out_shape;
    std::string error;

    EXPECT_FALSE(tc::frontend::InferReshapeOutputShape(
        { 2, 3, 4 }, { -1, -1 }, out_shape, error));

    EXPECT_NE(error.find("multiple -1"), std::string::npos);
}

TEST(ShapeInference, ComputesSpatialOutputSizeWithDilation)
{
    EXPECT_EQ(tc::frontend::ComputeSpatialOutputSize(10, 3, 2, 1, 1, 2), 4);
}

TEST(GraphUtils, FindsNamedAttributeAndSkipsNullAttributes)
{
    tc::frontend::Node node;
    tc::frontend::Node::AttrVecT attrs;
    attrs.push_back(nullptr);
    attrs.push_back(MakeIntAttr("perm", { 0, 2, 1 }));
    node.set_attr(std::move(attrs));

    const auto* attr = tc::frontend::FindAttr(node, "perm");

    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->get_values<int64_t>(), (std::vector<int64_t>{ 0, 2, 1 }));
    EXPECT_EQ(tc::frontend::FindAttr(node, "missing"), nullptr);
}

TEST(GraphUtils, FormatsNodeContextFromGraphNode)
{
    const tc::frontend::Node named =
        MakeNamedNode("conv0", tc::frontend::OpKind::kConv);
    const tc::frontend::Node unnamed =
        MakeNamedNode("", tc::frontend::OpKind::kRelu);

    EXPECT_EQ(tc::frontend::BuildNodeContext(named, 3), "node[3]('conv0')");
    EXPECT_EQ(tc::frontend::BuildNodeContext(unnamed, 4), "node[4]");
}

TEST(GraphUtils, FormatsNodeContextFromName)
{
    EXPECT_EQ(tc::frontend::BuildNodeContext("onnx_node", 5),
              "node[5]('onnx_node')");
    EXPECT_EQ(tc::frontend::BuildNodeContext("", 6), "node[6]");
}

TEST(GraphUtils, DetectsSyntheticBiasAdd)
{
    tc::frontend::Node add_node =
        MakeNamedNode("conv0.add", tc::frontend::OpKind::kAdd);
    tc::frontend::Node relu_node =
        MakeNamedNode("conv0.add", tc::frontend::OpKind::kRelu);

    EXPECT_TRUE(tc::frontend::IsSyntheticBiasAdd(add_node));
    EXPECT_FALSE(tc::frontend::IsSyntheticBiasAdd(relu_node));
}

TEST(GraphUtils, ClassifiesSupportedNumericDtypes)
{
    EXPECT_TRUE(
        tc::frontend::IsSupportedNumericDtype(tc::frontend::DataID::INT8));
    EXPECT_TRUE(tc::frontend::IsSupportedNumericDtype(
        tc::frontend::DataID::UNSIGNED_INT64));
    EXPECT_TRUE(
        tc::frontend::IsSupportedNumericDtype(tc::frontend::DataID::FLOAT));
    EXPECT_TRUE(
        tc::frontend::IsSupportedNumericDtype(tc::frontend::DataID::DOUBLE));

    EXPECT_FALSE(
        tc::frontend::IsSupportedNumericDtype(tc::frontend::DataID::COMPLEX64));
    EXPECT_FALSE(
        tc::frontend::IsSupportedNumericDtype(tc::frontend::DataID::STRING));
    EXPECT_FALSE(
        tc::frontend::IsSupportedNumericDtype(tc::frontend::DataID::UNDEFINED));
}

TEST(FrontendConstants, ExposesSyntheticAndAutoPadSpellings)
{
    EXPECT_EQ(tc::frontend::kSyntheticAddSuffix, std::string_view(".add"));
    EXPECT_EQ(tc::frontend::kSyntheticMatMulSuffix,
              std::string_view(".matmul"));
    EXPECT_EQ(tc::frontend::kAutoPadNotSet, std::string_view("NOTSET"));
    EXPECT_EQ(tc::frontend::kAutoPadSameUpper, std::string_view("SAME_UPPER"));
}

} // namespace
