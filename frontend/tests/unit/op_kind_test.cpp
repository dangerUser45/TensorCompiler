#include "graph.hpp"
#include "op_kind.hpp"

#include <gtest/gtest.h>

namespace {

TEST(OpKind, ParsesKnownOperators)
{
    EXPECT_EQ(tc::frontend::OpKindFromString("Relu"),
              tc::frontend::OpKind::kRelu);
    EXPECT_EQ(tc::frontend::OpKindFromString("Add"),
              tc::frontend::OpKind::kAdd);
    EXPECT_EQ(tc::frontend::OpKindFromString("Mul"),
              tc::frontend::OpKind::kMul);
    EXPECT_EQ(tc::frontend::OpKindFromString("Conv"),
              tc::frontend::OpKind::kConv);
    EXPECT_EQ(tc::frontend::OpKindFromString("MatMul"),
              tc::frontend::OpKind::kMatMul);
    EXPECT_EQ(tc::frontend::OpKindFromString("Reshape"),
              tc::frontend::OpKind::kReshape);
    EXPECT_EQ(tc::frontend::OpKindFromString("Transpose"),
              tc::frontend::OpKind::kTranspose);
}

TEST(OpKind, MapsUnknownOperatorToUnknown)
{
    EXPECT_EQ(tc::frontend::OpKindFromString("UnsupportedConv"),
              tc::frontend::OpKind::kUnknown);
    EXPECT_EQ(tc::frontend::OpKindFromString(""),
              tc::frontend::OpKind::kUnknown);
}

TEST(OpKind, HasStableStringNames)
{
    EXPECT_EQ(tc::frontend::ToString(tc::frontend::OpKind::kRelu), "Relu");
    EXPECT_EQ(tc::frontend::ToString(tc::frontend::OpKind::kAdd), "Add");
    EXPECT_EQ(tc::frontend::ToString(tc::frontend::OpKind::kMul), "Mul");
    EXPECT_EQ(tc::frontend::ToString(tc::frontend::OpKind::kConv), "Conv");
    EXPECT_EQ(tc::frontend::ToString(tc::frontend::OpKind::kMatMul), "MatMul");
    EXPECT_EQ(tc::frontend::ToString(tc::frontend::OpKind::kReshape),
              "Reshape");
    EXPECT_EQ(tc::frontend::ToString(tc::frontend::OpKind::kTranspose),
              "Transpose");
    EXPECT_EQ(tc::frontend::ToString(tc::frontend::OpKind::kUnknown),
              "Unknown");
}

TEST(GraphNode, OpKindDefaultsToUnknown)
{
    tc::frontend::Node node;
    EXPECT_EQ(node.get_op_kind(), tc::frontend::OpKind::kUnknown);
}

TEST(GraphNode, OpKindCanBeSetExplicitly)
{
    tc::frontend::Node node;
    node.set_op_kind(tc::frontend::OpKind::kMatMul);
    EXPECT_EQ(node.get_op_kind(), tc::frontend::OpKind::kMatMul);
}

} // namespace
