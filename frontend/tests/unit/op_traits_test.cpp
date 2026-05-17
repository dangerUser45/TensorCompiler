#include "op_kind.hpp"
#include "op_traits.hpp"

#include <array>
#include <cstddef>

#include <gtest/gtest.h>

namespace {

struct ExpectedOpTrait final
{
    tc::frontend::OpKind kind;
    std::size_t min_inputs;
    std::size_t max_inputs;
    std::size_t outputs;
    bool supports_attributes;
    bool strict_mlir_supported;
};

constexpr std::array<ExpectedOpTrait, 8> kExpectedTraits = {
    ExpectedOpTrait{ tc::frontend::OpKind::kRelu, 1, 1, 1, false, true },
    ExpectedOpTrait{ tc::frontend::OpKind::kAdd, 2, 2, 1, false, true },
    ExpectedOpTrait{ tc::frontend::OpKind::kMul, 2, 2, 1, false, true },
    ExpectedOpTrait{ tc::frontend::OpKind::kConv, 2, 2, 1, true, true },
    ExpectedOpTrait{ tc::frontend::OpKind::kMatMul, 2, 2, 1, false, true },
    ExpectedOpTrait{ tc::frontend::OpKind::kReshape, 2, 2, 1, false, true },
    ExpectedOpTrait{ tc::frontend::OpKind::kTranspose, 1, 1, 1, true, true },
    ExpectedOpTrait{ tc::frontend::OpKind::kMaxPool, 1, 1, 1, true, true },
};

} // namespace

TEST(OpTraits, DefinesStableMetadataForEveryKnownOp)
{
    for (const auto& expected : kExpectedTraits) {
        const auto* traits = tc::frontend::GetOpTraits(expected.kind);

        ASSERT_NE(traits, nullptr) << tc::frontend::ToString(expected.kind);
        EXPECT_EQ(traits->kind, expected.kind);
        EXPECT_EQ(traits->inputs.min, expected.min_inputs);
        EXPECT_EQ(traits->inputs.max, expected.max_inputs);
        EXPECT_EQ(traits->outputs, expected.outputs);
        EXPECT_EQ(traits->supports_attributes, expected.supports_attributes);
        EXPECT_EQ(traits->strict_mlir_supported,
                  expected.strict_mlir_supported);
    }
}

TEST(OpTraits, UnknownOpHasNoTraitsAndNoCapabilities)
{
    EXPECT_EQ(tc::frontend::GetOpTraits(tc::frontend::OpKind::kUnknown),
              nullptr);
    EXPECT_FALSE(tc::frontend::IsKnownOpKind(tc::frontend::OpKind::kUnknown));
    EXPECT_FALSE(
        tc::frontend::SupportsAttributes(tc::frontend::OpKind::kUnknown));
    EXPECT_FALSE(
        tc::frontend::IsStrictMlirSupported(tc::frontend::OpKind::kUnknown));
}

TEST(OpTraits, KnownOpsHaveLookupHelpers)
{
    for (const auto& expected : kExpectedTraits) {
        EXPECT_TRUE(tc::frontend::IsKnownOpKind(expected.kind))
            << tc::frontend::ToString(expected.kind);
        EXPECT_EQ(tc::frontend::SupportsAttributes(expected.kind),
                  expected.supports_attributes)
            << tc::frontend::ToString(expected.kind);
        EXPECT_EQ(tc::frontend::IsStrictMlirSupported(expected.kind),
                  expected.strict_mlir_supported)
            << tc::frontend::ToString(expected.kind);
    }
}
