#include "type_info.hpp"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

namespace {

TEST(DataIDToString, ReturnsCorrectStringForEachId)
{
    using namespace tc::frontend;
    EXPECT_STREQ(DataIDToString(DataID::UNDEFINED), "UNDEFINED");
    EXPECT_STREQ(DataIDToString(DataID::INT8), "INT8");
    EXPECT_STREQ(DataIDToString(DataID::INT16), "INT16");
    EXPECT_STREQ(DataIDToString(DataID::INT32), "INT32");
    EXPECT_STREQ(DataIDToString(DataID::INT64), "INT64");
    EXPECT_STREQ(DataIDToString(DataID::UNSIGNED_INT8), "UNSIGNED_INT8");
    EXPECT_STREQ(DataIDToString(DataID::UNSIGNED_INT16), "UNSIGNED_INT16");
    EXPECT_STREQ(DataIDToString(DataID::UNSIGNED_INT32), "UNSIGNED_INT32");
    EXPECT_STREQ(DataIDToString(DataID::UNSIGNED_INT64), "UNSIGNED_INT64");
    EXPECT_STREQ(DataIDToString(DataID::FLOAT), "FLOAT");
    EXPECT_STREQ(DataIDToString(DataID::DOUBLE), "DOUBLE");
    EXPECT_STREQ(DataIDToString(DataID::COMPLEX64), "COMPLEX64");
    EXPECT_STREQ(DataIDToString(DataID::COMPLEX128), "COMPLEX128");
    EXPECT_STREQ(DataIDToString(DataID::STRING), "STRING");
}

TEST(TypeInfo, IdMatchesDataIDForPrimitiveTypes)
{
    using namespace tc::frontend;
    EXPECT_EQ(TypeInfo<int8_t>::type.id, DataID::INT8);
    EXPECT_EQ(TypeInfo<int16_t>::type.id, DataID::INT16);
    EXPECT_EQ(TypeInfo<int32_t>::type.id, DataID::INT32);
    EXPECT_EQ(TypeInfo<int64_t>::type.id, DataID::INT64);
    EXPECT_EQ(TypeInfo<uint8_t>::type.id, DataID::UNSIGNED_INT8);
    EXPECT_EQ(TypeInfo<uint16_t>::type.id, DataID::UNSIGNED_INT16);
    EXPECT_EQ(TypeInfo<uint32_t>::type.id, DataID::UNSIGNED_INT32);
    EXPECT_EQ(TypeInfo<uint64_t>::type.id, DataID::UNSIGNED_INT64);
    EXPECT_EQ(TypeInfo<float>::type.id, DataID::FLOAT);
    EXPECT_EQ(TypeInfo<double>::type.id, DataID::DOUBLE);
    EXPECT_EQ(TypeInfo<std::string>::type.id, DataID::STRING);
}

TEST(TypeInfo, DataIDToStringRoundtripsWithTypeInfo)
{
    using namespace tc::frontend;
    EXPECT_STREQ(DataIDToString(TypeInfo<float>::type.id), "FLOAT");
    EXPECT_STREQ(DataIDToString(TypeInfo<int64_t>::type.id), "INT64");
}

} // namespace
