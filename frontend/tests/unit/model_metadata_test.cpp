#include "graph.hpp"
#include "model_metadata.hpp"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::unique_ptr<tc::frontend::TensorInfo> MakeTensor(std::string name,
                                                     std::vector<int64_t> shape)
{
    auto tensor = std::make_unique<tc::frontend::TensorInfo>();
    tensor->set_name(std::move(name));
    tensor->set_shape(std::move(shape));
    tensor->set_data_type(tc::frontend::TypeInfo<float>::type);
    return tensor;
}

tc::frontend::Graph MakeExecReadyGraph()
{
    tc::frontend::Graph graph;
    graph.set_name("metadata_graph");

    tc::frontend::Graph::TensVecT inputs;
    inputs.push_back(MakeTensor("input_0", { 1, 2, 8, 8 }));
    graph.set_input_tensors(std::move(inputs));

    tc::frontend::Graph::TensVecT outputs;
    outputs.push_back(MakeTensor("output_0", { 1, 2, 7, 7 }));
    graph.set_output_tensors(std::move(outputs));

    return graph;
}

} // namespace

TEST(ModelMetadata, EmitsDeterministicExecJson)
{
    const auto graph = MakeExecReadyGraph();

    std::string json;
    std::string error;
    ASSERT_TRUE(tc::frontend::metadata::BuildMetadataJson(graph, json, error))
        << error;

    const std::string expected = "{\n"
                                 "  \"model_entry\": \"tc_model\",\n"
                                 "  \"input_count\": 1,\n"
                                 "  \"inputs\": [\n"
                                 "    {\n"
                                 "      \"name\": \"input_0\",\n"
                                 "      \"dtype\": \"float32\",\n"
                                 "      \"shape\": [1, 2, 8, 8],\n"
                                 "      \"layout\": \"NCHW\",\n"
                                 "      \"byte_size\": 512\n"
                                 "    }\n"
                                 "  ],\n"
                                 "  \"output_count\": 1,\n"
                                 "  \"outputs\": [\n"
                                 "    {\n"
                                 "      \"name\": \"output_0\",\n"
                                 "      \"dtype\": \"float32\",\n"
                                 "      \"shape\": [1, 2, 7, 7],\n"
                                 "      \"layout\": \"NCHW\",\n"
                                 "      \"byte_size\": 392\n"
                                 "    }\n"
                                 "  ]\n"
                                 "}\n";
    EXPECT_EQ(json, expected);
}

TEST(ModelMetadata, RejectsNonFloatRuntimeTensor)
{
    auto graph = MakeExecReadyGraph();
    tc::frontend::Graph::TensVecT inputs;
    auto tensor = MakeTensor("input_0", { 1, 2, 8, 8 });
    tensor->set_data_type(tc::frontend::TypeInfo<int32_t>::type);
    inputs.push_back(std::move(tensor));
    graph.set_input_tensors(std::move(inputs));

    std::string json;
    std::string error;
    EXPECT_FALSE(tc::frontend::metadata::BuildMetadataJson(graph, json, error));
    EXPECT_NE(error.find("float32"), std::string::npos) << error;
}
