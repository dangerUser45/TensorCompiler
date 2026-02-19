#include "onnx_importer.hpp"

#include <algorithm>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }
    
    ::onnx::ModelProto model;
    std::string error;
    if (!tc::frontend::onnx::LoadOnnxModel(argv[1], model, error)) {
        std::cerr <<  error << std::endl;
        return 1;
    }
   
    const auto& graph = model.graph();
    std::cout << "IR version: "         << model.ir_version() << std::endl;
    std::cout << "Graph name: "         << graph.name() << "\n";
    std::cout << "Graph inputs: "       << graph.input_size() << std::endl;
    std::cout << "Graph outputs: "      << graph.output_size() << std::endl;
    std::cout << "Graph initializer: "  << graph.initializer_size() << std::endl;
    std::cout << "Graph value info: "   << graph.value_info_size() << std::endl;
    std::cout << "Graph node: "         << graph.node_size() << std::endl;

    const int limit = std::min(graph.node_size(), 100);
    for (int i = 0; i < limit; ++i) {
        const auto& node = graph.node(i);
        std::cout << "[" << i << "] op = " << node.op_type()
                << ", name = " << node.name() << std::endl;
    }

    return 0;
}
