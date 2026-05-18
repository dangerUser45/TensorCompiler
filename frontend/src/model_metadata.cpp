#include "model_metadata.hpp"
#include "graph.hpp"

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace {

std::string JsonEscape(std::string_view value)
{
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            default:
                out += c;
                break;
        }
    }
    return out;
}

std::string FormatShape(const std::vector<int64_t>& shape)
{
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

bool ComputeByteSize(const tc::frontend::TensorInfo& tensor,
                     int64_t& out_byte_size,
                     std::string& out_error,
                     std::string_view role)
{
    if (tensor.get_data_type().id != tc::frontend::DataID::FLOAT) {
        out_error = "ERROR: metadata " + std::string(role) + " tensor '" +
                    tensor.get_name() + "' must be float32";
        return false;
    }

    int64_t elements = 1;
    const auto& shape = tensor.get_shape();
    if (shape.empty()) {
        out_error = "ERROR: metadata " + std::string(role) + " tensor '" +
                    tensor.get_name() + "' must have shape";
        return false;
    }
    for (const int64_t dim : shape) {
        if (dim < 0) {
            out_error = "ERROR: metadata " + std::string(role) + " tensor '" +
                        tensor.get_name() + "' must have static shape";
            return false;
        }
        elements *= dim;
    }

    out_byte_size = elements * static_cast<int64_t>(sizeof(float));
    return true;
}

std::string LayoutForShape(const std::vector<int64_t>& shape)
{
    if (shape.size() == 4) {
        return "NCHW";
    }
    return "";
}

bool EmitTensorObject(std::ostream& out,
                      const tc::frontend::TensorInfo& tensor,
                      std::string_view role,
                      std::string& out_error)
{
    int64_t byte_size = 0;
    if (!ComputeByteSize(tensor, byte_size, out_error, role)) {
        return false;
    }

    out << "    {\n";
    out << "      \"name\": \"" << JsonEscape(tensor.get_name()) << "\",\n";
    out << "      \"dtype\": \"float32\",\n";
    out << "      \"shape\": " << FormatShape(tensor.get_shape()) << ",\n";
    out << "      \"layout\": \"" << LayoutForShape(tensor.get_shape())
        << "\",\n";
    out << "      \"byte_size\": " << byte_size << '\n';
    out << "    }";
    return true;
}

bool EmitTensorArray(std::ostream& out,
                     const tc::frontend::Graph::TensVecT& tensors,
                     std::string_view role,
                     std::string& out_error)
{
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i]) {
            out_error = "ERROR: metadata " + std::string(role) + " tensor[" +
                        std::to_string(i) + "] is null";
            return false;
        }

        if (!EmitTensorObject(out, *tensors[i], role, out_error)) {
            return false;
        }
        if (i + 1 != tensors.size()) {
            out << ",";
        }
        out << '\n';
    }
    return true;
}

} // namespace

namespace tc::frontend::metadata {

bool BuildMetadataJson(const Graph& graph,
                       std::string& out_json,
                       std::string& out_error)
{
    out_json.clear();
    out_error.clear();

    std::ostringstream out;
    out << "{\n";
    out << "  \"model_entry\": \"tc_model\",\n";
    out << "  \"input_count\": " << graph.get_input_tensors().size() << ",\n";
    out << "  \"inputs\": [\n";
    if (!EmitTensorArray(out, graph.get_input_tensors(), "input", out_error)) {
        out_json.clear();
        return false;
    }
    out << "  ],\n";
    out << "  \"output_count\": " << graph.get_output_tensors().size() << ",\n";
    out << "  \"outputs\": [\n";
    if (!EmitTensorArray(
            out, graph.get_output_tensors(), "output", out_error)) {
        out_json.clear();
        return false;
    }
    out << "  ]\n";
    out << "}\n";

    out_json = out.str();
    return true;
}

} // namespace tc::frontend::metadata
