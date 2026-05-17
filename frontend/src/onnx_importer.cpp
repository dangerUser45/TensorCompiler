#include "onnx_importer.hpp"
#include "frontend_constants.hpp"
#include "graph.hpp"
#include "graph_utils.hpp"
#include "onnx.pb.h"
#include "op_traits.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <google/protobuf/repeated_field.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
bool SetError(std::string& out_error, std::string msg)
{
    out_error = std::move(msg);
    return false;
}

bool LoadOnnxModel(const std::string& path,
                   ::onnx::ModelProto& out_model,
                   std::string& out_error)
{
    out_model.Clear();
    out_error.clear();

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        out_error = "Failed to open file: " + path;
        return false;
    }

    if (!out_model.ParseFromIstream(&input)) {
        out_error = "Failed to parse ONNX protobuf: " + path;
        return false;
    }

    if (!out_model.has_graph()) {
        out_error = "ONNX model has no graph: " + path;
        return false;
    }

    return true;
}
} // namespace

namespace tc::frontend {

namespace detail {

bool ParseDimsFromValueInfo(const ::onnx::ValueInfoProto& vi,
                            std::vector<int64_t>& out_shape,
                            std::string& /*out_error*/)
{
    out_shape.clear();

    if (!vi.has_type() || !vi.type().has_tensor_type() ||
        !vi.type().tensor_type().has_shape()) {
        return true; // it is possible
    }

    const auto& shape = vi.type().tensor_type().shape();
    out_shape.reserve(shape.dim_size());

    for (int i = 0; i < shape.dim_size(); ++i) {
        const auto& dim = shape.dim(i);
        if (dim.has_dim_value()) {
            out_shape.push_back(dim.dim_value());
        } else {
            out_shape.push_back(-1);
        }
    }

    return true;
}

bool ConvertOnnxElemTypeToDataType(int elem_type,
                                   DataT& out_type,
                                   std::string& out_error,
                                   const std::string& tensor_context)
{
    const auto type = static_cast<::onnx::TensorProto_DataType>(elem_type);
    switch (type) {
        case ::onnx::TensorProto_DataType_FLOAT:
            out_type = TypeInfo<float>::type;
            return true;
        case ::onnx::TensorProto_DataType_DOUBLE:
            out_type = TypeInfo<double>::type;
            return true;
        case ::onnx::TensorProto_DataType_INT8:
            out_type = TypeInfo<int8_t>::type;
            return true;
        case ::onnx::TensorProto_DataType_INT16:
            out_type = TypeInfo<int16_t>::type;
            return true;
        case ::onnx::TensorProto_DataType_INT32:
            out_type = TypeInfo<int32_t>::type;
            return true;
        case ::onnx::TensorProto_DataType_INT64:
            out_type = TypeInfo<int64_t>::type;
            return true;
        case ::onnx::TensorProto_DataType_UINT8:
            out_type = TypeInfo<uint8_t>::type;
            return true;
        case ::onnx::TensorProto_DataType_UINT16:
            out_type = TypeInfo<uint16_t>::type;
            return true;
        case ::onnx::TensorProto_DataType_UINT32:
            out_type = TypeInfo<uint32_t>::type;
            return true;
        case ::onnx::TensorProto_DataType_UINT64:
            out_type = TypeInfo<uint64_t>::type;
            return true;
        case ::onnx::TensorProto_DataType_STRING:
            out_type = TypeInfo<std::string>::type;
            return true;
        default:
            break;
    }

    return SetError(out_error,
                    "ERROR: " + tensor_context +
                        " has unsupported elem_type: " +
                        ::onnx::TensorProto_DataType_Name(type));
}

bool ParseTensorMetaFromValueInfo(const ::onnx::ValueInfoProto& vi,
                                  std::vector<int64_t>& out_shape,
                                  DataT& out_data_type,
                                  std::string& out_error,
                                  const std::string& tensor_context)
{
    if (!ParseDimsFromValueInfo(vi, out_shape, out_error)) {
        return false;
    }

    out_data_type = DataT{};
    if (!vi.has_type() || !vi.type().has_tensor_type() ||
        !vi.type().tensor_type().has_elem_type()) {
        return true;
    }

    return ConvertOnnxElemTypeToDataType(vi.type().tensor_type().elem_type(),
                                         out_data_type,
                                         out_error,
                                         tensor_context);
}

template<typename T>
bool ParseRawVector(const std::string& raw,
                    std::vector<T>& out,
                    std::string& out_error,
                    const std::string& init_name)
{
    if (raw.size() % sizeof(T) != 0) {
        return SetError(out_error,
                        "ERROR: initializer '" + init_name +
                            "' has invalid raw_data size");
    }
    out.resize(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
}

template<typename OutT, typename InT>
bool FillNumericFromFieldOrRaw(
    const ::onnx::TensorProto& tp,
    const ::google::protobuf::RepeatedField<InT>& field,
    Initializers& out_init,
    std::string& out_error,
    const char* type_name)
{
    std::vector<OutT> values;
    if (!field.empty()) {
        values.reserve(field.size());
        for (const auto x : field) {
            values.push_back(static_cast<OutT>(x));
        }
    } else if (tp.has_raw_data()) {
        if (!ParseRawVector<OutT>(tp.raw_data(), values, out_error, tp.name()))
            return false;
    } else {
        return SetError(out_error,
                        "ERROR: initializer '" + tp.name() + "' has no " +
                            std::string(type_name) + " data");
    }

    out_init.set_values(std::move(values));
    return true;
}

bool FillInitializerValues(const ::onnx::TensorProto& tp,
                           Initializers& out_init,
                           std::string& out_error)
{
    const auto dt = static_cast<::onnx::TensorProto_DataType>(tp.data_type());

    switch (dt) {
        case ::onnx::TensorProto_DataType_FLOAT:
            return FillNumericFromFieldOrRaw<float>(
                tp, tp.float_data(), out_init, out_error, "FLOAT");

        case ::onnx::TensorProto_DataType_DOUBLE:
            return FillNumericFromFieldOrRaw<double>(
                tp, tp.double_data(), out_init, out_error, "DOUBLE");

        case ::onnx::TensorProto_DataType_INT32:
            return FillNumericFromFieldOrRaw<int32_t>(
                tp, tp.int32_data(), out_init, out_error, "INT32");

        case ::onnx::TensorProto_DataType_INT64:
            return FillNumericFromFieldOrRaw<int64_t>(
                tp, tp.int64_data(), out_init, out_error, "INT64");

        case ::onnx::TensorProto_DataType_UINT8:
            return FillNumericFromFieldOrRaw<uint8_t>(
                tp, tp.int32_data(), out_init, out_error, "UINT8");

        case ::onnx::TensorProto_DataType_INT8:
            return FillNumericFromFieldOrRaw<int8_t>(
                tp, tp.int32_data(), out_init, out_error, "INT8");

        case ::onnx::TensorProto_DataType_UINT16:
            return FillNumericFromFieldOrRaw<uint16_t>(
                tp, tp.int32_data(), out_init, out_error, "UINT16");

        case ::onnx::TensorProto_DataType_INT16:
            return FillNumericFromFieldOrRaw<int16_t>(
                tp, tp.int32_data(), out_init, out_error, "INT16");

        case ::onnx::TensorProto_DataType_UINT32:
            return FillNumericFromFieldOrRaw<uint32_t>(
                tp, tp.uint64_data(), out_init, out_error, "UINT32");

        case ::onnx::TensorProto_DataType_UINT64:
            return FillNumericFromFieldOrRaw<uint64_t>(
                tp, tp.uint64_data(), out_init, out_error, "UINT64");

        case ::onnx::TensorProto_DataType_STRING: {
            std::vector<std::string> v;
            v.reserve(tp.string_data_size());
            for (int i = 0; i < tp.string_data_size(); ++i)
                v.push_back(tp.string_data(i));
            out_init.set_values(std::move(v));
            return true;
        }

        default:
            return SetError(out_error,
                            "ERROR: unsupported initializer type for '" +
                                tp.name() +
                                "': " + ::onnx::TensorProto_DataType_Name(dt));
    }
}

bool BuildInitializers(const ::onnx::GraphProto& g,
                       Graph::InitVecT& out_inits,
                       std::string& out_error)
{
    out_inits.clear();

    out_inits.reserve(g.initializer_size());
    for (int i = 0; i < g.initializer_size(); ++i) {
        const auto& tp = g.initializer(i);

        if (tp.name().empty()) {
            return SetError(out_error, "ERROR: initializer has empty name");
        }
        if (!tp.has_data_type()) {
            return SetError(out_error,
                            "ERROR: initializer '" + tp.name() +
                                "' has no data_type");
        }

        auto init = std::make_unique<Initializers>();
        init->set_name(tp.name());

        std::vector<int64_t> shape;
        shape.reserve(tp.dims_size());
        for (int d = 0; d < tp.dims_size(); ++d)
            shape.push_back(tp.dims(d));
        init->set_shape(std::move(shape));

        if (!FillInitializerValues(tp, *init, out_error))
            return false;

        out_inits.push_back(std::move(init));
    }

    return true;
}

::onnx::AttributeProto_AttributeType ResolveAttributeType(
    const ::onnx::AttributeProto& attr)
{
    if (attr.has_type()) {
        return attr.type();
    }

    if (attr.has_f())
        return ::onnx::AttributeProto_AttributeType_FLOAT;
    if (attr.has_i())
        return ::onnx::AttributeProto_AttributeType_INT;
    if (attr.has_s())
        return ::onnx::AttributeProto_AttributeType_STRING;
    if (attr.floats_size() > 0)
        return ::onnx::AttributeProto_AttributeType_FLOATS;
    if (attr.ints_size() > 0)
        return ::onnx::AttributeProto_AttributeType_INTS;
    if (attr.strings_size() > 0)
        return ::onnx::AttributeProto_AttributeType_STRINGS;

    return ::onnx::AttributeProto_AttributeType_UNDEFINED;
}

bool FillAttributeValues(const ::onnx::AttributeProto& src,
                         Attribute& dst,
                         std::string& out_error,
                         const std::string& attr_context)
{
    const auto type = ResolveAttributeType(src);

    switch (type) {
        case ::onnx::AttributeProto_AttributeType_FLOAT: {
            if (!src.has_f()) {
                return SetError(out_error,
                                "ERROR: " + attr_context +
                                    " FLOAT has no value");
            }
            dst.set_values<float>({ src.f() });
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_INT: {
            if (!src.has_i()) {
                return SetError(out_error,
                                "ERROR: " + attr_context + " INT has no value");
            }
            dst.set_values<int64_t>({ src.i() });
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_STRING: {
            if (!src.has_s()) {
                return SetError(out_error,
                                "ERROR: " + attr_context +
                                    " STRING has no value");
            }
            dst.set_values<std::string>({ src.s() });
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_FLOATS: {
            std::vector<float> values;
            values.reserve(src.floats_size());
            for (int i = 0; i < src.floats_size(); ++i) {
                values.push_back(src.floats(i));
            }
            dst.set_values<float>(std::move(values));
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_INTS: {
            std::vector<int64_t> values;
            values.reserve(src.ints_size());
            for (int i = 0; i < src.ints_size(); ++i) {
                values.push_back(src.ints(i));
            }
            dst.set_values<int64_t>(std::move(values));
            return true;
        }

        case ::onnx::AttributeProto_AttributeType_STRINGS: {
            std::vector<std::string> values;
            values.reserve(src.strings_size());
            for (int i = 0; i < src.strings_size(); ++i) {
                values.push_back(src.strings(i));
            }
            dst.set_values<std::string>(std::move(values));
            return true;
        }

        default:
            return SetError(
                out_error,
                "ERROR: " + attr_context + " unsupported type: " +
                    ::onnx::AttributeProto_AttributeType_Name(type));
    }
}

bool ParseAttribute(const ::onnx::AttributeProto& src,
                    std::unique_ptr<Attribute>& out_attr,
                    std::string& out_error,
                    const std::string& node_context,
                    int attr_index)
{
    const std::string attr_context =
        node_context + ".attribute[" + std::to_string(attr_index) + "]";

    auto attr = std::make_unique<Attribute>();
    attr->set_name(src.name());

    if (!FillAttributeValues(src, *attr, out_error, attr_context)) {
        return false;
    }

    out_attr = std::move(attr);
    return true;
}

bool BuildNodeAttributes(const ::onnx::NodeProto& src,
                         Node::AttrVecT& out_attrs,
                         std::string& out_error,
                         const std::string& node_context)
{
    out_attrs.clear();
    out_attrs.reserve(src.attribute_size());

    for (int i = 0; i < src.attribute_size(); ++i) {
        std::unique_ptr<Attribute> attr;
        if (!ParseAttribute(
                src.attribute(i), attr, out_error, node_context, i)) {
            return false;
        }
        out_attrs.push_back(std::move(attr));
    }

    return true;
}

std::unique_ptr<Attribute> MakeIntAttr(std::string name,
                                       std::vector<int64_t> values)
{
    auto attr = std::make_unique<Attribute>();
    attr->set_name(std::move(name));
    attr->set_values<int64_t>(std::move(values));
    return attr;
}

std::unique_ptr<Attribute> MakeStringAttr(std::string name, std::string value)
{
    auto attr = std::make_unique<Attribute>();
    attr->set_name(std::move(name));
    attr->set_values<std::string>({ std::move(value) });
    return attr;
}

const Attribute* FindAttribute(const Node::AttrVecT& attrs,
                               std::string_view name)
{
    for (const auto& attr : attrs) {
        if (attr && attr->get_name() == name) {
            return attr.get();
        }
    }
    return nullptr;
}

bool ValidateIntAttrLength(const Attribute& attr,
                           std::size_t expected_size,
                           std::string& out_error,
                           const std::string& node_context)
{
    if (attr.get_data_type().id != DataID::INT64) {
        return SetError(out_error,
                        "ERROR: " + node_context + " op 'Conv' attribute '" +
                            attr.get_name() + "' must be INTS");
    }
    if (attr.get_values<int64_t>().size() != expected_size) {
        return SetError(out_error,
                        "ERROR: " + node_context + " op 'Conv' attribute '" +
                            attr.get_name() + "' must have " +
                            std::to_string(expected_size) + " values");
    }
    return true;
}

bool ComputeSameUpperPads(const std::vector<int64_t>* input_shape,
                          const std::vector<int64_t>& kernel_shape,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& dilations,
                          std::vector<int64_t>& out_pads,
                          std::string& out_error,
                          const std::string& node_context)
{
    out_pads.clear();

    const bool unit_stride =
        strides.size() == 2 && strides[0] == 1 && strides[1] == 1;
    const bool unit_dilation =
        dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1;

    if (unit_stride && unit_dilation) {
        // For stride=1 and dilation=1, SAME_UPPER pads are independent of
        // input spatial dims: total_pad = kernel - 1. SAME_UPPER places the
        // extra pad at the end (bottom/right) when total is odd.
        const int64_t pad_h = kernel_shape[0] > 0 ? kernel_shape[0] - 1 : 0;
        const int64_t pad_w = kernel_shape[1] > 0 ? kernel_shape[1] - 1 : 0;
        out_pads = {
            pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2
        };
        return true;
    }

    if (input_shape == nullptr || input_shape->size() != 4) {
        return SetError(out_error,
                        "ERROR: " + node_context +
                            " op 'Conv' auto_pad=SAME_UPPER requires rank-4 "
                            "static input shape for non-unit stride/dilation");
    }
    if ((*input_shape)[2] < 0 || (*input_shape)[3] < 0) {
        return SetError(out_error,
                        "ERROR: " + node_context +
                            " op 'Conv' auto_pad=SAME_UPPER requires static "
                            "spatial dimensions");
    }

    auto compute = [](int64_t in,
                      int64_t kernel,
                      int64_t stride,
                      int64_t dilation) -> int64_t {
        const int64_t out = (in + stride - 1) / stride; // ceil(in/stride)
        const int64_t effective_kernel = (kernel - 1) * dilation + 1;
        const int64_t total = (out - 1) * stride + effective_kernel - in;
        return total > 0 ? total : 0;
    };
    const int64_t total_h =
        compute((*input_shape)[2], kernel_shape[0], strides[0], dilations[0]);
    const int64_t total_w =
        compute((*input_shape)[3], kernel_shape[1], strides[1], dilations[1]);
    out_pads = {
        total_h / 2, total_w / 2, total_h - total_h / 2, total_w - total_w / 2
    };
    return true;
}

bool NormalizeConvAttributes(Node::AttrVecT& attrs,
                             std::string& out_error,
                             const std::string& node_context,
                             const std::vector<int64_t>* inferred_kernel_shape,
                             const std::vector<int64_t>* input_shape)
{
    std::unordered_set<std::string> seen_names;
    seen_names.reserve(attrs.size());

    bool has_kernel_shape = false;
    bool has_strides = false;
    bool has_pads = false;
    bool has_dilations = false;
    bool has_group = false;
    bool has_auto_pad = false;

    for (std::size_t i = 0; i < attrs.size(); ++i) {
        if (!attrs[i]) {
            return SetError(out_error,
                            "ERROR: " + node_context + ".attribute[" +
                                std::to_string(i) + "] is null");
        }
        const std::string& attr_name = attrs[i]->get_name();
        if (attr_name.empty()) {
            return SetError(out_error,
                            "ERROR: " + node_context + ".attribute[" +
                                std::to_string(i) + "] has empty name");
        }
        if (!seen_names.insert(attr_name).second) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " has duplicate attribute '" + attr_name + "'");
        }

        if (attr_name == "kernel_shape") {
            has_kernel_shape = true;
            if (!ValidateIntAttrLength(*attrs[i], 2, out_error, node_context)) {
                return false;
            }
            continue;
        }
        if (attr_name == "strides") {
            has_strides = true;
            if (!ValidateIntAttrLength(*attrs[i], 2, out_error, node_context)) {
                return false;
            }
            continue;
        }
        if (attr_name == "pads") {
            has_pads = true;
            if (!ValidateIntAttrLength(*attrs[i], 4, out_error, node_context)) {
                return false;
            }
            continue;
        }
        if (attr_name == "dilations") {
            has_dilations = true;
            if (!ValidateIntAttrLength(*attrs[i], 2, out_error, node_context)) {
                return false;
            }
            continue;
        }
        if (attr_name == "group") {
            has_group = true;
            if (attrs[i]->get_data_type().id != DataID::INT64 ||
                attrs[i]->get_values<int64_t>().size() != 1) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'Conv' attribute 'group' must be INT");
            }
            continue;
        }
        if (attr_name == "auto_pad") {
            has_auto_pad = true;
            if (attrs[i]->get_data_type().id != DataID::STRING ||
                attrs[i]->get_values<std::string>().size() != 1) {
                return SetError(
                    out_error,
                    "ERROR: " + node_context +
                        " op 'Conv' attribute 'auto_pad' must be STRING");
            }
            const auto& value = attrs[i]->get_values<std::string>().front();
            if (value != std::string(tc::frontend::kAutoPadNotSet) &&
                value != std::string(tc::frontend::kAutoPadSameUpper)) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'Conv' supports only "
                                    "auto_pad in {NOTSET, "
                                    "SAME_UPPER}, got " +
                                    value);
            }
            continue;
        }

        return SetError(out_error,
                        "ERROR: " + node_context +
                            " op 'Conv' has unsupported attribute '" +
                            attr_name + "'");
    }

    if (!has_kernel_shape) {
        if (inferred_kernel_shape == nullptr ||
            inferred_kernel_shape->size() != 2) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Conv' cannot infer kernel_shape");
        }
        attrs.push_back(MakeIntAttr("kernel_shape", *inferred_kernel_shape));
    }
    if (!has_strides) {
        attrs.push_back(MakeIntAttr("strides", { 1, 1 }));
    }
    if (!has_dilations) {
        attrs.push_back(MakeIntAttr("dilations", { 1, 1 }));
    }
    if (!has_group) {
        attrs.push_back(MakeIntAttr("group", { 1 }));
    }
    if (!has_auto_pad) {
        attrs.push_back(MakeStringAttr(
            "auto_pad", std::string(tc::frontend::kAutoPadNotSet)));
    }

    // Normalize auto_pad=SAME_UPPER to explicit pads. After normalization the
    // canonical IR form always carries auto_pad="NOTSET" with explicit pads.
    Attribute* auto_pad_attr = nullptr;
    Attribute* kernel_attr = nullptr;
    Attribute* strides_attr = nullptr;
    Attribute* dilations_attr = nullptr;
    for (auto& attr : attrs) {
        if (!attr) {
            continue;
        }
        if (attr->get_name() == "auto_pad") {
            auto_pad_attr = attr.get();
        } else if (attr->get_name() == "kernel_shape") {
            kernel_attr = attr.get();
        } else if (attr->get_name() == "strides") {
            strides_attr = attr.get();
        } else if (attr->get_name() == "dilations") {
            dilations_attr = attr.get();
        }
    }

    const std::string auto_pad_value =
        auto_pad_attr ? auto_pad_attr->get_values<std::string>().front()
                      : std::string(tc::frontend::kAutoPadNotSet);

    if (auto_pad_value == std::string(tc::frontend::kAutoPadSameUpper)) {
        if (has_pads) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Conv' must not set 'pads' when "
                                "auto_pad='SAME_UPPER'");
        }
        if (!kernel_attr || !strides_attr || !dilations_attr) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Conv' SAME_UPPER normalization missing "
                                "kernel/strides/dilations");
        }
        std::vector<int64_t> normalized_pads;
        if (!ComputeSameUpperPads(input_shape,
                                  kernel_attr->get_values<int64_t>(),
                                  strides_attr->get_values<int64_t>(),
                                  dilations_attr->get_values<int64_t>(),
                                  normalized_pads,
                                  out_error,
                                  node_context)) {
            return false;
        }
        attrs.push_back(MakeIntAttr("pads", std::move(normalized_pads)));
        auto_pad_attr->set_values<std::string>(
            { std::string(tc::frontend::kAutoPadNotSet) });
    } else if (!has_pads) {
        attrs.push_back(MakeIntAttr("pads", { 0, 0, 0, 0 }));
    }

    return true;
}

bool ValidateMaxPoolIntAttrLength(const Attribute& attr,
                                  std::size_t expected_size,
                                  std::string& out_error,
                                  const std::string& node_context)
{
    if (attr.get_data_type().id != DataID::INT64) {
        return SetError(out_error,
                        "ERROR: " + node_context + " op 'MaxPool' attribute '" +
                            attr.get_name() + "' must be INTS");
    }
    if (attr.get_values<int64_t>().size() != expected_size) {
        return SetError(out_error,
                        "ERROR: " + node_context + " op 'MaxPool' attribute '" +
                            attr.get_name() + "' must have " +
                            std::to_string(expected_size) + " values");
    }
    return true;
}

bool NormalizeMaxPoolAttributes(Node::AttrVecT& attrs,
                                std::string& out_error,
                                const std::string& node_context)
{
    std::unordered_set<std::string> seen_names;
    seen_names.reserve(attrs.size());

    bool has_kernel_shape = false;
    bool has_strides = false;
    bool has_pads = false;
    bool has_auto_pad = false;

    for (std::size_t i = 0; i < attrs.size(); ++i) {
        if (!attrs[i]) {
            return SetError(out_error,
                            "ERROR: " + node_context + ".attribute[" +
                                std::to_string(i) + "] is null");
        }
        const std::string& attr_name = attrs[i]->get_name();
        if (attr_name.empty()) {
            return SetError(out_error,
                            "ERROR: " + node_context + ".attribute[" +
                                std::to_string(i) + "] has empty name");
        }
        if (!seen_names.insert(attr_name).second) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " has duplicate attribute '" + attr_name + "'");
        }

        if (attr_name == "kernel_shape") {
            has_kernel_shape = true;
            if (!ValidateMaxPoolIntAttrLength(
                    *attrs[i], 2, out_error, node_context)) {
                return false;
            }
            const auto& values = attrs[i]->get_values<int64_t>();
            if (values[0] <= 0 || values[1] <= 0) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'MaxPool' kernel_shape must be "
                                    "positive");
            }
            continue;
        }
        if (attr_name == "strides") {
            has_strides = true;
            if (!ValidateMaxPoolIntAttrLength(
                    *attrs[i], 2, out_error, node_context)) {
                return false;
            }
            const auto& values = attrs[i]->get_values<int64_t>();
            if (values[0] <= 0 || values[1] <= 0) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'MaxPool' strides must be positive");
            }
            continue;
        }
        if (attr_name == "pads") {
            has_pads = true;
            if (!ValidateMaxPoolIntAttrLength(
                    *attrs[i], 4, out_error, node_context)) {
                return false;
            }
            for (const int64_t pad : attrs[i]->get_values<int64_t>()) {
                if (pad < 0) {
                    return SetError(out_error,
                                    "ERROR: " + node_context +
                                        " op 'MaxPool' pads must be "
                                        "non-negative");
                }
            }
            continue;
        }
        if (attr_name == "auto_pad") {
            has_auto_pad = true;
            if (attrs[i]->get_data_type().id != DataID::STRING ||
                attrs[i]->get_values<std::string>().size() != 1) {
                return SetError(
                    out_error,
                    "ERROR: " + node_context +
                        " op 'MaxPool' attribute 'auto_pad' must be STRING");
            }
            const auto& value = attrs[i]->get_values<std::string>().front();
            if (value != std::string(tc::frontend::kAutoPadNotSet)) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'MaxPool' supports only "
                                    "auto_pad=NOTSET");
            }
            continue;
        }
        if (attr_name == "dilations") {
            if (!ValidateMaxPoolIntAttrLength(
                    *attrs[i], 2, out_error, node_context)) {
                return false;
            }
            const auto& values = attrs[i]->get_values<int64_t>();
            if (values[0] != 1 || values[1] != 1) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'MaxPool' supports only "
                                    "dilations=[1,1]");
            }
            continue;
        }
        if (attr_name == "ceil_mode") {
            if (attrs[i]->get_data_type().id != DataID::INT64 ||
                attrs[i]->get_values<int64_t>().size() != 1) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'MaxPool' attribute 'ceil_mode' must "
                                    "be INT");
            }
            if (attrs[i]->get_values<int64_t>().front() != 0) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'MaxPool' supports only ceil_mode=0");
            }
            continue;
        }

        return SetError(out_error,
                        "ERROR: " + node_context +
                            " op 'MaxPool' has unsupported attribute '" +
                            attr_name + "'");
    }

    if (!has_kernel_shape) {
        return SetError(out_error,
                        "ERROR: " + node_context +
                            " op 'MaxPool' missing required attribute "
                            "'kernel_shape'");
    }
    if (!has_strides) {
        attrs.push_back(MakeIntAttr("strides", { 1, 1 }));
    }
    if (!has_pads) {
        attrs.push_back(MakeIntAttr("pads", { 0, 0, 0, 0 }));
    }
    if (!has_auto_pad) {
        attrs.push_back(MakeStringAttr(
            "auto_pad", std::string(tc::frontend::kAutoPadNotSet)));
    }
    return true;
}

bool NormalizeNodeAttributes(OpKind op_kind,
                             Node::AttrVecT& attrs,
                             std::string& out_error,
                             const std::string& node_context,
                             const std::vector<int64_t>* inferred_kernel_shape,
                             const std::vector<int64_t>* input_shape)
{
    if (op_kind == OpKind::kConv) {
        return NormalizeConvAttributes(
            attrs, out_error, node_context, inferred_kernel_shape, input_shape);
    }
    if (op_kind == OpKind::kMaxPool) {
        return NormalizeMaxPoolAttributes(attrs, out_error, node_context);
    }

    std::unordered_set<std::string> seen_names;
    seen_names.reserve(attrs.size());

    for (std::size_t i = 0; i < attrs.size(); ++i) {
        if (!attrs[i]) {
            return SetError(out_error,
                            "ERROR: " + node_context + ".attribute[" +
                                std::to_string(i) + "] is null");
        }

        const std::string& attr_name = attrs[i]->get_name();
        if (attr_name.empty()) {
            return SetError(out_error,
                            "ERROR: " + node_context + ".attribute[" +
                                std::to_string(i) + "] has empty name");
        }

        if (!seen_names.insert(attr_name).second) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " has duplicate attribute '" + attr_name + "'");
        }

        if (op_kind == OpKind::kUnknown) {
            continue;
        }

        if (!SupportsAttributes(op_kind)) {
            return SetError(out_error,
                            "ERROR: " + node_context + " op '" +
                                std::string(ToString(op_kind)) +
                                "' does not support attribute '" + attr_name +
                                "'");
        }

        if (op_kind == OpKind::kTranspose) {
            if (attr_name != "perm") {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " op 'Transpose' has unsupported "
                                    "attribute '" +
                                    attr_name + "'");
            }
            if (attrs[i]->get_data_type().id != DataID::INT64) {
                return SetError(
                    out_error,
                    "ERROR: " + node_context +
                        " op 'Transpose' attribute 'perm' must be INTS");
            }
            continue;
        }

        return SetError(out_error,
                        "ERROR: " + node_context + " op '" +
                            std::string(ToString(op_kind)) +
                            "' has unsupported attribute '" + attr_name + "'");
    }

    return true;
}

bool ParseNode(const ::onnx::NodeProto& src,
               std::vector<std::unique_ptr<Node>>& out_nodes,
               std::string& out_error,
               int node_index,
               const std::unordered_map<std::string, std::vector<int64_t>>&
                   initializer_shapes,
               const std::unordered_map<std::string, std::vector<int64_t>>&
                   value_info_shapes)
{
    const std::string node_context = tc::frontend::BuildNodeContext(
        src.name(), static_cast<std::size_t>(node_index));
    const std::string& op_type = src.op_type();
    if (op_type.empty()) {
        return SetError(out_error, "ERROR: " + node_context + " has empty op");
    }

    if (op_type == "Gemm") {
        if (src.input_size() < 2 || src.input_size() > 3) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Gemm' expects 2 or 3 inputs");
        }
        if (src.output_size() != 1) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Gemm' expects 1 output");
        }

        float alpha = 1.0f;
        float beta = 1.0f;
        int64_t trans_a = 0;
        int64_t trans_b = 0;
        int64_t broadcast = 0;
        std::unordered_set<std::string> seen_names;
        seen_names.reserve(src.attribute_size());

        for (int i = 0; i < src.attribute_size(); ++i) {
            const auto& attr = src.attribute(i);
            const std::string& attr_name = attr.name();
            const std::string attr_context =
                node_context + ".attribute[" + std::to_string(i) + "]";
            if (attr_name.empty()) {
                return SetError(out_error,
                                "ERROR: " + attr_context + " has empty name");
            }
            if (!seen_names.insert(attr_name).second) {
                return SetError(out_error,
                                "ERROR: " + node_context +
                                    " has duplicate attribute '" + attr_name +
                                    "'");
            }

            const auto attr_type = ResolveAttributeType(attr);
            if (attr_name == "alpha" || attr_name == "beta") {
                if (attr_type != ::onnx::AttributeProto_AttributeType_FLOAT ||
                    !attr.has_f()) {
                    return SetError(out_error,
                                    "ERROR: " + node_context +
                                        " op 'Gemm' "
                                        "attribute '" +
                                        attr_name + "' must be FLOAT");
                }
                if (attr_name == "alpha") {
                    alpha = attr.f();
                } else {
                    beta = attr.f();
                }
                continue;
            }

            if (attr_name == "transA" || attr_name == "transB" ||
                attr_name == "broadcast") {
                if (attr_type != ::onnx::AttributeProto_AttributeType_INT ||
                    !attr.has_i()) {
                    return SetError(out_error,
                                    "ERROR: " + node_context +
                                        " op 'Gemm' "
                                        "attribute '" +
                                        attr_name + "' must be INT");
                }
                if (attr_name == "transA") {
                    trans_a = attr.i();
                } else if (attr_name == "transB") {
                    trans_b = attr.i();
                } else {
                    broadcast = attr.i();
                }
                continue;
            }

            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Gemm' has "
                                "unsupported attribute '" +
                                attr_name + "'");
        }

        constexpr float kEpsilon = 1.0e-6F;
        if (std::fabs(alpha - 1.0F) > kEpsilon) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Gemm' supports only alpha=1");
        }
        if (std::fabs(beta - 1.0F) > kEpsilon) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Gemm' supports only beta=1");
        }
        if (trans_a != 0 || trans_b != 0) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Gemm' supports only transA=0 and "
                                "transB=0");
        }
        if (broadcast != 0 && broadcast != 1) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Gemm' broadcast must be 0 or 1");
        }

        const bool has_bias = src.input_size() >= 3 && !src.input(2).empty();
        const std::string& final_output = src.output(0);
        std::string matmul_output = final_output;
        if (has_bias) {
            matmul_output +=
                std::string(tc::frontend::kGemmMatMulOutputSuffix) +
                std::to_string(node_index);
        }

        auto matmul_node = std::make_unique<Node>();
        matmul_node->set_name_node(
            src.name().empty()
                ? std::string(tc::frontend::kGemmMatMulNodePrefix) +
                      std::to_string(node_index)
                : src.name() +
                      std::string(tc::frontend::kSyntheticMatMulSuffix));
        matmul_node->set_name_op("MatMul");
        matmul_node->set_op_kind(OpKind::kMatMul);
        matmul_node->set_inputs({ src.input(0), src.input(1) });
        matmul_node->set_outputs({ matmul_output });
        out_nodes.push_back(std::move(matmul_node));

        if (has_bias) {
            auto add_node = std::make_unique<Node>();
            add_node->set_name_node(
                src.name().empty()
                    ? std::string(tc::frontend::kGemmAddNodePrefix) +
                          std::to_string(node_index)
                    : src.name() +
                          std::string(tc::frontend::kSyntheticAddSuffix));
            add_node->set_name_op("Add");
            add_node->set_op_kind(OpKind::kAdd);
            add_node->set_inputs({ matmul_output, src.input(2) });
            add_node->set_outputs({ final_output });
            out_nodes.push_back(std::move(add_node));
        }

        return true;
    }

    const OpKind op_kind = OpKindFromString(op_type);
    if (op_kind == OpKind::kUnknown) {
        return SetError(out_error,
                        "ERROR: " + node_context +
                            " unsupported operator: " + op_type);
    }

    auto node = std::make_unique<Node>();
    node->set_name_node(src.name());
    node->set_name_op(op_type);
    node->set_op_kind(op_kind);

    std::vector<std::string> inputs;
    inputs.reserve(src.input_size());
    for (int i = 0; i < src.input_size(); ++i) {
        inputs.push_back(src.input(i));
    }
    node->set_inputs(std::move(inputs));

    std::vector<std::string> outputs;
    outputs.reserve(src.output_size());
    for (int i = 0; i < src.output_size(); ++i) {
        outputs.push_back(src.output(i));
    }
    node->set_outputs(std::move(outputs));

    if (op_kind == OpKind::kConv) {
        if (src.input_size() < 2 || src.input_size() > 3) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Conv' expects 2 or 3 inputs");
        }
        if (src.output_size() != 1) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Conv' expects 1 output");
        }
    }
    if (op_kind == OpKind::kReshape) {
        if (src.input_size() != 2) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Reshape' expects 2 inputs");
        }
        if (src.output_size() != 1) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'Reshape' expects 1 output");
        }
    }
    if (op_kind == OpKind::kMaxPool) {
        if (src.input_size() != 1) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'MaxPool' expects 1 input");
        }
        if (src.output_size() != 1) {
            return SetError(out_error,
                            "ERROR: " + node_context +
                                " op 'MaxPool' expects 1 output");
        }
    }

    Node::AttrVecT attrs;
    if (!BuildNodeAttributes(src, attrs, out_error, node_context)) {
        return false;
    }
    std::vector<int64_t> inferred_kernel_shape;
    const std::vector<int64_t>* inferred_kernel_shape_ptr = nullptr;
    const std::vector<int64_t>* input_shape_ptr = nullptr;
    if (op_kind == OpKind::kConv && src.input_size() >= 2) {
        const auto weight_it = initializer_shapes.find(src.input(1));
        if (weight_it != initializer_shapes.end() &&
            weight_it->second.size() == 4) {
            inferred_kernel_shape = { weight_it->second[2],
                                      weight_it->second[3] };
            inferred_kernel_shape_ptr = &inferred_kernel_shape;
        }
        const auto input_it = value_info_shapes.find(src.input(0));
        if (input_it != value_info_shapes.end()) {
            input_shape_ptr = &input_it->second;
        }
    }
    if (!NormalizeNodeAttributes(op_kind,
                                 attrs,
                                 out_error,
                                 node_context,
                                 inferred_kernel_shape_ptr,
                                 input_shape_ptr)) {
        return false;
    }
    node->set_attr(std::move(attrs));

    if (op_kind == OpKind::kConv && src.input_size() == 3) {
        const std::string final_output = src.output(0);
        const std::string conv_output =
            final_output + std::string(tc::frontend::kConvOutputSuffix) +
            std::to_string(node_index);
        node->set_inputs({ src.input(0), src.input(1) });
        node->set_outputs({ conv_output });
        out_nodes.push_back(std::move(node));

        auto add_node = std::make_unique<Node>();
        add_node->set_name_node(
            src.name().empty()
                ? std::string(tc::frontend::kConvAddNodePrefix) +
                      std::to_string(node_index)
                : src.name() + std::string(tc::frontend::kSyntheticAddSuffix));
        add_node->set_name_op("Add");
        add_node->set_op_kind(OpKind::kAdd);
        add_node->set_inputs({ conv_output, src.input(2) });
        add_node->set_outputs({ final_output });
        out_nodes.push_back(std::move(add_node));
        return true;
    }

    out_nodes.push_back(std::move(node));
    return true;
}

bool BuildNodes(const ::onnx::GraphProto& g,
                Graph::NodeVecT& out_nodes,
                std::string& out_error)
{
    out_nodes.clear();

    std::unordered_map<std::string, std::vector<int64_t>> initializer_shapes;
    initializer_shapes.reserve(g.initializer_size());
    for (int i = 0; i < g.initializer_size(); ++i) {
        const auto& init = g.initializer(i);
        std::vector<int64_t> shape;
        shape.reserve(init.dims_size());
        for (int d = 0; d < init.dims_size(); ++d) {
            shape.push_back(init.dims(d));
        }
        initializer_shapes[init.name()] = std::move(shape);
    }

    // Collect static shapes for graph inputs, intermediate value_info, and
    // outputs so that auto_pad=SAME_UPPER normalization can read input shapes
    // for non-unit stride/dilation cases.
    std::unordered_map<std::string, std::vector<int64_t>> value_info_shapes =
        initializer_shapes;
    auto collect_value_info =
        [&value_info_shapes](const ::onnx::ValueInfoProto& vi) {
            if (vi.name().empty()) {
                return;
            }
            std::vector<int64_t> shape;
            std::string ignored_error;
            ParseDimsFromValueInfo(vi, shape, ignored_error);
            value_info_shapes[vi.name()] = std::move(shape);
        };
    for (int i = 0; i < g.input_size(); ++i) {
        collect_value_info(g.input(i));
    }
    for (int i = 0; i < g.value_info_size(); ++i) {
        collect_value_info(g.value_info(i));
    }
    for (int i = 0; i < g.output_size(); ++i) {
        collect_value_info(g.output(i));
    }

    out_nodes.reserve(g.node_size() * 2);
    for (int i = 0; i < g.node_size(); ++i) {
        if (!ParseNode(g.node(i),
                       out_nodes,
                       out_error,
                       i,
                       initializer_shapes,
                       value_info_shapes)) {
            return false;
        }
    }

    return true;
}

bool BuildInputTensors(const ::onnx::GraphProto& g,
                       Graph::TensVecT& out_inputs,
                       std::string& out_error)
{
    out_inputs.clear();
    out_error.clear();

    std::unordered_set<std::string> init_names;
    init_names.reserve(g.initializer_size());
    for (int i = 0; i < g.initializer_size(); ++i) {
        init_names.insert(g.initializer(i).name());
    }

    out_inputs.reserve(g.input_size());
    for (int i = 0; i < g.input_size(); ++i) {
        const auto& in = g.input(i);
        if (in.name().empty()) {
            out_error = "Input '" + in.name() + "' has empty name";
            return false;
        }

        if (init_names.find(in.name()) != init_names.end()) {
            continue;
        }

        std::vector<int64_t> shape;
        DataT data_type;
        if (!ParseTensorMetaFromValueInfo(
                in, shape, data_type, out_error, "input '" + in.name() + "'")) {
            return false;
        }

        auto t = std::make_unique<TensorInfo>();
        t->set_name(in.name());
        t->set_shape(shape);
        t->set_data_type(data_type);
        out_inputs.push_back(std::move(t));
    }
    return true;
}

bool BuildOutputTensors(const ::onnx::GraphProto& g,
                        Graph::TensVecT& out_outputs,
                        std::string& out_error)
{
    out_outputs.clear();
    out_error.clear();

    out_outputs.reserve(g.output_size());

    for (int i = 0; i < g.output_size(); ++i) {
        const auto& out = g.output(i);
        if (out.name().empty()) {
            out_error = "Output has empty name";
            return false;
        }

        std::vector<int64_t> shape;
        DataT data_type;
        if (!ParseTensorMetaFromValueInfo(out,
                                          shape,
                                          data_type,
                                          out_error,
                                          "output '" + out.name() + "'")) {
            return false;
        }

        auto t = std::make_unique<TensorInfo>();
        t->set_name(out.name());
        t->set_shape(shape);
        t->set_data_type(data_type);
        out_outputs.push_back(std::move(t));
    }

    return true;
}

bool BuildGraph(const ::onnx::GraphProto& g,
                Graph& out_graph,
                std::string& out_error)
{
    out_graph.set_name(g.name());

    Graph::TensVecT input_tensors;
    if (!BuildInputTensors(g, input_tensors, out_error)) {
        return false;
    }

    out_graph.set_input_tensors(std::move(input_tensors));

    Graph::TensVecT output_tensors;
    if (!BuildOutputTensors(g, output_tensors, out_error)) {
        return false;
    }

    out_graph.set_output_tensors(std::move(output_tensors));

    Graph::InitVecT initializers;
    if (!BuildInitializers(g, initializers, out_error)) {
        return false;
    }

    out_graph.set_inits(std::move(initializers));

    Graph::NodeVecT nodes;
    if (!BuildNodes(g, nodes, out_error)) {
        return false;
    }

    out_graph.set_nodes(std::move(nodes));

    return true;
}

} // namespace detail

bool onnx::ImportOnnxToGraph(const std::string& path,
                             Graph& out_graph,
                             std::string& out_error)
{
    ::onnx::ModelProto input_model;
    if (!LoadOnnxModel(path, input_model, out_error)) {
        return false;
    }

    return detail::BuildGraph(input_model.graph(), out_graph, out_error);
}
} // namespace tc::frontend
