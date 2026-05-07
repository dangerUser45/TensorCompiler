#include "graph_verifier.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

bool IsMvpNumericDataId(tc::frontend::DataID id) noexcept
{
    switch (id) {
        case tc::frontend::DataID::INT8:
        case tc::frontend::DataID::INT16:
        case tc::frontend::DataID::INT32:
        case tc::frontend::DataID::INT64:
        case tc::frontend::DataID::UNSIGNED_INT8:
        case tc::frontend::DataID::UNSIGNED_INT16:
        case tc::frontend::DataID::UNSIGNED_INT32:
        case tc::frontend::DataID::UNSIGNED_INT64:
        case tc::frontend::DataID::FLOAT:
        case tc::frontend::DataID::DOUBLE:
            return true;
        case tc::frontend::DataID::COMPLEX64:
        case tc::frontend::DataID::COMPLEX128:
        case tc::frontend::DataID::STRING:
        case tc::frontend::DataID::UNDEFINED:
            return false;
    }
    return false;
}

std::string DtypeName(const tc::frontend::DataT& type)
{
    if (!type.data_type_str.empty()) {
        return type.data_type_str;
    }
    return "UNDEFINED";
}

std::string BuildNodeContext(const tc::frontend::Node& node,
                             std::size_t node_index)
{
    if (!node.get_name_node().empty()) {
        return "node[" + std::to_string(node_index) + "]('" +
               node.get_name_node() + "')";
    }
    return "node[" + std::to_string(node_index) + "]";
}

bool ShapesMatchForMvp(const std::vector<int64_t>& lhs,
                       const std::vector<int64_t>& rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] == -1 || rhs[i] == -1) {
            continue;
        }
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

bool BroadcastDimsCompatible(int64_t lhs_dim, int64_t rhs_dim) noexcept
{
    if (lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1) {
        return true;
    }
    if (lhs_dim == -1 || rhs_dim == -1) {
        return true;
    }
    return false;
}

int64_t ResolveBroadcastDim(int64_t lhs_dim, int64_t rhs_dim) noexcept
{
    if (lhs_dim == rhs_dim) {
        return lhs_dim;
    }
    if (lhs_dim == 1) {
        return rhs_dim == -1 ? -1 : rhs_dim;
    }
    if (rhs_dim == 1) {
        return lhs_dim == -1 ? -1 : lhs_dim;
    }
    if (lhs_dim == -1 && rhs_dim == -1) {
        return -1;
    }
    if (lhs_dim == -1) {
        return rhs_dim;
    }
    if (rhs_dim == -1) {
        return lhs_dim;
    }
    return -1;
}

bool ComputeBroadcastShape(const std::vector<int64_t>& lhs_shape,
                           const std::vector<int64_t>& rhs_shape,
                           std::vector<int64_t>& out_shape)
{
    const std::size_t rank = std::max(lhs_shape.size(), rhs_shape.size());
    out_shape.assign(rank, -1);

    for (std::size_t axis = 0; axis < rank; ++axis) {
        const std::size_t lhs_offset = rank - lhs_shape.size();
        const std::size_t rhs_offset = rank - rhs_shape.size();
        const int64_t lhs_dim =
            axis < lhs_offset ? 1 : lhs_shape[axis - lhs_offset];
        const int64_t rhs_dim =
            axis < rhs_offset ? 1 : rhs_shape[axis - rhs_offset];

        if (!BroadcastDimsCompatible(lhs_dim, rhs_dim)) {
            return false;
        }
        out_shape[axis] = ResolveBroadcastDim(lhs_dim, rhs_dim);
    }

    return true;
}

class ExecutionVerifierVisitor final
{
public:
    explicit ExecutionVerifierVisitor(tc::frontend::verify::Report& report)
      : report_(report)
    {
    }

    void VisitGraph(const tc::frontend::Graph& graph)
    {
        if (graph.get_name().empty()) {
            report_.add_warning("WARN: graph has no name");
        }

        if (graph.get_output_tensors().empty()) {
            report_.add_error("ERROR: graph has no output tensors");
        }

        VisitOutputTensors(graph.get_output_tensors());
        VisitInitializers(graph.get_inits());

        std::unordered_map<std::string, tc::frontend::DataT> tensor_dtypes;
        std::unordered_map<std::string, std::vector<int64_t>> tensor_shapes;
        BuildTensorDtypeTable(graph, tensor_dtypes);
        BuildTensorShapeTable(graph, tensor_shapes);
        VisitNodes(graph.get_nodes(), tensor_dtypes, tensor_shapes);
    }

private:
    void AddArityError(const std::string& node_context,
                       tc::frontend::OpKind op_kind,
                       const char* field_name,
                       std::size_t expected,
                       std::size_t got)
    {
        report_.add_error("ERROR: " + node_context + " op '" +
                          std::string(tc::frontend::ToString(op_kind)) +
                          "' expects " + std::to_string(expected) + " " +
                          field_name + ", got " + std::to_string(got));
    }

    void ValidateNodeSemantics(
        const tc::frontend::Node& node,
        const std::string& node_context,
        std::unordered_map<std::string, tc::frontend::DataT>& tensor_dtypes,
        std::unordered_map<std::string, std::vector<int64_t>>& tensor_shapes)
    {
        const auto op_kind = node.get_op_kind();
        const std::size_t input_count = node.get_inputs().size();
        const std::size_t output_count = node.get_outputs().size();

        if (op_kind == tc::frontend::OpKind::kUnknown) {
            report_.add_error("ERROR: " + node_context +
                              " has unknown op kind");
            return;
        }

        switch (op_kind) {
            case tc::frontend::OpKind::kRelu:
                if (input_count != 1) {
                    AddArityError(
                        node_context, op_kind, "input(s)", 1, input_count);
                }
                if (output_count != 1) {
                    AddArityError(
                        node_context, op_kind, "output(s)", 1, output_count);
                }
                break;
            case tc::frontend::OpKind::kAdd:
            case tc::frontend::OpKind::kMul:
            case tc::frontend::OpKind::kMatMul:
                if (input_count != 2) {
                    AddArityError(
                        node_context, op_kind, "input(s)", 2, input_count);
                }
                if (output_count != 1) {
                    AddArityError(
                        node_context, op_kind, "output(s)", 1, output_count);
                }
                break;
            case tc::frontend::OpKind::kTranspose:
                if (input_count != 1) {
                    AddArityError(
                        node_context, op_kind, "input(s)", 1, input_count);
                }
                if (output_count != 1) {
                    AddArityError(
                        node_context, op_kind, "output(s)", 1, output_count);
                }
                break;
            case tc::frontend::OpKind::kUnknown:
                break;
        }

        ValidateNodeAttributes(node, node_context);
        ValidateNodeDtypes(node, node_context, tensor_dtypes);
        ValidateNodeShapes(node, node_context, tensor_shapes);
    }

    const tc::frontend::DataT* FindTensorDtype(
        const std::unordered_map<std::string, tc::frontend::DataT>&
            tensor_dtypes,
        const std::string& tensor_name) const
    {
        const auto it = tensor_dtypes.find(tensor_name);
        if (it == tensor_dtypes.end()) {
            return nullptr;
        }
        return &it->second;
    }

    bool ResolveInputDtype(
        const tc::frontend::Node& node,
        const std::string& node_context,
        std::size_t input_index,
        const std::unordered_map<std::string, tc::frontend::DataT>&
            tensor_dtypes,
        tc::frontend::DataT& out_dtype)
    {
        if (input_index >= node.get_inputs().size()) {
            return false;
        }

        const std::string& input_name = node.get_inputs()[input_index];
        if (input_name.empty()) {
            return false;
        }

        const tc::frontend::DataT* dtype =
            FindTensorDtype(tensor_dtypes, input_name);
        if (!dtype || dtype->id == tc::frontend::DataID::UNDEFINED) {
            report_.add_error("ERROR: " + node_context + " input[" +
                              std::to_string(input_index) + "] '" + input_name +
                              "' has undefined dtype");
            return false;
        }

        out_dtype = *dtype;
        return true;
    }

    void PropagateNodeOutputDtype(
        const tc::frontend::Node& node,
        const std::string& node_context,
        const tc::frontend::DataT& inferred_dtype,
        std::unordered_map<std::string, tc::frontend::DataT>& tensor_dtypes)
    {
        for (const std::string& output_name : node.get_outputs()) {
            if (output_name.empty()) {
                continue;
            }

            auto it = tensor_dtypes.find(output_name);
            if (it == tensor_dtypes.end()) {
                tensor_dtypes.emplace(output_name, inferred_dtype);
                continue;
            }

            if (it->second.id != tc::frontend::DataID::UNDEFINED &&
                it->second.id != inferred_dtype.id) {
                report_.add_error("ERROR: " + node_context +
                                  " output dtype mismatch for '" + output_name +
                                  "': inferred " + DtypeName(inferred_dtype) +
                                  ", declared " + DtypeName(it->second));
                continue;
            }

            it->second = inferred_dtype;
        }
    }

    void ValidateNodeDtypes(
        const tc::frontend::Node& node,
        const std::string& node_context,
        std::unordered_map<std::string, tc::frontend::DataT>& tensor_dtypes)
    {
        tc::frontend::DataT inferred_output_dtype;
        bool has_inferred_output_dtype = false;

        switch (node.get_op_kind()) {
            case tc::frontend::OpKind::kRelu:
            case tc::frontend::OpKind::kTranspose: {
                if (node.get_inputs().empty()) {
                    return;
                }

                tc::frontend::DataT input_dtype;
                if (!ResolveInputDtype(
                        node, node_context, 0, tensor_dtypes, input_dtype)) {
                    return;
                }

                if (!IsMvpNumericDataId(input_dtype.id)) {
                    report_.add_error("ERROR: " + node_context + " op '" +
                                      std::string(tc::frontend::ToString(
                                          node.get_op_kind())) +
                                      "' expects numeric dtype, got " +
                                      DtypeName(input_dtype));
                    return;
                }

                inferred_output_dtype = input_dtype;
                has_inferred_output_dtype = true;
                break;
            }

            case tc::frontend::OpKind::kAdd:
            case tc::frontend::OpKind::kMul:
            case tc::frontend::OpKind::kMatMul: {
                if (node.get_inputs().size() < 2) {
                    return;
                }

                tc::frontend::DataT lhs_dtype;
                tc::frontend::DataT rhs_dtype;
                const bool lhs_ok = ResolveInputDtype(
                    node, node_context, 0, tensor_dtypes, lhs_dtype);
                const bool rhs_ok = ResolveInputDtype(
                    node, node_context, 1, tensor_dtypes, rhs_dtype);
                if (!lhs_ok || !rhs_ok) {
                    return;
                }

                if (!IsMvpNumericDataId(lhs_dtype.id)) {
                    report_.add_error(
                        "ERROR: " + node_context + " op '" +
                        std::string(
                            tc::frontend::ToString(node.get_op_kind())) +
                        "' expects numeric dtype for input[0], got " +
                        DtypeName(lhs_dtype));
                    return;
                }

                if (!IsMvpNumericDataId(rhs_dtype.id)) {
                    report_.add_error(
                        "ERROR: " + node_context + " op '" +
                        std::string(
                            tc::frontend::ToString(node.get_op_kind())) +
                        "' expects numeric dtype for input[1], got " +
                        DtypeName(rhs_dtype));
                    return;
                }

                if (lhs_dtype.id != rhs_dtype.id) {
                    report_.add_error(
                        "ERROR: " + node_context + " op '" +
                        std::string(
                            tc::frontend::ToString(node.get_op_kind())) +
                        "' input dtypes mismatch: " + DtypeName(lhs_dtype) +
                        " vs " + DtypeName(rhs_dtype));
                    return;
                }

                inferred_output_dtype = lhs_dtype;
                has_inferred_output_dtype = true;
                break;
            }

            case tc::frontend::OpKind::kUnknown:
                return;
        }

        if (has_inferred_output_dtype) {
            PropagateNodeOutputDtype(
                node, node_context, inferred_output_dtype, tensor_dtypes);
        }
    }

    const std::vector<int64_t>* FindTensorShape(
        const std::unordered_map<std::string, std::vector<int64_t>>&
            tensor_shapes,
        const std::string& tensor_name) const
    {
        const auto it = tensor_shapes.find(tensor_name);
        if (it == tensor_shapes.end()) {
            return nullptr;
        }
        return &it->second;
    }

    bool ResolveInputShape(
        const tc::frontend::Node& node,
        const std::string& node_context,
        std::size_t input_index,
        const std::unordered_map<std::string, std::vector<int64_t>>&
            tensor_shapes,
        std::vector<int64_t>& out_shape)
    {
        if (input_index >= node.get_inputs().size()) {
            return false;
        }

        const std::string& input_name = node.get_inputs()[input_index];
        if (input_name.empty()) {
            return false;
        }

        const std::vector<int64_t>* shape =
            FindTensorShape(tensor_shapes, input_name);
        if (!shape) {
            report_.add_error("ERROR: " + node_context + " input[" +
                              std::to_string(input_index) + "] '" + input_name +
                              "' has undefined shape");
            return false;
        }

        out_shape = *shape;
        return true;
    }

    bool ResolveBinaryInputShapes(
        const tc::frontend::Node& node,
        const std::string& node_context,
        const std::unordered_map<std::string, std::vector<int64_t>>&
            tensor_shapes,
        std::vector<int64_t>& lhs_shape,
        std::vector<int64_t>& rhs_shape)
    {
        return ResolveInputShape(
                   node, node_context, 0, tensor_shapes, lhs_shape) &&
               ResolveInputShape(
                   node, node_context, 1, tensor_shapes, rhs_shape);
    }

    void ValidateDeclaredOutputShapes(
        const tc::frontend::Node& node,
        const std::string& node_context,
        const char* op_name,
        const std::vector<int64_t>& inferred_shape,
        const std::unordered_map<std::string, std::vector<int64_t>>&
            tensor_shapes)
    {
        for (const std::string& output_name : node.get_outputs()) {
            const std::vector<int64_t>* output_shape =
                FindTensorShape(tensor_shapes, output_name);
            if (output_shape &&
                !ShapesMatchForMvp(*output_shape, inferred_shape)) {
                report_.add_error("ERROR: " + node_context + " op '" +
                                  std::string(op_name) +
                                  "' output shape mismatch");
            }
        }
    }

    void PropagateNodeOutputShape(
        const tc::frontend::Node& node,
        const std::string& node_context,
        const std::vector<int64_t>& inferred_shape,
        std::unordered_map<std::string, std::vector<int64_t>>& tensor_shapes)
    {
        for (const std::string& output_name : node.get_outputs()) {
            if (output_name.empty()) {
                continue;
            }

            auto it = tensor_shapes.find(output_name);
            if (it == tensor_shapes.end()) {
                tensor_shapes.emplace(output_name, inferred_shape);
                continue;
            }

            if (!ShapesMatchForMvp(it->second, inferred_shape)) {
                report_.add_error("ERROR: " + node_context +
                                  " output shape mismatch for '" + output_name +
                                  "'");
                continue;
            }

            it->second = inferred_shape;
        }
    }

    void ValidateNodeShapes(
        const tc::frontend::Node& node,
        const std::string& node_context,
        std::unordered_map<std::string, std::vector<int64_t>>& tensor_shapes)
    {
        switch (node.get_op_kind()) {
            case tc::frontend::OpKind::kRelu: {
                if (node.get_inputs().empty()) {
                    return;
                }
                std::vector<int64_t> input_shape;
                if (!ResolveInputShape(
                        node, node_context, 0, tensor_shapes, input_shape)) {
                    return;
                }

                ValidateDeclaredOutputShapes(
                    node, node_context, "Relu", input_shape, tensor_shapes);
                PropagateNodeOutputShape(
                    node, node_context, input_shape, tensor_shapes);
                return;
            }

            case tc::frontend::OpKind::kAdd: {
                if (node.get_inputs().size() < 2) {
                    return;
                }
                std::vector<int64_t> lhs_shape;
                std::vector<int64_t> rhs_shape;
                if (!ResolveBinaryInputShapes(node,
                                              node_context,
                                              tensor_shapes,
                                              lhs_shape,
                                              rhs_shape)) {
                    return;
                }

                std::vector<int64_t> inferred_shape;
                if (!ComputeBroadcastShape(
                        lhs_shape, rhs_shape, inferred_shape)) {
                    report_.add_error("ERROR: " + node_context +
                                      " op 'Add' input shapes mismatch");
                    return;
                }

                ValidateDeclaredOutputShapes(
                    node, node_context, "Add", inferred_shape, tensor_shapes);
                PropagateNodeOutputShape(
                    node, node_context, inferred_shape, tensor_shapes);
                return;
            }

            case tc::frontend::OpKind::kMul: {
                if (node.get_inputs().size() < 2) {
                    return;
                }
                std::vector<int64_t> lhs_shape;
                std::vector<int64_t> rhs_shape;
                if (!ResolveBinaryInputShapes(node,
                                              node_context,
                                              tensor_shapes,
                                              lhs_shape,
                                              rhs_shape)) {
                    return;
                }

                std::vector<int64_t> inferred_shape;
                if (!ComputeBroadcastShape(
                        lhs_shape, rhs_shape, inferred_shape)) {
                    report_.add_error("ERROR: " + node_context +
                                      " op 'Mul' input shapes mismatch");
                    return;
                }

                ValidateDeclaredOutputShapes(
                    node, node_context, "Mul", inferred_shape, tensor_shapes);
                PropagateNodeOutputShape(
                    node, node_context, inferred_shape, tensor_shapes);
                return;
            }

            case tc::frontend::OpKind::kMatMul: {
                if (node.get_inputs().size() < 2) {
                    return;
                }
                std::vector<int64_t> lhs_shape;
                std::vector<int64_t> rhs_shape;
                if (!ResolveBinaryInputShapes(node,
                                              node_context,
                                              tensor_shapes,
                                              lhs_shape,
                                              rhs_shape)) {
                    return;
                }

                if (lhs_shape.size() != 2 || rhs_shape.size() != 2) {
                    report_.add_error("ERROR: " + node_context +
                                      " op 'MatMul' expects rank-2 inputs");
                    return;
                }

                if (lhs_shape[1] != -1 && rhs_shape[0] != -1 &&
                    lhs_shape[1] != rhs_shape[0]) {
                    report_.add_error("ERROR: " + node_context +
                                      " op 'MatMul' inner dimensions mismatch");
                    return;
                }

                std::vector<int64_t> inferred_shape{ lhs_shape[0],
                                                     rhs_shape[1] };
                ValidateDeclaredOutputShapes(node,
                                             node_context,
                                             "MatMul",
                                             inferred_shape,
                                             tensor_shapes);
                PropagateNodeOutputShape(
                    node, node_context, inferred_shape, tensor_shapes);
                return;
            }

            case tc::frontend::OpKind::kTranspose: {
                if (node.get_inputs().empty()) {
                    return;
                }
                std::vector<int64_t> input_shape;
                if (!ResolveInputShape(
                        node, node_context, 0, tensor_shapes, input_shape)) {
                    return;
                }

                const tc::frontend::Attribute* perm_attr = nullptr;
                for (const auto& attr : node.get_attrs()) {
                    if (attr && attr->get_name() == "perm") {
                        perm_attr = attr.get();
                        break;
                    }
                }
                if (!perm_attr || perm_attr->get_data_type().id !=
                                      tc::frontend::DataID::INT64) {
                    return;
                }

                const auto& perm = perm_attr->get_values<int64_t>();
                if (perm.size() != input_shape.size()) {
                    report_.add_error("ERROR: " + node_context +
                                      " op 'Transpose' perm rank mismatch");
                    return;
                }

                std::vector<int64_t> inferred_shape;
                inferred_shape.reserve(perm.size());
                for (const int64_t axis : perm) {
                    if (axis < 0 ||
                        static_cast<std::size_t>(axis) >= input_shape.size()) {
                        return;
                    }
                    inferred_shape.push_back(input_shape[axis]);
                }

                ValidateDeclaredOutputShapes(node,
                                             node_context,
                                             "Transpose",
                                             inferred_shape,
                                             tensor_shapes);
                PropagateNodeOutputShape(
                    node, node_context, inferred_shape, tensor_shapes);
                return;
            }

            case tc::frontend::OpKind::kUnknown:
                return;
        }
    }

    void RegisterTensorDtype(
        const std::string& tensor_name,
        const tc::frontend::DataT& dtype,
        const std::string& source_context,
        std::unordered_map<std::string, tc::frontend::DataT>& tensor_dtypes)
    {
        if (tensor_name.empty()) {
            return;
        }

        const auto [it, inserted] = tensor_dtypes.emplace(tensor_name, dtype);
        if (inserted) {
            return;
        }

        if (dtype.id == tc::frontend::DataID::UNDEFINED ||
            it->second.id == tc::frontend::DataID::UNDEFINED) {
            if (it->second.id == tc::frontend::DataID::UNDEFINED &&
                dtype.id != tc::frontend::DataID::UNDEFINED) {
                it->second = dtype;
            }
            return;
        }

        if (it->second.id != dtype.id) {
            report_.add_error(
                "ERROR: tensor '" + tensor_name +
                "' has conflicting dtypes: " + DtypeName(it->second) + " vs " +
                DtypeName(dtype) + " (" + source_context + ")");
        }
    }

    void BuildTensorDtypeTable(
        const tc::frontend::Graph& graph,
        std::unordered_map<std::string, tc::frontend::DataT>& tensor_dtypes)
    {
        tensor_dtypes.clear();

        for (std::size_t i = 0; i < graph.get_input_tensors().size(); ++i) {
            const auto& tensor = graph.get_input_tensors()[i];
            if (!tensor) {
                continue;
            }
            RegisterTensorDtype(tensor->get_name(),
                                tensor->get_data_type(),
                                "input_tensor[" + std::to_string(i) + "]",
                                tensor_dtypes);
        }

        for (std::size_t i = 0; i < graph.get_output_tensors().size(); ++i) {
            const auto& tensor = graph.get_output_tensors()[i];
            if (!tensor) {
                continue;
            }
            RegisterTensorDtype(tensor->get_name(),
                                tensor->get_data_type(),
                                "output_tensor[" + std::to_string(i) + "]",
                                tensor_dtypes);
        }

        for (std::size_t i = 0; i < graph.get_inits().size(); ++i) {
            const auto& init = graph.get_inits()[i];
            if (!init) {
                continue;
            }
            RegisterTensorDtype(init->get_name(),
                                init->get_data_type(),
                                "initializer[" + std::to_string(i) + "]",
                                tensor_dtypes);
        }
    }

    void RegisterTensorShape(
        const std::string& tensor_name,
        const std::vector<int64_t>& shape,
        std::unordered_map<std::string, std::vector<int64_t>>& tensor_shapes)
    {
        if (tensor_name.empty()) {
            return;
        }

        const auto [it, inserted] = tensor_shapes.emplace(tensor_name, shape);
        if (inserted) {
            return;
        }

        if (!it->second.empty() && !shape.empty() &&
            !ShapesMatchForMvp(it->second, shape)) {
            report_.add_error("ERROR: tensor '" + tensor_name +
                              "' has conflicting shapes");
        }

        if (it->second.empty() && !shape.empty()) {
            it->second = shape;
        }
    }

    void BuildTensorShapeTable(
        const tc::frontend::Graph& graph,
        std::unordered_map<std::string, std::vector<int64_t>>& tensor_shapes)
    {
        tensor_shapes.clear();

        for (const auto& tensor : graph.get_input_tensors()) {
            if (!tensor) {
                continue;
            }
            RegisterTensorShape(
                tensor->get_name(), tensor->get_shape(), tensor_shapes);
        }

        for (const auto& tensor : graph.get_output_tensors()) {
            if (!tensor) {
                continue;
            }
            RegisterTensorShape(
                tensor->get_name(), tensor->get_shape(), tensor_shapes);
        }

        for (const auto& init : graph.get_inits()) {
            if (!init) {
                continue;
            }
            RegisterTensorShape(
                init->get_name(), init->get_shape(), tensor_shapes);
        }
    }

    void ValidateNodeAttributes(const tc::frontend::Node& node,
                                const std::string& node_context)
    {
        const auto op_kind = node.get_op_kind();
        const auto& attrs = node.get_attrs();

        std::unordered_map<std::string, const tc::frontend::Attribute*>
            attrs_by_name;
        attrs_by_name.reserve(attrs.size());

        for (std::size_t i = 0; i < attrs.size(); ++i) {
            if (!attrs[i]) {
                report_.add_error("ERROR: " + node_context + ".attribute[" +
                                  std::to_string(i) + "] is null");
                continue;
            }

            const std::string& attr_name = attrs[i]->get_name();
            if (attr_name.empty()) {
                report_.add_error("ERROR: " + node_context + ".attribute[" +
                                  std::to_string(i) + "] has empty name");
                continue;
            }

            if (!attrs_by_name.emplace(attr_name, attrs[i].get()).second) {
                report_.add_error("ERROR: " + node_context +
                                  " has duplicate attribute '" + attr_name +
                                  "'");
                continue;
            }

            switch (op_kind) {
                case tc::frontend::OpKind::kRelu:
                case tc::frontend::OpKind::kAdd:
                case tc::frontend::OpKind::kMul:
                case tc::frontend::OpKind::kMatMul:
                    report_.add_error(
                        "ERROR: " + node_context + " op '" +
                        std::string(tc::frontend::ToString(op_kind)) +
                        "' does not support attributes");
                    break;

                case tc::frontend::OpKind::kTranspose:
                    if (attr_name != "perm") {
                        report_.add_error(
                            "ERROR: " + node_context +
                            " op 'Transpose' has unsupported attribute '" +
                            attr_name + "'");
                    }
                    break;

                case tc::frontend::OpKind::kUnknown:
                    break;
            }
        }

        if (op_kind != tc::frontend::OpKind::kTranspose) {
            return;
        }

        const auto perm_it = attrs_by_name.find("perm");
        if (perm_it == attrs_by_name.end()) {
            return;
        }

        const auto* perm_attr = perm_it->second;
        if (perm_attr->get_data_type().id != tc::frontend::DataID::INT64) {
            report_.add_error("ERROR: " + node_context +
                              " op 'Transpose' attribute 'perm' must be INTS");
            return;
        }

        const auto& perm = perm_attr->get_values<int64_t>();
        if (perm.empty()) {
            report_.add_error("ERROR: " + node_context +
                              " op 'Transpose' attribute 'perm' is empty");
            return;
        }

        std::unordered_set<int64_t> seen_axes;
        seen_axes.reserve(perm.size());
        for (const int64_t axis : perm) {
            if (axis < 0) {
                report_.add_error("ERROR: " + node_context +
                                  " op 'Transpose' attribute 'perm' contains "
                                  "negative axis " +
                                  std::to_string(axis));
            }

            if (!seen_axes.insert(axis).second) {
                report_.add_error("ERROR: " + node_context +
                                  " op 'Transpose' attribute 'perm' has "
                                  "duplicate axis " +
                                  std::to_string(axis));
            }
        }
    }

    void VisitOutputTensors(const tc::frontend::Graph::TensVecT& outputs)
    {
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (!outputs[i]) {
                report_.add_error("ERROR: output_tensor[" + std::to_string(i) +
                                  "] is null");
                continue;
            }

            if (outputs[i]->get_name().empty()) {
                report_.add_error("ERROR: output_tensor[" + std::to_string(i) +
                                  "] has empty name");
            }
        }
    }

    void VisitInitializers(const tc::frontend::Graph::InitVecT& initializers)
    {
        std::unordered_set<std::string> init_names;
        init_names.reserve(initializers.size());

        for (std::size_t i = 0; i < initializers.size(); ++i) {
            if (!initializers[i]) {
                report_.add_error("ERROR: initializer[" + std::to_string(i) +
                                  "] is null");
                continue;
            }

            const std::string& name = initializers[i]->get_name();
            if (name.empty()) {
                report_.add_error("ERROR: initializer[" + std::to_string(i) +
                                  "] has empty name");
                continue;
            }

            if (!init_names.insert(name).second) {
                report_.add_error("ERROR: duplicate initializer name '" + name +
                                  "'");
            }
        }
    }

    void VisitNodes(
        const tc::frontend::Graph::NodeVecT& nodes,
        std::unordered_map<std::string, tc::frontend::DataT>& tensor_dtypes,
        std::unordered_map<std::string, std::vector<int64_t>>& tensor_shapes)
    {
        std::unordered_set<std::string> global_outputs;

        for (std::size_t ni = 0; ni < nodes.size(); ++ni) {
            if (!nodes[ni]) {
                report_.add_error("ERROR: node[" + std::to_string(ni) +
                                  "] is null");
                continue;
            }

            const auto& node = *nodes[ni];
            const std::string node_context = BuildNodeContext(node, ni);

            if (node.get_outputs().empty()) {
                report_.add_error("ERROR: " + node_context + " has no output");
            }

            if (node.get_name_op().empty()) {
                report_.add_error("ERROR: " + node_context + " has no op type");
            }

            ValidateNodeSemantics(
                node, node_context, tensor_dtypes, tensor_shapes);

            std::unordered_set<std::string> local_outputs;
            local_outputs.reserve(node.get_outputs().size());
            for (const std::string& out : node.get_outputs()) {
                if (out.empty()) {
                    report_.add_error("ERROR: " + node_context +
                                      " has empty output name");
                    continue;
                }

                if (!local_outputs.insert(out).second) {
                    report_.add_error("ERROR: " + node_context +
                                      " has duplicate output name '" + out +
                                      "'");
                }

                if (!global_outputs.insert(out).second) {
                    report_.add_error(
                        "ERROR: duplicate output name across nodes '" + out +
                        "'");
                }
            }
        }
    }

    tc::frontend::verify::Report& report_;
};

} // namespace

namespace tc::frontend::verify {

void Report::clear()
{
    diagnostics_.clear();
    error_count_ = 0;
    warning_count_ = 0;
}

void Report::add_error(std::string message)
{
    diagnostics_.push_back({ Severity::kError, std::move(message) });
    ++error_count_;
}

void Report::add_warning(std::string message)
{
    diagnostics_.push_back({ Severity::kWarning, std::move(message) });
    ++warning_count_;
}

const char* SeverityToString(Severity severity) noexcept
{
    switch (severity) {
        case Severity::kError:
            return "ERROR";
        case Severity::kWarning:
            return "WARN";
    }
    return "UNKNOWN";
}

bool VerifyGraphForExecution(const Graph& graph, Report& out_report)
{
    out_report.clear();

    ExecutionVerifierVisitor verifier(out_report);
    verifier.VisitGraph(graph);

    return out_report.ok();
}

bool VerifyGraphForExecutable(const Graph& graph, Report& out_report)
{
    VerifyGraphForExecution(graph, out_report);

    if (graph.get_input_tensors().size() != 1) {
        out_report.add_error(
            "ERROR: executable verifier expects exactly 1 runtime input, got " +
            std::to_string(graph.get_input_tensors().size()));
    }
    if (graph.get_output_tensors().size() != 1) {
        out_report.add_error("ERROR: executable verifier expects exactly 1 "
                             "runtime output, got " +
                             std::to_string(graph.get_output_tensors().size()));
    }

    auto validate_runtime_tensor = [&out_report](const TensorInfo& tensor,
                                                 const char* role) {
        const std::string& name = tensor.get_name();
        const std::string label =
            std::string("runtime ") + role + " '" + name + "'";

        if (tensor.get_data_type().id != DataID::FLOAT) {
            out_report.add_error("ERROR: executable verifier " + label +
                                 " must be float32");
        }

        const auto& shape = tensor.get_shape();
        if (shape.empty()) {
            out_report.add_error("ERROR: executable verifier " + label +
                                 " must have shape");
            return;
        }

        for (const int64_t dim : shape) {
            if (dim < 0) {
                out_report.add_error("ERROR: executable verifier " + label +
                                     " must have static shape");
                return;
            }
        }
    };

    for (const auto& input : graph.get_input_tensors()) {
        if (input) {
            validate_runtime_tensor(*input, "input");
        }
    }
    for (const auto& output : graph.get_output_tensors()) {
        if (output) {
            validate_runtime_tensor(*output, "output");
        }
    }

    for (const auto& init : graph.get_inits()) {
        if (!init) {
            continue;
        }

        const std::string label = "initializer '" + init->get_name() + "'";
        if (init->get_data_type().id != DataID::FLOAT) {
            out_report.add_error("ERROR: executable verifier " + label +
                                 " must be float32");
        }

        const auto& shape = init->get_shape();
        if (shape.empty()) {
            out_report.add_error("ERROR: executable verifier " + label +
                                 " must have shape");
        } else {
            for (const int64_t dim : shape) {
                if (dim < 0) {
                    out_report.add_error("ERROR: executable verifier " + label +
                                         " must have static shape");
                    break;
                }
            }
        }

        if (!init->has_values()) {
            out_report.add_error("ERROR: executable verifier " + label +
                                 " must have data");
        }
    }

    return out_report.ok();
}

} // namespace tc::frontend::verify
