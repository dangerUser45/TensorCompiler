#include "graph_verifier.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>

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
        BuildTensorDtypeTable(graph, tensor_dtypes);
        VisitNodes(graph.get_nodes(), tensor_dtypes);
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
        std::unordered_map<std::string, tc::frontend::DataT>& tensor_dtypes)
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
        std::unordered_map<std::string, tc::frontend::DataT>& tensor_dtypes)
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

            ValidateNodeSemantics(node, node_context, tensor_dtypes);

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

} // namespace tc::frontend::verify
