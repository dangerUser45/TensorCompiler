#include "graph_verifier.hpp"

#include <cstddef>
#include <string>
#include <unordered_set>

namespace {

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
        VisitNodes(graph.get_nodes());
    }

private:
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

    void VisitNodes(const tc::frontend::Graph::NodeVecT& nodes)
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
