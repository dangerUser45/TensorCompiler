#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tc::frontend {
class Graph;
} // namespace tc::frontend

namespace tc::frontend::verify {

enum class Severity
{
    kError,
    kWarning
};

struct Diagnostic final
{
    Severity severity = Severity::kError;
    std::string message;
};

class Report final
{
public:
    void clear();

    void add_error(std::string message);
    void add_warning(std::string message);

    bool ok() const noexcept
    {
        return error_count_ == 0;
    }
    std::size_t error_count() const noexcept
    {
        return error_count_;
    }
    std::size_t warning_count() const noexcept
    {
        return warning_count_;
    }
    const std::vector<Diagnostic>& diagnostics() const noexcept
    {
        return diagnostics_;
    }

private:
    std::size_t error_count_ = 0;
    std::size_t warning_count_ = 0;
    std::vector<Diagnostic> diagnostics_;
};

const char* SeverityToString(Severity severity) noexcept;

bool VerifyGraphForExecution(const Graph& graph, Report& out_report);

bool VerifyGraphForExecutable(const Graph& graph, Report& out_report);

} // namespace tc::frontend::verify
