#pragma once

#include <memory>
#include <string>

namespace tc::backend {

struct BackendDiagnostic final
{
    std::string stage = "backend";
    std::string message;
};

std::string FormatBackendDiagnostic(const BackendDiagnostic& diagnostic);
std::string GetDefaultLlvmLoweringPassPipeline();

class ParsedMlirModule final
{
public:
    ParsedMlirModule();
    ~ParsedMlirModule();

    ParsedMlirModule(const ParsedMlirModule&) = delete;
    ParsedMlirModule& operator=(const ParsedMlirModule&) = delete;
    ParsedMlirModule(ParsedMlirModule&&) noexcept;
    ParsedMlirModule& operator=(ParsedMlirModule&&) noexcept;

    bool empty() const noexcept;
    void clear();
    std::string Dump() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend bool LowerMlirModuleToLlvmDialect(ParsedMlirModule& module,
                                             BackendDiagnostic& out_diagnostic,
                                             const std::string& pass_pipeline);
    friend bool ParseMlirModuleFromString(const std::string& source,
                                          const std::string& source_name,
                                          ParsedMlirModule& out_module,
                                          BackendDiagnostic& out_diagnostic);
    friend bool RunMlirPassPipeline(ParsedMlirModule& module,
                                    const std::string& pass_pipeline,
                                    BackendDiagnostic& out_diagnostic);
};

bool ParseMlirModuleFromString(const std::string& source,
                               const std::string& source_name,
                               ParsedMlirModule& out_module,
                               BackendDiagnostic& out_diagnostic);

bool RunMlirPassPipeline(ParsedMlirModule& module,
                         const std::string& pass_pipeline,
                         BackendDiagnostic& out_diagnostic);

bool LowerMlirModuleToLlvmDialect(ParsedMlirModule& module,
                                  BackendDiagnostic& out_diagnostic,
                                  const std::string& pass_pipeline = {});

} // namespace tc::backend
