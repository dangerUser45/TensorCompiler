#include "tc/backend/mlir_pipeline.hpp"

#include "tc/backend/mlir_context.hpp"

#if TC_BACKEND_HAS_MLIR
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Parser/Parser.h>
#endif

namespace tc::backend {

struct ParsedMlirModule::Impl final
{
#if TC_BACKEND_HAS_MLIR
    std::unique_ptr<mlir::MLIRContext> context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
#endif
};

std::string FormatBackendDiagnostic(const BackendDiagnostic& diagnostic)
{
    if (diagnostic.stage.empty()) {
        return diagnostic.message;
    }
    return diagnostic.stage + ": " + diagnostic.message;
}

ParsedMlirModule::ParsedMlirModule() = default;
ParsedMlirModule::~ParsedMlirModule() = default;
ParsedMlirModule::ParsedMlirModule(ParsedMlirModule&&) noexcept = default;
ParsedMlirModule& ParsedMlirModule::operator=(ParsedMlirModule&&) noexcept =
    default;

bool ParsedMlirModule::empty() const noexcept
{
#if TC_BACKEND_HAS_MLIR
    return !impl_ || !impl_->module;
#else
    return true;
#endif
}

void ParsedMlirModule::clear()
{
    impl_.reset();
}

std::string ParsedMlirModule::Dump() const
{
#if TC_BACKEND_HAS_MLIR
    if (empty()) {
        return {};
    }

    std::string buffer;
    llvm::raw_string_ostream out(buffer);
    impl_->module->print(out);
    return buffer;
#else
    return {};
#endif
}

bool ParseMlirModuleFromString(const std::string& source,
                               const std::string& source_name,
                               ParsedMlirModule& out_module,
                               BackendDiagnostic& out_diagnostic)
{
    out_module.clear();
    out_diagnostic = BackendDiagnostic{};

    if (!IsMlirAvailable()) {
        out_diagnostic.message =
            "MLIR support is disabled: backend was configured without MLIR";
        return false;
    }

#if TC_BACKEND_HAS_MLIR
    auto impl = std::make_unique<ParsedMlirModule::Impl>();
    impl->context = CreateMlirContext();
    if (!impl->context) {
        out_diagnostic.message = "failed to create MLIR context";
        return false;
    }

    std::string captured_diagnostics;
    mlir::ScopedDiagnosticHandler diagnostic_handler(
        impl->context.get(), [&](mlir::Diagnostic& diagnostic) {
            if (!captured_diagnostics.empty()) {
                captured_diagnostics += '\n';
            }
            captured_diagnostics += diagnostic.str();
        });

    const llvm::StringRef effective_source_name =
        source_name.empty() ? llvm::StringRef("<memory>")
                            : llvm::StringRef(source_name);
    const mlir::ParserConfig parser_config(impl->context.get());
    impl->module = mlir::parseSourceString<mlir::ModuleOp>(
        source, parser_config, effective_source_name);

    if (!impl->module) {
        out_diagnostic.message =
            "failed to parse MLIR module from '" +
            effective_source_name.str() + "'";
        if (!captured_diagnostics.empty()) {
            out_diagnostic.message += '\n' + captured_diagnostics;
        }
        return false;
    }

    out_module.impl_ = std::move(impl);
    return true;
#else
    (void)source;
    (void)source_name;
    return false;
#endif
}

} // namespace tc::backend
