#include "mlir_pipeline.hpp"

#include "mlir_context.hpp"

#if TC_BACKEND_HAS_MLIR
#include <mutex>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/InitAllPasses.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Parser/Parser.h>
#endif

namespace tc::backend {

struct ParsedMlirModule::Impl final
{
#if TC_BACKEND_HAS_MLIR
    MlirContextPtr context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
#endif
};

#if TC_BACKEND_HAS_MLIR
namespace {

void EnsureMlirPassesRegistered()
{
    static std::once_flag once;
    std::call_once(once, []() { mlir::registerAllPasses(); });
}

} // namespace
#endif

std::string FormatBackendDiagnostic(const BackendDiagnostic& diagnostic)
{
    if (diagnostic.stage.empty()) {
        return diagnostic.message;
    }
    return diagnostic.stage + ": " + diagnostic.message;
}

std::string GetDefaultLlvmLoweringPassPipeline()
{
    return "canonicalize,cse,convert-to-llvm,reconcile-unrealized-casts";
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

bool RunMlirPassPipeline(ParsedMlirModule& module,
                         const std::string& pass_pipeline,
                         BackendDiagnostic& out_diagnostic)
{
    out_diagnostic = BackendDiagnostic{};

    if (!IsMlirAvailable()) {
        out_diagnostic.message =
            "MLIR support is disabled: backend was configured without MLIR";
        return false;
    }

#if TC_BACKEND_HAS_MLIR
    if (module.empty()) {
        out_diagnostic.stage = "mlir-pass-pipeline";
        out_diagnostic.message = "cannot run pass pipeline on an empty MLIR module";
        return false;
    }

    if (pass_pipeline.empty()) {
        return true;
    }

    EnsureMlirPassesRegistered();

    mlir::PassManager pass_manager = mlir::PassManager::on<mlir::ModuleOp>(
        module.impl_->context.get(), mlir::OpPassManager::Nesting::Implicit);
    pass_manager.enableVerifier(true);

    std::string pipeline_errors;
    llvm::raw_string_ostream pipeline_errors_stream(pipeline_errors);
    if (mlir::failed(
            mlir::parsePassPipeline(pass_pipeline, pass_manager,
                                    pipeline_errors_stream))) {
        pipeline_errors_stream.flush();
        out_diagnostic.stage = "mlir-pass-pipeline";
        out_diagnostic.message =
            "failed to parse MLIR pass pipeline '" + pass_pipeline + "'";
        if (!pipeline_errors.empty()) {
            out_diagnostic.message += '\n' + pipeline_errors;
        }
        return false;
    }

    std::string captured_diagnostics;
    mlir::ScopedDiagnosticHandler diagnostic_handler(
        module.impl_->context.get(), [&](mlir::Diagnostic& diagnostic) {
            if (!captured_diagnostics.empty()) {
                captured_diagnostics += '\n';
            }
            captured_diagnostics += diagnostic.str();
        });

    if (mlir::failed(pass_manager.run(module.impl_->module.get()))) {
        out_diagnostic.stage = "mlir-pass-pipeline";
        out_diagnostic.message =
            "MLIR pass pipeline failed: '" + pass_pipeline + "'";
        if (!captured_diagnostics.empty()) {
            out_diagnostic.message += '\n' + captured_diagnostics;
        }
        return false;
    }

    return true;
#else
    (void)module;
    (void)pass_pipeline;
    return false;
#endif
}

bool LowerMlirModuleToLlvmDialect(ParsedMlirModule& module,
                                  BackendDiagnostic& out_diagnostic,
                                  const std::string& pass_pipeline)
{
    out_diagnostic = BackendDiagnostic{};

    if (!IsMlirAvailable()) {
        out_diagnostic.message =
            "MLIR support is disabled: backend was configured without MLIR";
        return false;
    }

    const std::string& effective_pipeline =
        pass_pipeline.empty() ? GetDefaultLlvmLoweringPassPipeline()
                              : pass_pipeline;

#if TC_BACKEND_HAS_MLIR
    if (!RunMlirPassPipeline(module, effective_pipeline, out_diagnostic)) {
        out_diagnostic.stage = "llvm-lowering";
        return false;
    }

    return true;
#else
    (void)module;
    (void)effective_pipeline;
    return false;
#endif
}

} // namespace tc::backend
