#include "backend_runner.hpp"

#include <fstream>
#include <iostream>
#include <string_view>

#if TC_FRONTEND_HAS_BACKEND
#include "codegen_driver.hpp"
#include "executable_linker.hpp"
#include "object_emitter.hpp"
#endif

namespace tc::frontend::driver {

#if !TC_FRONTEND_HAS_BACKEND

int RunBackendPipeline(const BackendRequest&,
                       const std::string&,
                       const std::string&,
                       const std::string&,
                       const std::string&)
{
    std::cerr << "ERROR[backend]: frontend_driver was built without backend "
                 "codegen support\n";
    return 1;
}

#else

namespace {

bool WriteTextFile(const std::string& path,
                   const std::string& contents,
                   std::string_view stage)
{
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "ERROR[" << stage
                  << "]: failed to open output file: " << path << '\n';
        return false;
    }
    out << contents;
    if (!out.good()) {
        std::cerr << "ERROR[" << stage
                  << "]: failed to write output file: " << path << '\n';
        return false;
    }
    return true;
}

void PrintBackendError(const tc::backend::BackendDiagnostic& diag)
{
    std::cerr << "ERROR[backend]: "
              << tc::backend::FormatBackendDiagnostic(diag) << '\n';
}

} // namespace

int RunBackendPipeline(const BackendRequest& request,
                       const std::string& mlir_text,
                       const std::string& input_path,
                       const std::string& metadata_json,
                       const std::string& metadata_path)
{
    tc::backend::BackendDiagnostic diag;

    if (request.llvm_path) {
        std::string llvm_ir;
        if (!tc::backend::EmitLlvmIrFromMlirText(mlir_text,
                                                 input_path,
                                                 request.pass_pipeline,
                                                 request.target_triple,
                                                 llvm_ir,
                                                 diag)) {
            PrintBackendError(diag);
            return 1;
        }
        if (!WriteTextFile(*request.llvm_path, llvm_ir, "backend")) {
            return 1;
        }
        std::cout << "LLVM IR written to: " << *request.llvm_path << '\n';
    }

    if (request.asm_path) {
        std::string asm_text;
        if (!tc::backend::EmitAsmFromMlirText(mlir_text,
                                              input_path,
                                              request.pass_pipeline,
                                              request.target_triple,
                                              asm_text,
                                              diag)) {
            PrintBackendError(diag);
            return 1;
        }
        if (!WriteTextFile(*request.asm_path, asm_text, "backend")) {
            return 1;
        }
        std::cout << "ASM written to: " << *request.asm_path << '\n';
    }

    if (request.object_path) {
        if (!tc::backend::EmitObjectFromMlirText(mlir_text,
                                                 input_path,
                                                 request.pass_pipeline,
                                                 request.target_triple,
                                                 *request.object_path,
                                                 diag)) {
            PrintBackendError(diag);
            return 1;
        }
        std::cout << "Object written to: " << *request.object_path << '\n';
    }

    if (request.exe_path) {
        if (!WriteTextFile(metadata_path, metadata_json, "frontend")) {
            return 1;
        }
        std::cout << "Metadata written to: " << metadata_path << '\n';

        if (!tc::backend::LinkExecutableWithRuntime(
                *request.object_path, metadata_path, *request.exe_path, diag)) {
            PrintBackendError(diag);
            return 1;
        }
        std::cout << "Executable written to: " << *request.exe_path << '\n';
    }

    return 0;
}

#endif // TC_FRONTEND_HAS_BACKEND

} // namespace tc::frontend::driver
