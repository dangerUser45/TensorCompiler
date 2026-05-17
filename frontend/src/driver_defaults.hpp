#pragma once

#include <string_view>

#ifndef TC_DEFAULT_DUMP_DIR
#define TC_DEFAULT_DUMP_DIR "build/dump"
#endif
#ifndef TC_DEFAULT_HASH_DIR
#define TC_DEFAULT_HASH_DIR "build/hash"
#endif
#ifndef TC_DEFAULT_MLIR_DIR
#define TC_DEFAULT_MLIR_DIR "build/mlir"
#endif
#ifndef TC_DEFAULT_METADATA_DIR
#define TC_DEFAULT_METADATA_DIR "build/metadata"
#endif
#ifndef TC_DEFAULT_LLVM_DIR
#define TC_DEFAULT_LLVM_DIR "build/llvm"
#endif
#ifndef TC_DEFAULT_ASM_DIR
#define TC_DEFAULT_ASM_DIR "build/asm"
#endif
#ifndef TC_DEFAULT_OBJECT_DIR
#define TC_DEFAULT_OBJECT_DIR "build/object"
#endif
#ifndef TC_DEFAULT_EXE_DIR
#define TC_DEFAULT_EXE_DIR "build/bin"
#endif

namespace tc::frontend::driver {

inline constexpr std::string_view kDefaultDumpDir = TC_DEFAULT_DUMP_DIR;
inline constexpr std::string_view kDefaultHashDir = TC_DEFAULT_HASH_DIR;
inline constexpr std::string_view kDefaultMlirDir = TC_DEFAULT_MLIR_DIR;
inline constexpr std::string_view kDefaultMetadataDir = TC_DEFAULT_METADATA_DIR;
inline constexpr std::string_view kDefaultLlvmDir = TC_DEFAULT_LLVM_DIR;
inline constexpr std::string_view kDefaultAsmDir = TC_DEFAULT_ASM_DIR;
inline constexpr std::string_view kDefaultObjectDir = TC_DEFAULT_OBJECT_DIR;
inline constexpr std::string_view kDefaultExeDir = TC_DEFAULT_EXE_DIR;

} // namespace tc::frontend::driver
