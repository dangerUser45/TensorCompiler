cmake_minimum_required(VERSION 3.20)
project(TensorCompilerBackend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT TARGET project_warnings)
    add_library(project_warnings INTERFACE)
    target_compile_options(project_warnings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR CONFIG QUIET)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
if (MLIR_FOUND)
    message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
    set(TC_BACKEND_HAS_MLIR 1)
else()
    message(STATUS
        "MLIR package was not found. Building LLVM-only backend stubs. "
        "Set MLIR_DIR to enable MLIR integration."
    )
    set(TC_BACKEND_HAS_MLIR 0)
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
if (MLIR_FOUND AND DEFINED MLIR_CMAKE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
endif()

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")

add_library(tc_backend STATIC
    src/llvm_target_init.cpp
    src/mlir_context.cpp
    src/mlir_pipeline.cpp
)

target_compile_options(tc_backend PRIVATE
    ${LLVM_DEFINITIONS_LIST}
)

target_compile_definitions(tc_backend PRIVATE
    TC_BACKEND_HAS_MLIR=${TC_BACKEND_HAS_MLIR}
    TC_BACKEND_LLVM_VERSION_STRING=\"${LLVM_PACKAGE_VERSION}\"
)

target_include_directories(tc_backend
    PUBLIC
        include
    SYSTEM PRIVATE
        ${LLVM_INCLUDE_DIRS}
)

if (MLIR_FOUND)
    target_include_directories(tc_backend SYSTEM PRIVATE
        ${MLIR_INCLUDE_DIRS}
    )
endif()

target_link_libraries(tc_backend PRIVATE
    LLVM
    project_warnings
)

if (MLIR_FOUND)
    target_link_libraries(tc_backend PRIVATE
        MLIRArithDialect
        MLIRAsmParser
        MLIRFuncDialect
        MLIRIR
        MLIRLinalgDialect
        MLIRParser
        MLIRSupport
        MLIRTensorDialect
    )
endif()

enable_testing()
add_subdirectory(tests)
