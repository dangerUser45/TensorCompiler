cmake_minimum_required(VERSION 3.20)
project(TensorCompilerFrontend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Protobuf CONFIG QUIET)
if (NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
endif()

set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnx_lib")
set(PROTO_FILES
    "${PROTO_DIR}/onnx.proto"
)
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnx_lib")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

if (TARGET protobuf::protoc)
    set(PROTOC_EXE $<TARGET_FILE:protobuf::protoc>)
elseif (DEFINED Protobuf_PROTOC_EXECUTABLE)
    set(PROTOC_EXE "${Protobuf_PROTOC_EXECUTABLE}")
else()
    message(FATAL_ERROR "Could not find protoc executable from Protobuf package")
endif()

set(PROTO_SRCS)
set(PROTO_HDRS)
foreach(PROTO_FILE IN LISTS PROTO_FILES)
    get_filename_component(PROTO_NAME "${PROTO_FILE}" NAME_WE)
    list(APPEND PROTO_SRCS "${PROTO_GEN_DIR}/${PROTO_NAME}.pb.cc")
    list(APPEND PROTO_HDRS "${PROTO_GEN_DIR}/${PROTO_NAME}.pb.h")
endforeach()

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND ${PROTOC_EXE}
            --cpp_out=${PROTO_GEN_DIR}
            --proto_path=${PROTO_DIR}
            ${PROTO_FILES}
    DEPENDS ${PROTO_FILES}
    COMMENT "Generating C++ sources from ONNX protobuf schema"
    VERBATIM
)

add_library(tc_frontend_onnx STATIC
    src/onnx_importer.cpp
    src/graph_verifier.cpp
    src/mlir_emitter.cpp
    src/model_metadata.cpp
    ${PROTO_SRCS}
)

add_library(project_warnings INTERFACE)
target_compile_options(project_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

target_include_directories(tc_frontend_onnx PUBLIC
    include
    tests/end2end/include
    "${PROTO_GEN_DIR}"
)

target_link_libraries(tc_frontend_onnx PUBLIC protobuf::libprotobuf)
target_link_libraries(tc_frontend_onnx PRIVATE project_warnings)

add_executable(frontend_driver
    src/frontend_driver.cpp
    tests/end2end/src/graph_hash.cpp    
)

target_compile_definitions(frontend_driver PRIVATE
    TC_DEFAULT_DUMP_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/dump\" 
    TC_DEFAULT_METADATA_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/metadata\"
)
target_link_libraries(frontend_driver PRIVATE
    tc_frontend_onnx
    project_warnings
)

enable_testing()
add_subdirectory(tests)
