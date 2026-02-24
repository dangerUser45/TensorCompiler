cmake_minimum_required(VERSION 3.20)
project(TensorCompilerFrontend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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
    ${PROTO_SRCS}
)

target_include_directories(tc_frontend_onnx PUBLIC
    include
    "${PROTO_GEN_DIR}"
)
target_link_libraries(tc_frontend_onnx PUBLIC protobuf::libprotobuf)

add_executable(onnx_smoke src/onnx_smoke.cpp)
target_link_libraries(onnx_smoke PRIVATE tc_frontend_onnx)

enable_testing()
add_subdirectory(tests)
