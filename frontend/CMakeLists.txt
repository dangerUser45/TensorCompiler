cmake_minimum_required(VERSION 3.20)
project(TensorCompilerFrontend)

find_package(Protobuf CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(utf8_range CONFIG REQUIRED)
find_package(ONNX CONFIG REQUIRED)

add_library(tc_frontend_onnx STATIC
    src/onnx_importer.cpp
)

target_compile_features(tc_frontend_onnx PUBLIC cxx_std_17)
target_include_directories(tc_frontend_onnx PUBLIC include)

target_link_libraries(tc_frontend_onnx PUBLIC ONNX::onnx)

add_executable(onnx_smoke src/onnx_smoke.cpp)
target_link_libraries(onnx_smoke PRIVATE tc_frontend_onnx)
