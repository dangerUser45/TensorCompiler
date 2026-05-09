cmake_minimum_required(VERSION 3.20)
project(TensorCompiler LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_subdirectory(backend)
add_subdirectory(frontend)

add_executable(compare_f32
    tests/reference/compare_f32.cpp
)

target_link_libraries(compare_f32 PRIVATE project_warnings)

add_test(
  NAME exec_conv_relu
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/run_exec_conv_relu.sh
          $<TARGET_FILE:frontend_driver>
          ${CMAKE_CURRENT_SOURCE_DIR}/frontend/models/exec_conv_relu.onnx
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/exec_conv_relu_input.f32
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/exec_conv_relu_expected.f32
          $<TARGET_FILE:compare_f32>
          ${CMAKE_CURRENT_BINARY_DIR}/reference
)
