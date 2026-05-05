add_test(
  NAME end2end
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/end2end.sh
          $<TARGET_FILE:frontend_driver>
          ${CMAKE_CURRENT_SOURCE_DIR}/expected_hashes.txt
          ${CMAKE_SOURCE_DIR}/models
)

set_tests_properties(end2end PROPERTIES
  LABELS "end2end"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

find_package(GTest CONFIG QUIET)
find_package(GTest QUIET)

if (NOT TARGET GTest::gtest_main)
  set(LOCAL_GTEST_ROOT "$ENV{HOME}/Desktop/libs/googletest")
  if (EXISTS "${LOCAL_GTEST_ROOT}/CMakeLists.txt")
    add_subdirectory(
      "${LOCAL_GTEST_ROOT}"
      "${CMAKE_BINARY_DIR}/_deps/local-googletest"
      EXCLUDE_FROM_ALL
    )
  endif()
endif()

if (TARGET GTest::gtest_main)
  set(TC_GTEST_MAIN_TARGET GTest::gtest_main)
elseif (TARGET gtest_main)
  set(TC_GTEST_MAIN_TARGET gtest_main)
else()
  message(FATAL_ERROR "gtest_main target not found. Install GTest or provide ~/Desktop/libs/googletest")
endif()

add_executable(unsupported_op_test
  src/unsupported_op_test.cpp
)
target_link_libraries(unsupported_op_test PRIVATE
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

add_test(
  NAME unsupported_op
  COMMAND unsupported_op_test
          --driver $<TARGET_FILE:frontend_driver>
          --model ${CMAKE_SOURCE_DIR}/models/candy-8.onnx
)

set_tests_properties(unsupported_op PROPERTIES
  LABELS "end2end"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(emit_mlir_test
  src/emit_mlir_test.cpp
)

target_link_libraries(emit_mlir_test PRIVATE
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

add_test(
  NAME emit_mlir
  COMMAND emit_mlir_test
          --driver $<TARGET_FILE:frontend_driver>
          --model ${CMAKE_SOURCE_DIR}/models/single_relu.onnx
)

set_tests_properties(emit_mlir PROPERTIES
  LABELS "end2end"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_test(
  NAME emit_mlir_two_transposes
  COMMAND emit_mlir_test
          --driver $<TARGET_FILE:frontend_driver>
          --model ${CMAKE_SOURCE_DIR}/models/two_transposes.onnx
          --expect op=Transpose
          --expect builtin.unrealized_conversion_cast
          --expect tensor<3x2x4xf32>
)

set_tests_properties(emit_mlir_two_transposes PROPERTIES
  LABELS "end2end"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_test(
  NAME emit_mlir_add
  COMMAND emit_mlir_test
          --driver $<TARGET_FILE:frontend_driver>
          --model ${CMAKE_SOURCE_DIR}/models/add.onnx
          --expect op=Add
)

set_tests_properties(emit_mlir_add PROPERTIES
  LABELS "end2end"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_test(
  NAME emit_mlir_matmul
  COMMAND emit_mlir_test
          --driver $<TARGET_FILE:frontend_driver>
          --model ${CMAKE_SOURCE_DIR}/models/matmul_2d.onnx
          --expect op=MatMul
          --expect builtin.unrealized_conversion_cast
)

set_tests_properties(emit_mlir_matmul PROPERTIES
  LABELS "end2end"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(verify_shape_invalid_test
  src/verify_shape_invalid_test.cpp
)

target_link_libraries(verify_shape_invalid_test PRIVATE
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

add_test(
  NAME verify_shape_invalid
  COMMAND verify_shape_invalid_test
          --driver $<TARGET_FILE:frontend_driver>
          --model ${CMAKE_SOURCE_DIR}/models/matmul_3d.onnx
)

set_tests_properties(verify_shape_invalid PROPERTIES
  LABELS "end2end"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
