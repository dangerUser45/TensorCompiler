find_package(GTest CONFIG QUIET)
find_package(GTest QUIET)

if (NOT TARGET GTest::gtest_main)
  set(LOCAL_GTEST_ROOT "$ENV{HOME}/Desktop/libs/googletest")
  if (EXISTS "${LOCAL_GTEST_ROOT}/CMakeLists.txt")
    add_subdirectory(
      "${LOCAL_GTEST_ROOT}"
      "${CMAKE_BINARY_DIR}/_deps/local-googletest-unit"
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

add_executable(op_kind_test
  op_kind_test.cpp
)

target_include_directories(op_kind_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(op_kind_test PRIVATE
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

add_test(
  NAME op_kind_test
  COMMAND op_kind_test
)

set_tests_properties(op_kind_test PROPERTIES
  LABELS "unit"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(importer_op_kind_test
  importer_op_kind_test.cpp
)

target_link_libraries(importer_op_kind_test PRIVATE
  tc_frontend_onnx
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

target_compile_definitions(importer_op_kind_test PRIVATE
  TC_TEST_MODELS_DIR=\"${CMAKE_SOURCE_DIR}/models\"
)

add_test(
  NAME importer_op_kind_test
  COMMAND importer_op_kind_test
)

set_tests_properties(importer_op_kind_test PROPERTIES
  LABELS "unit"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(graph_verifier_semantic_test
  graph_verifier_semantic_test.cpp
)

target_link_libraries(graph_verifier_semantic_test PRIVATE
  tc_frontend_onnx
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

add_test(
  NAME graph_verifier_semantic_test
  COMMAND graph_verifier_semantic_test
)

set_tests_properties(graph_verifier_semantic_test PROPERTIES
  LABELS "unit"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(importer_tensor_contract_test
  importer_tensor_contract_test.cpp
)

target_link_libraries(importer_tensor_contract_test PRIVATE
  tc_frontend_onnx
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

target_compile_definitions(importer_tensor_contract_test PRIVATE
  TC_TEST_MODELS_DIR=\"${CMAKE_SOURCE_DIR}/models\"
)

add_test(
  NAME importer_tensor_contract_test
  COMMAND importer_tensor_contract_test
)

set_tests_properties(importer_tensor_contract_test PROPERTIES
  LABELS "unit"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(importer_attr_normalization_test
  importer_attr_normalization_test.cpp
)

target_link_libraries(importer_attr_normalization_test PRIVATE
  tc_frontend_onnx
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

add_test(
  NAME importer_attr_normalization_test
  COMMAND importer_attr_normalization_test
)

set_tests_properties(importer_attr_normalization_test PROPERTIES
  LABELS "unit"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(mlir_emitter_test
  mlir_emitter_test.cpp
)

target_link_libraries(mlir_emitter_test PRIVATE
  tc_frontend_onnx
  ${TC_GTEST_MAIN_TARGET}
  project_warnings
)

add_test(
  NAME mlir_emitter_test
  COMMAND mlir_emitter_test
)

set_tests_properties(mlir_emitter_test PROPERTIES
  LABELS "unit"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
