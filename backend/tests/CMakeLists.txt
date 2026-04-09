add_executable(tc_backend_smoke
    smoke.cpp
)

target_link_libraries(tc_backend_smoke PRIVATE
    tc_backend
)

add_test(
  NAME backend_smoke
  COMMAND $<TARGET_FILE:tc_backend_smoke>
)

if (TC_BACKEND_HAS_MLIR)
add_executable(tc_backend_mlir_parse_test
    mlir_parse_test.cpp
)

target_link_libraries(tc_backend_mlir_parse_test PRIVATE
    tc_backend
)

add_test(
  NAME backend_mlir_parse
  COMMAND $<TARGET_FILE:tc_backend_mlir_parse_test>
)

add_executable(tc_backend_mlir_lower_to_llvm_test
    mlir_lower_to_llvm_test.cpp
)

target_link_libraries(tc_backend_mlir_lower_to_llvm_test PRIVATE
    tc_backend
)

add_test(
  NAME backend_mlir_lower_to_llvm
  COMMAND $<TARGET_FILE:tc_backend_mlir_lower_to_llvm_test>
)
endif()
