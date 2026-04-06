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
