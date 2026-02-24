add_test(
  NAME end_to_end
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/end_to_end.sh
          $<TARGET_FILE:Matrix>
)

set_tests_properties(end_to_end PROPERTIES
  LABELS "end_to_end"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
