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
