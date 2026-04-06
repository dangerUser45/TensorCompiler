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
