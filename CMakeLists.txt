cmake_minimum_required(VERSION 3.20)
project(TensorCompiler LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_subdirectory(backend)
add_subdirectory(frontend)
