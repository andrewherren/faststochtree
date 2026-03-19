cmake_minimum_required(VERSION 3.20)
project(faststochtree CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Core library
add_library(faststochtree
    src/quantize.cpp
    src/mcmc.cpp
    src/gfr.cpp
    src/sampler.cpp
)
target_include_directories(faststochtree PUBLIC include)

# Benchmark executable
add_executable(faststochtree_bench bench/main.cpp)
target_link_libraries(faststochtree_bench PRIVATE faststochtree)

# Tests
enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(test_bart tests/test_bart.cpp)
target_link_libraries(test_bart PRIVATE faststochtree GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_bart)
