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

# ── stochtree comparison benchmark ───────────────────────────────────────────
# Builds a second benchmark binary using stochtree (the reference BART impl)
# on identical simulated data for an apples-to-apples speed comparison.
# Fetches stochtree + its submodule dependencies from GitHub on first build
# (~200 MB download; cached under .cmake-fetch-cache/ on subsequent builds).
#
# Enable with:  cmake -DBUILD_STOCHTREE_COMPARE=ON ...
# Local override: cmake -DBUILD_STOCHTREE_COMPARE=ON -DSTOCHTREE_SOURCE_DIR=/path/to/stochtree ...
option(BUILD_STOCHTREE_COMPARE "Build stochtree comparison benchmark" OFF)

if(BUILD_STOCHTREE_COMPARE)
    # Prevent stochtree from building targets we don't need.
    set(BUILD_DEBUG_TARGETS OFF CACHE BOOL "" FORCE)
    set(BUILD_PYTHON        OFF CACHE BOOL "" FORCE)
    set(BUILD_TEST          OFF CACHE BOOL "" FORCE)
    # Apple Clang doesn't ship with OpenMP; disable to avoid build errors on macOS.
    set(USE_OPENMP          OFF CACHE BOOL "" FORCE)

    if(DEFINED STOCHTREE_SOURCE_DIR)
        message(STATUS "stochtree: using local source at ${STOCHTREE_SOURCE_DIR}")
        FetchContent_Declare(stochtree SOURCE_DIR "${STOCHTREE_SOURCE_DIR}")
    else()
        FetchContent_Declare(
            stochtree
            GIT_REPOSITORY    https://github.com/StochasticTree/stochtree.git
            GIT_TAG           main
            GIT_SUBMODULES_RECURSE TRUE
        )
    endif()

    FetchContent_MakeAvailable(stochtree)

    add_executable(stochtree_compare bench/stochtree_compare.cpp)
    target_link_libraries(stochtree_compare PRIVATE stochtree_objs)
    target_include_directories(stochtree_compare PRIVATE
        ${stochtree_SOURCE_DIR}/include
        ${stochtree_SOURCE_DIR}/deps/eigen
        ${stochtree_SOURCE_DIR}/deps/boost_math/include
        ${stochtree_SOURCE_DIR}/deps/fmt/include
        ${stochtree_SOURCE_DIR}/deps/fast_double_parser/include
    )
endif()

add_executable(test_bart tests/test_bart.cpp)
target_link_libraries(test_bart PRIVATE faststochtree GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_bart)
