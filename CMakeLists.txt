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
set_target_properties(faststochtree PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Benchmark executables
add_executable(faststochtree_bart_bench bench/bart_bench.cpp)
target_link_libraries(faststochtree_bart_bench PRIVATE faststochtree)

add_executable(faststochtree_bcf_bench bench/bcf_bench.cpp)
target_link_libraries(faststochtree_bcf_bench PRIVATE faststochtree)

add_executable(cpu_profile bench/cpu_profile.cpp)
target_link_libraries(cpu_profile PRIVATE faststochtree)

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

add_executable(test_tree tests/test_tree.cpp)
target_link_libraries(test_tree PRIVATE faststochtree GTest::gtest_main)

add_executable(test_bcf tests/test_bcf.cpp)
target_link_libraries(test_bcf PRIVATE faststochtree GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_bart)
gtest_discover_tests(test_tree)
gtest_discover_tests(test_bcf)

# Metal GPU benchmark (macOS only) — opt-in via -DGPU_METAL=ON
option(GPU_METAL "Enable Apple Metal GPU benchmark" OFF)
if(GPU_METAL)
    add_subdirectory(gpu)
endif()

# Python bindings (built by scikit-build-core via: pip install ./python)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python extension" OFF)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_faststochtree
        python/src/faststochtree/_bindings.cpp)
    target_link_libraries(_faststochtree PRIVATE faststochtree)
    target_include_directories(_faststochtree PRIVATE .)

    if(GPU_METAL)
        find_library(METAL_LIB     Metal      REQUIRED)
        find_library(FOUNDATION_LIB Foundation REQUIRED)
        target_sources(_faststochtree PRIVATE
            gpu/MetalContext.mm
            gpu/GFRAccel.cpp)
        set_source_files_properties(gpu/MetalContext.mm PROPERTIES
            COMPILE_FLAGS "-x objective-c++")
        target_compile_definitions(_faststochtree PRIVATE GPU_METAL)
        target_include_directories(_faststochtree PRIVATE gpu)
        target_link_libraries(_faststochtree PRIVATE ${METAL_LIB} ${FOUNDATION_LIB})
    endif()

    install(TARGETS _faststochtree DESTINATION faststochtree)
endif()
