# Metal GPU benchmark — only built when -DGPU_METAL=ON (macOS only).
find_library(METAL_LIB     Metal      REQUIRED)
find_library(FOUNDATION_LIB Foundation REQUIRED)

add_executable(gpu_bench
    ../bench/gpu_bench.cpp
    GFRAccel.cpp
    MetalContext.mm
)

set_source_files_properties(MetalContext.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++"
)

target_include_directories(gpu_bench PRIVATE . ../include)
target_link_libraries(gpu_bench PRIVATE faststochtree ${METAL_LIB} ${FOUNDATION_LIB})
target_compile_features(gpu_bench PRIVATE cxx_std_17)
