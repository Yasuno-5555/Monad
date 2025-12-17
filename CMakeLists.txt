cmake_minimum_required(VERSION 3.18)
project(MonadTwoAssetCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Find CUDA Toolkit to get CUDA::cudart
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/build_phase3/_deps/eigen-src)

# Collect Sources
file(GLOB_RECURSE ALL_SOURCES "src/*.cpp" "src/*.hpp" "src/gpu/*.cu" "src/gpu/*.hpp")

# Filter out files that require extra dependencies (Python, JSON) or are alternative mains
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_analytical.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_engine.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_phase3.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_trivial.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/python_bindings.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/python_bindings.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/test_multidim.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_debug_dual.cpp")

add_executable(MonadTwoAssetCUDA ${ALL_SOURCES})

# Link against CUDA Runtime
target_link_libraries(MonadTwoAssetCUDA PRIVATE CUDA::cudart)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(MonadTwoAssetCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
endif()
