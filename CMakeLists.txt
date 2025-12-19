cmake_minimum_required(VERSION 3.18)
project(MonadTwoAssetCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty/eigen
    ${CMAKE_SOURCE_DIR}/src
)

file(GLOB_RECURSE ALL_SOURCES "src/*.cpp" "src/*.hpp" "src/gpu/*.cu" "src/gpu/*.hpp")

# --- FILTERING ---
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/python_bindings.*")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/test_bindings.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_analytical.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_engine.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_phase3.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_trivial.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/main_debug_dual.cpp")
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/test_.*")

add_executable(MonadTwoAssetCUDA ${ALL_SOURCES})
target_link_libraries(MonadTwoAssetCUDA PRIVATE CUDA::cudart)

# Compile Options Fixing
if(MSVC)
    # Apply /utf-8 to C++ files directly
    target_compile_options(MonadTwoAssetCUDA PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    
    # Apply /utf-8 to CUDA files via -Xcompiler
    target_compile_options(MonadTwoAssetCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(MonadTwoAssetCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
endif()
