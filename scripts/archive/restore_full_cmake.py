
import os

content = r"""cmake_minimum_required(VERSION 3.18)
project(MonadTwoAssetCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Find CUDA Toolkit to get CUDA::cudart
# Find CUDA Toolkit with components
find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas)

# Find OpenMP for Zigen acceleration
find_package(OpenMP)
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/Zigen/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/Sokudo/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/MathUniverse/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Collect Sources for Library
file(GLOB_RECURSE MONAD_SOURCES "src/*.cpp" "src/gpu/*.cu")
file(GLOB_RECURSE ZIGEN_SOURCES "3rdparty/Zigen/src/*.cpp" "3rdparty/Zigen/src/*.cu")

# Filter out all main files and problematic headers/sources
list(FILTER MONAD_SOURCES EXCLUDE REGEX "src/main_.*\\.cpp")
list(FILTER MONAD_SOURCES EXCLUDE REGEX "src/test_.*\\.cpp")
list(FILTER MONAD_SOURCES EXCLUDE REGEX "src/python_bindings.*\\.cpp")
list(FILTER MONAD_SOURCES EXCLUDE REGEX "src/UniversalEngine.cpp")

# Core Library
add_library(MonadLib STATIC ${MONAD_SOURCES} ${ZIGEN_SOURCES})
target_compile_definitions(MonadLib PUBLIC ZIGEN_USE_CUDA)
target_link_libraries(MonadLib PUBLIC CUDA::cudart CUDA::cublas)

# Zigen Verification Targets
add_executable(ZigenCheck "src/test_zigen_integration.cpp")
target_link_libraries(ZigenCheck PRIVATE MonadLib CUDA::cudart)

add_executable(MarketClearingDemo "src/main_two_asset.cpp")
target_link_libraries(MarketClearingDemo PRIVATE MonadLib CUDA::cudart CUDA::cublas)

add_executable(HankSSZigen "src/test_hank_ss_zigen.cpp")
target_link_libraries(HankSSZigen PRIVATE MonadLib CUDA::cudart CUDA::cublas)

add_executable(ZigenCudaBasic "src/test_zigen_cuda_basic.cpp")
target_link_libraries(ZigenCudaBasic PRIVATE MonadLib CUDA::cudart CUDA::cublas)

add_executable(HybridHankTest "src/test_hybrid_hank.cpp")
target_link_libraries(HybridHankTest PRIVATE MonadLib CUDA::cudart CUDA::cublas)

# Link OpenMP if available (for Zigen acceleration)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ZigenCheck PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(MarketClearingDemo PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(HankSSZigen PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - Zigen will use parallel execution")
else()
    message(WARNING "OpenMP not found - Zigen will run single-threaded")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(ZigenCheck PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
    target_compile_options(MarketClearingDemo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
    target_compile_options(HankSSZigen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
endif()
"""

with open("CMakeLists.txt", "w") as f:
    f.write(content)
print("Restored CMakeLists.txt with Includes")
