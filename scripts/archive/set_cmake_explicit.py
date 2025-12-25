
import os

content = r"""cmake_minimum_required(VERSION 3.18)
project(MonadTwoAssetCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas)
find_package(OpenMP)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/Zigen/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/Sokudo/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/MathUniverse/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# EXPLICIT SOURCES FOR DEBUGGING
set(MONAD_SOURCES 
    "src/test_kernel.cpp"
    "src/gpu/CudaKernels.cu"
    "src/gpu/CudaKernelsF32.cu"
    "src/gpu/CudaUtils.hpp"
    "src/kernel/TwoAssetKernel.hpp"
    "src/solver/TwoAssetSolver.hpp"
    "src/aggregator/distribution.cpp"
    # Add CudaBackend.cu if found?
    # "src/gpu/CudaBackend.cu" 
)
# Zigen sources need to be included too?
file(GLOB_RECURSE ZIGEN_SOURCES "3rdparty/Zigen/src/*.cpp" "3rdparty/Zigen/src/*.cu")

add_library(MonadLib STATIC ${MONAD_SOURCES} ${ZIGEN_SOURCES})
target_compile_definitions(MonadLib PUBLIC ZIGEN_USE_CUDA)
target_link_libraries(MonadLib PUBLIC CUDA::cudart CUDA::cublas)

add_executable(MarketClearingDemo "src/main_two_asset.cpp")
target_link_libraries(MarketClearingDemo PRIVATE MonadLib CUDA::cudart CUDA::cublas)
"""

with open("CMakeLists.txt", "w") as f:
    f.write(content)
print("Set CMakeLists with Explicit Sources")
