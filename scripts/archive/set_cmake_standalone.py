
import os

cmake_content = r"""cmake_minimum_required(VERSION 3.18)
project(MonadTwoAssetCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas)
find_package(OpenMP)

# Include Directories
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/Zigen/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/Sokudo/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/MathUniverse/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Define Sources Explicitly for the App
set(APP_SOURCES
    "src/main_two_asset.cpp"
    "src/gpu/CudaKernels.cu"
)

# Add Executable directly (No MonadLib)
add_executable(RunTwoAsset ${APP_SOURCES})

# Definitions
target_compile_definitions(RunTwoAsset PUBLIC ZIGEN_USE_CUDA)

# Formatting/Warnings if needed (optional)
# target_compile_options(RunTwoAsset PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3>)

# Link Libraries
target_link_libraries(RunTwoAsset 
    PRIVATE 
    CUDA::cudart 
    CUDA::cublas
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(RunTwoAsset PRIVATE OpenMP::OpenMP_CXX)
endif()
"""

with open("CMakeLists.txt", "w") as f:
    f.write(cmake_content)

print("Generated standalone CMakeLists.txt for RunTwoAsset")
