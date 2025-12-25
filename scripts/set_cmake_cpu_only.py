import os

cmake_content = r"""cmake_minimum_required(VERSION 3.18)
project(MonadTwoAsset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)  # Turing, Ampere, Ada
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
add_compile_definitions(MONAD_GPU)

# Include Directories
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/Zigen/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/Sokudo/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/MathUniverse/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Standalone Executable
add_executable(RunTwoAsset "src/main_two_asset.cpp" "src/backend/cpu/CpuBackend.cpp" "src/backend/gpu/GpuBackend.cu")
add_executable(TestBackend "src/test_backend.cpp" "src/backend/cpu/CpuBackend.cpp")
add_executable(TestInterface "src/test_interface.cpp")

# Optional: OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(RunTwoAsset PRIVATE OpenMP::OpenMP_CXX)
endif()
"""

with open("CMakeLists.txt", "w") as f:
    f.write(cmake_content)

print("Generated CMakeLists.txt for CPU-only RunTwoAsset")
