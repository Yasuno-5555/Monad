
import os

cmake_content = r"""cmake_minimum_required(VERSION 3.18)
project(TestHeader LANGUAGES CXX CUDA)

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

add_executable(TestHeader "src/test_solver.cpp")
target_compile_definitions(TestHeader PUBLIC ZIGEN_USE_CUDA MONAD_GPU)
target_link_libraries(TestHeader PRIVATE CUDA::cudart CUDA::cublas)
"""

with open("CMakeLists.txt", "w") as f:
    f.write(cmake_content)

print("Generated CMakeLists.txt for TestHeader")
