import os

# DEFINE ABSOLUTE PROJECT ROOT
project_root = "E:/Projects/MonadLab"

# VERIFIED ABSOLUTE PATHS (Forward Slashes)
monad_src = [
    f"{project_root}/src/backend/gpu/GpuBackend.cu",
    f"{project_root}/src/backend/cpu/CpuBackend.cpp"
]

zigen_src = [
    f"{project_root}/3rdparty/mathuniverse/Zigen/src/CUDA/Geometry.cu",
    f"{project_root}/3rdparty/mathuniverse/Zigen/src/CUDA/LinearAlgebra.cu",
    f"{project_root}/3rdparty/mathuniverse/Zigen/src/IR/CudaBackend.cu",
    f"{project_root}/3rdparty/mathuniverse/Zigen/src/IR/FusionMatcher.cpp",
    f"{project_root}/3rdparty/mathuniverse/Zigen/src/IR/Graph.cpp"
]

demo_src_2 = f"{project_root}/src/main_two_asset.cpp"
demo_src_3 = f"{project_root}/src/main_three_asset.cpp"
demo_src_4 = f"{project_root}/src/main_three_asset_full.cpp"

lib_sources_joined = " ".join([f'"{s}"' for s in (monad_src + zigen_src)])
monad_only_src = " ".join([f'"{s}"' for s in monad_src])

content = f"""cmake_minimum_required(VERSION 3.12)
project(MonadMultiAssetCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas)
find_package(OpenMP)

# ABSOLUTE INCLUDE PATHS
include_directories("{project_root}/3rdparty/eigen")
include_directories("{project_root}/3rdparty/mathuniverse/Zigen/include")
include_directories("{project_root}/3rdparty/mathuniverse/mathuniverse/include")
include_directories("{project_root}/3rdparty")
include_directories("{project_root}/src")

add_library(MonadLib STATIC {lib_sources_joined})
target_compile_definitions(MonadLib PUBLIC ZIGEN_USE_CUDA MONAD_GPU)
target_link_libraries(MonadLib PUBLIC CUDA::cudart CUDA::cublas)

# New Isolated Target for Debugging
add_library(MonadCore STATIC {monad_only_src})
target_compile_definitions(MonadCore PUBLIC MONAD_GPU)
target_link_libraries(MonadCore PUBLIC CUDA::cudart CUDA::cublas)

add_executable(MarketClearingDemo "{demo_src_2}")
target_link_libraries(MarketClearingDemo PRIVATE MonadLib CUDA::cudart CUDA::cublas)

add_executable(ThreeAssetDemo "{demo_src_3}")
target_link_libraries(ThreeAssetDemo PRIVATE MonadCore CUDA::cudart CUDA::cublas)

add_executable(ThreeAssetFull "{demo_src_4}")
target_link_libraries(ThreeAssetFull PRIVATE MonadCore CUDA::cudart CUDA::cublas)

if(OpenMP_CXX_FOUND)
    target_link_libraries(MarketClearingDemo PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ThreeAssetDemo PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ThreeAssetFull PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(MarketClearingDemo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
    target_compile_options(ThreeAssetDemo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
    target_compile_options(ThreeAssetFull PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
endif()
"""

with open("CMakeLists.txt", "w", encoding='utf-8') as f:
    f.write(content)

print(f"SUCCESS: Generated CMakeLists.txt with ABSOLUTE PATHS.")
