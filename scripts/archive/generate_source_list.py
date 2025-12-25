
import os

def generate_cmake():
    root_dir = "src"
    sources = []
    
    exclusion_list = [
        "egm_core.cpp",
        "UniversalEngine.cpp",
        "main_two_asset.cpp",
        "test_", 
        "python_bindings",
        "AnalyticalSolver.cpp",
        "InequalityAnalyzer.cpp"
    ]

    for dirpath, _, filenames in os.walk(root_dir):
        # Skip ssj directory
        if "ssj" in dirpath.replace("\\", "/").split("/"):
            continue

        for f in filenames:
            if f.endswith(".cpp") or f.endswith(".cu"):
                # Check exclusions
                if "ssj/" in dirpath.replace("\\", "/"):
                     continue
                if "benchmark/" in dirpath.replace("\\", "/"):
                     continue
                if "analysis/" in dirpath.replace("\\", "/"):
                     continue
                if any(ex in f for ex in exclusion_list):
                    continue
                if f.startswith("main_"):
                    continue
                
                # Path relative to project root (e.g., src/foo/bar.cpp)
                path = os.path.join(dirpath, f).replace("\\", "/")
                sources.append(f'"{path}"')

    # Add specific test kernel if needed
    # sources.append('"src/test_kernel.cpp"')
    
    print("\nIncluded Sources:")
    for s in sources:
        print(s)

    
    # Generate CMake content
    cmake_content = r"""cmake_minimum_required(VERSION 3.18)
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

set(MONAD_SOURCES 
""" + "\n    ".join(sources) + "\n)\n" + r"""
# Zigen sources need to be included too?
file(GLOB_RECURSE ZIGEN_SOURCES "3rdparty/Zigen/src/*.cpp" "3rdparty/Zigen/src/*.cu")

add_library(MonadLib STATIC ${MONAD_SOURCES} ${ZIGEN_SOURCES})
target_compile_definitions(MonadLib PUBLIC ZIGEN_USE_CUDA)
target_link_libraries(MonadLib PUBLIC CUDA::cudart CUDA::cublas)

add_executable(MarketClearingDemo "src/main_two_asset.cpp")
target_link_libraries(MarketClearingDemo PRIVATE MonadLib CUDA::cudart CUDA::cublas)
"""

    with open("CMakeLists.txt", "w") as f:
        f.write(cmake_content)
    
    print(f"Generated CMakeLists.txt with {len(sources)} sources.")

if __name__ == "__main__":
    generate_cmake()
