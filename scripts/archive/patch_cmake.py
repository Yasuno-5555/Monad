import os

def patch_cmake():
    cmake_path = "CMakeLists.txt"
    if not os.path.exists(cmake_path):
        print("CMakeLists.txt not found")
        return

    with open(cmake_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    added_subdir = False
    
    for line in lines:
        # Add subdirectory early (e.g. after include_directories)
        if "include_directories(${CMAKE_SOURCE_DIR}/src)" in line and not added_subdir:
            new_lines.append(line)
            new_lines.append("\n# MathUniverse\n")
            new_lines.append("add_subdirectory(3rdparty/mathuniverse/MathUniverse)\n")
            added_subdir = True
            continue
            
        # Link MathUniverse to MarketClearingDemo
        if "target_link_libraries(MarketClearingDemo PRIVATE MonadLib CUDA::cudart CUDA::cublas)" in line:
            new_lines.append(line.replace("CUDA::cublas)", "CUDA::cublas MathUniverse)"))
            continue
            
        new_lines.append(line)
        
    with open(cmake_path, 'w') as f:
        f.writelines(new_lines)
    print("Patched CMakeLists.txt")

if __name__ == "__main__":
    patch_cmake()
