
import os

with open("CMakeLists.txt", "a") as f:
    f.write("\n# DEBUG TARGETS \n")
    f.write("add_library(DebugKernel STATIC src/test_kernel.cpp)\n")
    f.write("target_include_directories(DebugKernel PUBLIC src 3rdparty/eigen 3rdparty/Zigen/include 3rdparty)\n")
    f.write("add_library(DebugGraph STATIC 3rdparty/Zigen/src/IR/Graph.cpp)\n")
    f.write("target_include_directories(DebugGraph PUBLIC src 3rdparty/eigen 3rdparty/Zigen/include 3rdparty)\n")
    
print("Appended Debug Targets")
