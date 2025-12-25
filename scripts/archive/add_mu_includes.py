
import os

with open("CMakeLists.txt", "r") as f:
    content = f.read()

# Add mathuniverse paths
if "include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/Sokudo/include)" not in content:
    content = content.replace("include_directories(${CMAKE_SOURCE_DIR}/3rdparty)", 
        "include_directories(${CMAKE_SOURCE_DIR}/3rdparty)\ninclude_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/Sokudo/include)\ninclude_directories(${CMAKE_SOURCE_DIR}/3rdparty/mathuniverse/MathUniverse/include)")

with open("CMakeLists.txt", "w") as f:
    f.write(content)
print("Added MathUniverse include paths")
