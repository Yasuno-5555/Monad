cmake_minimum_required(VERSION 3.10)
project(MonadEngine VERSION 1.1)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

include(FetchContent)

# nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Eigen
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(eigen)

include_directories(${eigen_SOURCE_DIR})



add_executable(SolverTest
    src/grid/grid_generator.cpp
    src/main_analytical.cpp
)

add_executable(Phase3Test
    src/main_phase3.cpp
)
# Link nlohmann_json::nlohmann_json
target_link_libraries(Phase3Test PRIVATE nlohmann_json::nlohmann_json)

# --- Phase 5: Monad Engine (Production CLI) ---
add_executable(MonadEngine src/main_engine.cpp)
target_link_libraries(MonadEngine PRIVATE nlohmann_json::nlohmann_json)

# Copy executable to build root for easier access
if(WIN32)
    add_custom_command(TARGET MonadEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:MonadEngine>
        ${CMAKE_BINARY_DIR}/MonadEngine.exe
    )
endif()
