cmake_minimum_required(VERSION 3.20)

project(cpu_3d_renderer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin")

add_library(renderer_core STATIC
    src/acceleration/bvh.cpp
    src/core/image.cpp
    src/render/pathtracer/pathtracer_renderer.cpp
    src/render/raytracer/raytracer_renderer.cpp
    src/sampling/sampler.cpp
    src/scene/camera.cpp
    src/scene/mesh.cpp
    src/scene/scene.cpp
)
target_include_directories(renderer_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader"
)

add_executable(renderer src/main.cpp)
target_link_libraries(renderer PRIVATE renderer_core)

add_executable(renderer_tests tests/renderer_tests.cpp)
target_link_libraries(renderer_tests PRIVATE renderer_core)

enable_testing()
add_test(NAME renderer_tests COMMAND renderer_tests)

function(target_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

target_enable_warnings(renderer)
target_enable_warnings(renderer_core)
target_enable_warnings(renderer_tests)
