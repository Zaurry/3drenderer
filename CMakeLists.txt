cmake_minimum_required(VERSION 3.20)

project(cpu_3d_renderer VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin")

find_package(Eigen3 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(
        eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(eigen3)
    if(NOT eigen3_POPULATED)
        FetchContent_Populate(eigen3)
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${eigen3_SOURCE_DIR}"
    )
endif()

add_library(renderer_core STATIC
    src/acceleration/bvh.cpp
    src/core/image.cpp
    src/interactive/orbit_camera_controller.cpp
    src/render/depth_buffer.cpp
    src/render/framebuffer.cpp
    src/render/pathtracer/pathtracer_renderer.cpp
    src/render/rasterizer/rasterizer_renderer.cpp
    src/render/raytracer/raytracer_renderer.cpp
    src/sampling/sampler.cpp
    src/scene/camera.cpp
    src/scene/mesh.cpp
    src/scene/obj_loader.cpp
    src/scene/scene_asset_loader.cpp
    src/scene/scene.cpp
)
target_include_directories(renderer_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader"
)
target_link_libraries(renderer_core PUBLIC Eigen3::Eigen)

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
