cmake_minimum_required(VERSION 3.20)

project(cpu_3d_renderer VERSION 0.1.0 LANGUAGES C CXX)

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

find_package(SDL3 CONFIG QUIET)
if(NOT TARGET SDL3::SDL3)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
    set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
    set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
    set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
    set(SDL_OPENGL OFF CACHE BOOL "" FORCE)
    set(SDL_OPENGLES OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
    set(SDL_VULKAN OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
    set(local_sdl3_source_dir "${CMAKE_BINARY_DIR}/_deps/sdl3-src")
    if(EXISTS "${local_sdl3_source_dir}/CMakeLists.txt")
        set(FETCHCONTENT_SOURCE_DIR_SDL3 "${local_sdl3_source_dir}" CACHE PATH "" FORCE)
    endif()
    FetchContent_Declare(
        sdl3
        URL https://www.libsdl.org/release/SDL3-3.2.30.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(sdl3)
endif()

set(RENDERER_SDL_TARGET "")
if(TARGET SDL3::SDL3)
    set(RENDERER_SDL_TARGET SDL3::SDL3)
elseif(TARGET SDL3::SDL3-static)
    set(RENDERER_SDL_TARGET SDL3::SDL3-static)
elseif(TARGET SDL3-static)
    set(RENDERER_SDL_TARGET SDL3-static)
endif()

add_library(renderer_core STATIC
    src/acceleration/bvh.cpp
    src/core/image.cpp
    src/interactive/frame_rate_counter.cpp
    src/interactive/orbit_camera_controller.cpp
    src/render/depth_buffer.cpp
    src/render/framebuffer.cpp
    src/render/interactive/path_interactive_session.cpp
    src/render/interactive/raster_interactive_session.cpp
    src/render/interactive/ray_interactive_session.cpp
    src/render/pathtracer/pathtracer_renderer.cpp
    src/render/rasterizer/rasterizer_renderer.cpp
    src/render/raytracer/raytracer_renderer.cpp
    src/sampling/sampler.cpp
    src/scene/camera.cpp
    src/scene/material_evaluator.cpp
    src/scene/mesh.cpp
    src/scene/obj_loader.cpp
    src/scene/scene_asset_loader.cpp
    src/scene/scene.cpp
    src/scene/texture.cpp
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

add_executable(viewer
    src/platform/sdl/sdl_display_backend.cpp
    src/viewer_main.cpp
)
target_link_libraries(viewer PRIVATE renderer_core ${RENDERER_SDL_TARGET})

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
target_enable_warnings(viewer)
target_enable_warnings(renderer_tests)
