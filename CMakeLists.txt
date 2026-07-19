cmake_minimum_required(VERSION 3.21)

project(cpu_3d_renderer VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RENDERER_NATIVE_ARCH "Enable native CPU instruction tuning" OFF)

set(RENDERER_CUDA "AUTO" CACHE STRING "CUDA path tracer support: AUTO, ON, or OFF")
set_property(CACHE RENDERER_CUDA PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${RENDERER_CUDA}" RENDERER_CUDA_MODE)
if(NOT RENDERER_CUDA_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "RENDERER_CUDA must be AUTO, ON, or OFF")
endif()

set(RENDERER_HAS_CUDA OFF)
if(NOT RENDERER_CUDA_MODE STREQUAL "OFF")
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures")
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(RENDERER_HAS_CUDA ON)
    elseif(RENDERER_CUDA_MODE STREQUAL "ON")
        message(FATAL_ERROR "RENDERER_CUDA=ON but no CUDA compiler was found")
    endif()
endif()

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
        SOURCE_SUBDIR _renderer_fetchcontent_only
    )
    FetchContent_MakeAvailable(eigen3)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen SYSTEM INTERFACE
        "${eigen3_SOURCE_DIR}"
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
    set(SDL_RENDER ON CACHE BOOL "" FORCE)
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

FetchContent_Declare(
    imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.8.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(renderer_imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer3.cpp"
)
target_include_directories(renderer_imgui SYSTEM PUBLIC
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
)
target_link_libraries(renderer_imgui PUBLIC ${RENDERER_SDL_TARGET})

set(RENDERER_CORE_SOURCES
    src/acceleration/bvh.cpp
    src/core/image.cpp
    src/interactive/frame_rate_counter.cpp
    src/interactive/free_camera_controller.cpp
    src/interactive/orbit_camera_controller.cpp
    src/render/depth_buffer.cpp
    src/render/display_settings.cpp
    src/render/framebuffer.cpp
    src/render/interactive/path_interactive_session.cpp
    src/render/interactive/raster_interactive_session.cpp
    src/render/interactive/ray_interactive_session.cpp
    src/render/pathtracer/pathtracer_renderer.cpp
    src/render/pathtracer/path_backend.cpp
    src/render/rasterizer/raster_geometry.cpp
    src/render/rasterizer/rasterizer_renderer.cpp
    src/render/raytracer/raytracer_renderer.cpp
    src/render/scene_intersector.cpp
    src/sampling/sampler.cpp
    src/scene/camera.cpp
    src/scene/material_evaluator.cpp
    src/scene/mesh.cpp
    src/scene/obj_loader.cpp
    src/scene/scene_asset_loader.cpp
    src/scene/scene.cpp
    src/scene/texture.cpp
)

if(RENDERER_HAS_CUDA)
    list(APPEND RENDERER_CORE_SOURCES src/render/pathtracer/cuda_pathtracer.cu)
else()
    list(APPEND RENDERER_CORE_SOURCES src/render/pathtracer/cuda_pathtracer_stub.cpp)
endif()

add_library(renderer_core STATIC ${RENDERER_CORE_SOURCES})
target_include_directories(renderer_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader"
)
target_link_libraries(renderer_core PUBLIC Eigen3::Eigen)
if(RENDERER_HAS_CUDA)
    target_link_libraries(renderer_core PRIVATE CUDA::cudart)
    set_target_properties(renderer_core PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_compile_definitions(renderer_core PUBLIC RENDERER_HAS_CUDA=1)
else()
    target_compile_definitions(renderer_core PUBLIC RENDERER_HAS_CUDA=0)
endif()

add_executable(renderer src/main.cpp)
target_link_libraries(renderer PRIVATE renderer_core)

add_executable(viewer
    src/interactive/viewer_ui.cpp
    src/platform/sdl/sdl_display_backend.cpp
    src/viewer_main.cpp
)
target_link_libraries(viewer PRIVATE renderer_core renderer_imgui ${RENDERER_SDL_TARGET})

add_executable(renderer_tests tests/renderer_tests.cpp)
target_link_libraries(renderer_tests PRIVATE renderer_core)

function(target_enable_native_arch target_name)
    if(RENDERER_NATIVE_ARCH)
        if(MSVC)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>
            )
        else()
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-march=native>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
            )
        endif()
    endif()
endfunction()

foreach(target_name IN ITEMS renderer_core renderer viewer renderer_tests)
    target_enable_native_arch(${target_name})
endforeach()

enable_testing()
add_test(NAME renderer_tests COMMAND renderer_tests)

function(target_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-permissive-,-utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=68>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4068,/wd4211>
        )
    else()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
        )
    endif()
endfunction()

target_enable_warnings(renderer)
target_enable_warnings(renderer_core)
target_enable_warnings(viewer)
target_enable_warnings(renderer_tests)
