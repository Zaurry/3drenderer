cmake_minimum_required(VERSION 3.24)

project(renderer VERSION 0.2.0 LANGUAGES C CXX)

include(FetchContent)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

set(FASTGLTF_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_DOCS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_DEPRECATED_EXT ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    fastgltf
    URL https://github.com/spnda/fastgltf/archive/refs/tags/v0.9.0.tar.gz
    URL_HASH SHA256=0bb564e127b14c22f062db50f89381dd2e0a20dbaf4987ca138a4ae8728712f9
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(fastgltf)

set(TINYEXR_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)
set(TINYEXR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    tinyexr
    URL https://github.com/syoyo/tinyexr/archive/refs/tags/v3.2.0.tar.gz
    URL_HASH SHA256=df2bd61124a35d8138f8b0bc22418a1d4fe33622c818e0e022f5522afc0821b0
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(tinyexr)

option(RENDERER_NATIVE_ARCH "Enable native CPU instruction tuning" OFF)
option(RENDERER_BUILD_VIEWER "Build the SDL3/OpenGL interactive viewer" ON)
option(RENDERER_BUILD_BENCHMARKS "Build developer-only benchmark tools" OFF)

if(RENDERER_BUILD_BENCHMARKS AND NOT RENDERER_BUILD_VIEWER)
    message(FATAL_ERROR
        "RENDERER_BUILD_BENCHMARKS requires RENDERER_BUILD_VIEWER for the OpenGL timing path")
endif()

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
            set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
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

if(RENDERER_BUILD_VIEWER)
    find_package(SDL3 CONFIG QUIET)
    if(NOT TARGET SDL3::SDL3)
        set(SDL_SHARED OFF CACHE BOOL "" FORCE)
        set(SDL_STATIC ON CACHE BOOL "" FORCE)
        set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
        set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
        set(SDL_DIALOG ON CACHE BOOL "" FORCE)
        set(SDL_GPU OFF CACHE BOOL "" FORCE)
        set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
        set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
        set(SDL_OPENGL ON CACHE BOOL "" FORCE)
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

    find_package(OpenGL REQUIRED)

    FetchContent_Declare(
        imgui
        URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.8-docking.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(imgui)

    FetchContent_Declare(
        imguizmo
        URL https://github.com/CedricGuillemet/ImGuizmo/archive/dc25afb98bc3ebe00dfc9a23ba7235fead2ccb1d.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(imguizmo)
    target_include_directories(imguizmo SYSTEM PUBLIC "${imgui_SOURCE_DIR}")
    target_link_libraries(imguizmo PUBLIC renderer_imgui)

    add_library(renderer_glad STATIC external/glad/src/gl.c)
    target_include_directories(renderer_glad SYSTEM PUBLIC external/glad/include)
    target_link_libraries(renderer_glad PUBLIC OpenGL::GL)

    add_library(renderer_imgui STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
    )
    target_include_directories(renderer_imgui SYSTEM PUBLIC
        "${imgui_SOURCE_DIR}"
        "${imgui_SOURCE_DIR}/backends"
    )
    target_link_libraries(renderer_imgui PUBLIC ${RENDERER_SDL_TARGET} OpenGL::GL)

endif()

set(RENDERER_SCENE_SOURCES
    src/acceleration/bvh.cpp
    src/core/io/atomic_file.cpp
    src/core/image.cpp
    src/render/pbr.cpp
    src/render/scene_intersector.cpp
    src/sampling/sampler.cpp
    src/scene/camera.cpp
    src/scene/environment.cpp
    src/scene/gltf_loader.cpp
    src/scene/instanced_scene.cpp
    src/scene/material_evaluator.cpp
    src/scene/mesh.cpp
    src/scene/obj_loader.cpp
    src/scene/scene_asset_loader.cpp
    src/scene/scene_document.cpp
    src/scene/scene.cpp
    src/scene/texture.cpp
)

set(RENDERER_RENDER_COMMON_SOURCES
    src/render/display_settings.cpp
    src/render/framebuffer.cpp
)

set(RENDERER_INTERACTIVE_SOURCES
    src/interactive/frame_rate_counter.cpp
    src/interactive/free_camera_controller.cpp
    src/interactive/orbit_camera_controller.cpp
    src/interactive/viewer_session.cpp
    src/render/interactive/render_mode.cpp
    src/render/interactive/path_interactive_session.cpp
)

if(RENDERER_HAS_CUDA)
    set(RENDERER_CUDA_SOURCES
        src/render/pathtracer/cuda_device_context.cu
        src/render/pathtracer/cuda_pathtracer.cu)
else()
    set(RENDERER_CUDA_SOURCES
        src/render/pathtracer/cuda_device_context_stub.cpp
        src/render/pathtracer/cuda_pathtracer_stub.cpp)
endif()

set(RENDERER_PUBLIC_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader"
    "${tinyexr_SOURCE_DIR}"
)

add_library(renderer_scene STATIC ${RENDERER_SCENE_SOURCES})
target_include_directories(renderer_scene
    PUBLIC
        ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES}
)
target_link_libraries(renderer_scene PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    fastgltf::fastgltf
    tinyexr)

add_library(renderer_render_common STATIC ${RENDERER_RENDER_COMMON_SOURCES})
target_include_directories(renderer_render_common PUBLIC
    ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
target_link_libraries(renderer_render_common PUBLIC renderer_scene)

add_library(renderer_cuda STATIC ${RENDERER_CUDA_SOURCES})
target_include_directories(renderer_cuda PUBLIC
    ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
target_link_libraries(renderer_cuda PUBLIC
    renderer_scene
    renderer_render_common)
if(RENDERER_HAS_CUDA)
    target_link_libraries(renderer_cuda PRIVATE CUDA::cudart)
    set_target_properties(renderer_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_compile_definitions(renderer_cuda PUBLIC RENDERER_HAS_CUDA=1)
else()
    target_compile_definitions(renderer_cuda PUBLIC RENDERER_HAS_CUDA=0)
endif()

add_library(renderer_interactive STATIC ${RENDERER_INTERACTIVE_SOURCES})
target_include_directories(renderer_interactive PUBLIC
    ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
target_link_libraries(renderer_interactive PUBLIC
    renderer_scene
    renderer_render_common
    renderer_cuda)

add_library(renderer_core INTERFACE)
target_link_libraries(renderer_core INTERFACE
    renderer_scene
    renderer_render_common
    renderer_cuda
    renderer_interactive)

set(RENDERER_CORE_SOURCES
    ${RENDERER_SCENE_SOURCES}
    ${RENDERER_RENDER_COMMON_SOURCES}
    ${RENDERER_INTERACTIVE_SOURCES}
    ${RENDERER_CUDA_SOURCES}
)

add_executable(renderer src/main.cpp)
target_link_libraries(renderer PRIVATE renderer_core)

if(RENDERER_BUILD_VIEWER)
    set(RENDERER_OPENGL_SOURCES
        src/platform/opengl/gl_shader_program.cpp
        src/render/opengl/opengl_raster_renderer.cpp
        src/render/interactive/viewer_render_backend.cpp
    )
    if(RENDERER_HAS_CUDA)
        list(APPEND RENDERER_OPENGL_SOURCES
            src/platform/opengl/cuda_opengl_interop.cu
        )
    else()
        list(APPEND RENDERER_OPENGL_SOURCES
            src/platform/opengl/cuda_opengl_interop_stub.cpp
        )
    endif()

    add_library(renderer_opengl STATIC ${RENDERER_OPENGL_SOURCES})
    target_include_directories(renderer_opengl PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(renderer_opengl PUBLIC renderer_core renderer_glad)
    if(RENDERER_HAS_CUDA)
        target_link_libraries(renderer_opengl PRIVATE CUDA::cudart)
        set_target_properties(renderer_opengl PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )
    endif()

    add_executable(viewer
        src/interactive/viewer_ui.cpp
        src/platform/sdl/sdl_display_backend.cpp
        src/viewer_main.cpp
    )
    target_link_libraries(viewer PRIVATE
        renderer_core
        renderer_opengl
        renderer_imgui
        imguizmo
        renderer_glad
        ${RENDERER_SDL_TARGET}
    )
    add_custom_command(TARGET viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/opengl"
            "$<TARGET_FILE_DIR:viewer>/shaders/opengl"
    )
endif()

add_executable(renderer_tests tests/renderer_tests.cpp tests/test_main.cpp)
target_link_libraries(renderer_tests PRIVATE renderer_core)

add_executable(scene_tests tests/scene_tests.cpp tests/test_main.cpp)
target_link_libraries(scene_tests PRIVATE renderer_scene)

add_executable(sampling_tests tests/sampling_tests.cpp tests/test_main.cpp)
target_link_libraries(sampling_tests PRIVATE renderer_scene)

add_executable(document_tests tests/document_tests.cpp tests/test_main.cpp)
target_link_libraries(document_tests PRIVATE renderer_scene)

add_executable(opengl_contract_tests tests/opengl_contract_tests.cpp tests/test_main.cpp)
target_include_directories(opengl_contract_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(opengl_contract_tests PRIVATE renderer_scene)

add_executable(opengl_shader_lint_tests tests/opengl_shader_lint_tests.cpp tests/test_main.cpp)
target_include_directories(opengl_shader_lint_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(opengl_shader_lint_tests PRIVATE renderer_scene)
target_compile_definitions(opengl_shader_lint_tests PRIVATE
    RENDERER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(cuda_contract_tests tests/cuda_contract_tests.cpp tests/test_main.cpp)
target_link_libraries(cuda_contract_tests PRIVATE renderer_cuda)

set(RENDERER_TEST_TARGETS
    renderer_tests
    scene_tests
    sampling_tests
    document_tests
    opengl_contract_tests
    opengl_shader_lint_tests
    cuda_contract_tests)

if(RENDERER_BUILD_BENCHMARKS)
    add_library(renderer_benchmark_common STATIC
        src/benchmark/benchmark_framework.cpp
    )
    target_include_directories(renderer_benchmark_common PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader"
        "${tinyexr_SOURCE_DIR}"
    )
    target_link_libraries(renderer_benchmark_common PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
    )

    add_executable(viewer_benchmark
        tools/viewer_benchmark_main.cpp
    )
    target_link_libraries(viewer_benchmark PRIVATE
        renderer_benchmark_common
        renderer_core
        renderer_opengl
        renderer_glad
        ${RENDERER_SDL_TARGET}
    )
    target_compile_definitions(viewer_benchmark PRIVATE
        RENDERER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        RENDERER_BUILD_TYPE="$<CONFIG>"
    )

    add_executable(benchmark_tests tests/benchmark_tests.cpp tests/test_main.cpp)
    target_link_libraries(benchmark_tests PRIVATE
        renderer_benchmark_common
        renderer_core
    )
    target_compile_definitions(benchmark_tests PRIVATE
        RENDERER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME benchmark_tests COMMAND benchmark_tests)
    set_tests_properties(benchmark_tests PROPERTIES SKIP_RETURN_CODE 77)

    if(RENDERER_HAS_CUDA)
        add_library(renderer_core_diagnostics STATIC ${RENDERER_CORE_SOURCES})
        target_include_directories(renderer_core_diagnostics
            PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
                "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
                "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader"
                "${tinyexr_SOURCE_DIR}"
        )
        target_link_libraries(renderer_core_diagnostics PUBLIC
            Eigen3::Eigen
            nlohmann_json::nlohmann_json
            fastgltf::fastgltf
            tinyexr
        )
        target_link_libraries(renderer_core_diagnostics PRIVATE CUDA::cudart)
        target_compile_definitions(renderer_core_diagnostics PUBLIC
            RENDERER_HAS_CUDA=1
            RENDERER_BENCHMARK_DIAGNOSTICS=1
        )
        set_target_properties(renderer_core_diagnostics PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )

        add_executable(viewer_benchmark_diagnostics
            tools/viewer_benchmark_main.cpp
        )
        target_link_libraries(viewer_benchmark_diagnostics PRIVATE
            renderer_benchmark_common
            renderer_core_diagnostics
            CUDA::cudart
        )
        target_compile_definitions(viewer_benchmark_diagnostics PRIVATE
            RENDERER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
            RENDERER_BUILD_TYPE="$<CONFIG>"
            RENDERER_BENCHMARK_DIAGNOSTICS=1
        )

        add_executable(benchmark_diagnostics_tests
            tests/benchmark_diagnostics_tests.cpp
            tests/test_main.cpp
        )
        target_link_libraries(benchmark_diagnostics_tests PRIVATE
            renderer_core_diagnostics
            CUDA::cudart
        )
        target_compile_definitions(benchmark_diagnostics_tests PRIVATE
            RENDERER_BENCHMARK_DIAGNOSTICS=1
        )
        add_test(
            NAME benchmark_diagnostics_tests
            COMMAND benchmark_diagnostics_tests)
        set_tests_properties(benchmark_diagnostics_tests PROPERTIES
            SKIP_RETURN_CODE 77)
    endif()
endif()

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

set(RENDERER_NATIVE_TARGETS
    renderer_scene
    renderer_render_common
    renderer_cuda
    renderer_interactive
    renderer
    ${RENDERER_TEST_TARGETS})
if(RENDERER_BUILD_VIEWER)
    list(APPEND RENDERER_NATIVE_TARGETS renderer_opengl viewer)
endif()
if(RENDERER_BUILD_BENCHMARKS)
    list(APPEND RENDERER_NATIVE_TARGETS
        renderer_benchmark_common viewer_benchmark benchmark_tests)
    if(RENDERER_HAS_CUDA)
        list(APPEND RENDERER_NATIVE_TARGETS
            renderer_core_diagnostics viewer_benchmark_diagnostics
            benchmark_diagnostics_tests)
    endif()
endif()
foreach(target_name IN LISTS RENDERER_NATIVE_TARGETS)
    target_enable_native_arch(${target_name})
endforeach()

enable_testing()
add_test(NAME renderer_tests COMMAND renderer_tests)
add_test(NAME scene_tests COMMAND scene_tests)
add_test(NAME sampling_tests COMMAND sampling_tests)
add_test(NAME document_tests COMMAND document_tests)
add_test(NAME opengl_contract_tests COMMAND opengl_contract_tests)
add_test(NAME opengl_shader_lint_tests COMMAND opengl_shader_lint_tests)
add_test(NAME cuda_contract_tests COMMAND cuda_contract_tests)

# The shared runner exits 77 when every executed test was skipped (e.g. all
# CUDA tests on a no-CUDA build), so CTest can report "Skipped" instead of a
# misleading green pass.
foreach(test_target IN LISTS RENDERER_TEST_TARGETS)
    set_tests_properties(${test_target} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

function(target_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8 /bigobj>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-permissive-,-utf-8,/bigobj>
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
target_enable_warnings(renderer_scene)
target_enable_warnings(renderer_render_common)
target_enable_warnings(renderer_cuda)
target_enable_warnings(renderer_interactive)
foreach(test_target IN LISTS RENDERER_TEST_TARGETS)
    target_enable_warnings(${test_target})
endforeach()
if(RENDERER_BUILD_BENCHMARKS)
    target_enable_warnings(renderer_benchmark_common)
    target_enable_warnings(viewer_benchmark)
    target_enable_warnings(benchmark_tests)
    if(RENDERER_HAS_CUDA)
        target_enable_warnings(renderer_core_diagnostics)
        target_enable_warnings(viewer_benchmark_diagnostics)
        target_enable_warnings(benchmark_diagnostics_tests)
    endif()
endif()
if(RENDERER_BUILD_VIEWER)
    target_enable_warnings(renderer_opengl)
    target_enable_warnings(viewer)
endif()
