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
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
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
        FetchContent_Declare(
            sdl3
            URL https://www.libsdl.org/release/SDL3-3.2.30.tar.gz
            URL_HASH SHA256=4c3b09330d866dc52eb65b66259a6684ad387252ca8c57901b3a2b534eb42e3d
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
        URL_HASH SHA256=ca0653454ed371b7a87e9b0bc29a5d15c9be7f7c0fbe778042fc48c71df1d3d8
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(imgui)

    FetchContent_Declare(
        imguizmo
        URL https://github.com/CedricGuillemet/ImGuizmo/archive/dc25afb98bc3ebe00dfc9a23ba7235fead2ccb1d.tar.gz
        URL_HASH SHA256=bee18d71469b78b4bece9051deafc4834cda9f49f71707d2a7ed49d8623f5309
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
)

if(RENDERER_HAS_CUDA)
    set(RENDERER_CUDA_API_SOURCES
        src/render/pathtracer/cuda_device_context.cu)
    set(RENDERER_CUDA_PATHTRACER_SOURCES
        src/render/pathtracer/cuda_pathtracer.cu)
else()
    set(RENDERER_CUDA_API_SOURCES
        src/render/pathtracer/cuda_device_context_stub.cpp)
    set(RENDERER_CUDA_PATHTRACER_SOURCES
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

# CUDA availability probing and device context (small; stubbed without CUDA).
add_library(renderer_cuda_api STATIC ${RENDERER_CUDA_API_SOURCES})
target_include_directories(renderer_cuda_api PUBLIC
    ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
if(RENDERER_HAS_CUDA)
    target_link_libraries(renderer_cuda_api PUBLIC CUDA::cudart)
    set_target_properties(renderer_cuda_api PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_compile_definitions(renderer_cuda_api PUBLIC RENDERER_HAS_CUDA=1)
else()
    target_compile_definitions(renderer_cuda_api PUBLIC RENDERER_HAS_CUDA=0)
endif()

# The CUDA path tracer proper.
add_library(renderer_cuda_pathtracer STATIC ${RENDERER_CUDA_PATHTRACER_SOURCES})
target_include_directories(renderer_cuda_pathtracer PUBLIC
    ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
target_link_libraries(renderer_cuda_pathtracer PUBLIC
    renderer_scene
    renderer_render_common
    renderer_cuda_api)
if(RENDERER_HAS_CUDA)
    target_link_libraries(renderer_cuda_pathtracer PRIVATE CUDA::cudart)
    set_target_properties(renderer_cuda_pathtracer PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

option(RENDERER_OPTIX "Enable RTRT RT cores and OptiX AI denoising (requires a compatible NVIDIA driver)" ON)
if(RENDERER_HAS_CUDA AND RENDERER_OPTIX)
    # OptiX 9.1.0 headers: RT traversal and AOV/Temporal AOV neural denoising.
    FetchContent_Declare(optix_headers
        URL https://github.com/NVIDIA/optix-dev/archive/f1f6dd803f3159992d248178f6e09421c6eb8b6d.tar.gz
        URL_HASH SHA256=c36d6e52289138685dcce77befb8f966d2761c6840556f6a04945b01700c8ebc
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(optix_headers)
    add_library(renderer_rtrt_ptx OBJECT src/render/optix/optix_programs.cu)
    target_link_libraries(renderer_rtrt_ptx PRIVATE renderer_scene renderer_cuda_api)
    target_include_directories(renderer_rtrt_ptx PRIVATE "${optix_headers_SOURCE_DIR}/include")
    set_target_properties(renderer_rtrt_ptx PROPERTIES CUDA_PTX_COMPILATION ON
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 75-virtual)
    set(RTRT_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rtrt_optix_ptx.h")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    add_custom_command(OUTPUT "${RTRT_PTX_HEADER}"
        COMMAND ${CMAKE_COMMAND} "-DINPUT=$<TARGET_OBJECTS:renderer_rtrt_ptx>" "-DOUTPUT=${RTRT_PTX_HEADER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_optix_ptx.cmake"
        DEPENDS renderer_rtrt_ptx $<TARGET_OBJECTS:renderer_rtrt_ptx> tools/embed_optix_ptx.cmake
        VERBATIM)
    add_library(renderer_rtrt_optix STATIC src/render/optix/optix_backend.cpp "${RTRT_PTX_HEADER}")
    target_include_directories(renderer_rtrt_optix PRIVATE "${optix_headers_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(renderer_rtrt_optix PUBLIC renderer_scene renderer_cuda_api PRIVATE CUDA::cudart)
    add_library(renderer_realtime STATIC src/render/optix/optix_realtime_renderer.cu)
    target_link_libraries(renderer_realtime PRIVATE renderer_rtrt_optix CUDA::cudart)
    target_compile_definitions(renderer_realtime PUBLIC RENDERER_HAS_OPTIX=1)
    set_target_properties(renderer_realtime PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    file(READ "${optix_headers_SOURCE_DIR}/include/optix_stubs.h" OPTIX_STUBS_SOURCE)
    string(FIND "${OPTIX_STUBS_SOURCE}" "*/" OPTIX_NOTICE_END)
    math(EXPR OPTIX_NOTICE_LENGTH "${OPTIX_NOTICE_END}+2")
    string(SUBSTRING "${OPTIX_STUBS_SOURCE}" 0 ${OPTIX_NOTICE_LENGTH} OPTIX_BSD_NOTICE)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/OPTIX_BSD_NOTICE.txt" "${OPTIX_BSD_NOTICE}\n")
else()
    add_library(renderer_realtime STATIC src/render/optix/optix_realtime_renderer_stub.cpp)
    target_compile_definitions(renderer_realtime PUBLIC RENDERER_HAS_OPTIX=0)
endif()
target_include_directories(renderer_realtime PUBLIC ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
target_link_libraries(renderer_realtime PUBLIC renderer_scene renderer_render_common renderer_cuda_api)

# Session adapter over the interactive CUDA renderer, kept separate so UI
# libraries never drag in the path tracer.
add_library(renderer_path_session STATIC
    src/render/interactive/path_interactive_session.cpp)
target_include_directories(renderer_path_session PUBLIC
    ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
target_link_libraries(renderer_path_session PUBLIC
    renderer_cuda_api
    renderer_cuda_pathtracer)

add_library(renderer_interactive STATIC ${RENDERER_INTERACTIVE_SOURCES})
target_include_directories(renderer_interactive PUBLIC
    ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
target_link_libraries(renderer_interactive PUBLIC
    renderer_scene
    renderer_render_common
    renderer_cuda_api)

# Aggregate aliases for binaries that want the whole core.
add_library(renderer_cuda INTERFACE)
target_link_libraries(renderer_cuda INTERFACE
    renderer_scene
    renderer_render_common
    renderer_cuda_api
    renderer_cuda_pathtracer
    renderer_realtime
    renderer_path_session)

add_library(renderer_core INTERFACE)
target_link_libraries(renderer_core INTERFACE
    renderer_scene
    renderer_render_common
    renderer_cuda_api
    renderer_cuda_pathtracer
    renderer_realtime
    renderer_path_session
    renderer_interactive)

if(RENDERER_HAS_CUDA)
    add_library(renderer_ddgi STATIC src/render/ddgi/cuda_ddgi_volume.cu)
    target_link_libraries(renderer_ddgi PRIVATE CUDA::cudart)
    set_target_properties(renderer_ddgi PROPERTIES CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
else()
    add_library(renderer_ddgi STATIC src/render/ddgi/cuda_ddgi_volume_stub.cpp)
endif()
target_include_directories(renderer_ddgi PUBLIC ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
target_link_libraries(renderer_ddgi PUBLIC renderer_scene renderer_render_common renderer_cuda_api)

add_executable(renderer src/main.cpp)
target_link_libraries(renderer PRIVATE renderer_core)

if(RENDERER_BUILD_VIEWER)
    set(RENDERER_OPENGL_SOURCES
        src/platform/opengl/gl_shader_program.cpp
        src/render/opengl/opengl_raster_renderer.cpp
        src/render/opengl/opengl_ddgi_pass.cpp
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
    # Deliberately excludes the path tracer: static libraries may carry
    # unresolved CudaPathInteractiveRenderer/PathInteractiveSession symbols;
    # the final binaries link the (normal or diagnostics) pathtracer variant.
    target_link_libraries(renderer_opengl PUBLIC
        renderer_interactive
        renderer_ddgi
        renderer_realtime
        renderer_cuda_api
        renderer_glad)
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
    # Explicit GPU integration check; excluded from portable CPU-only CTest runs.
    add_executable(opengl_npr_smoke EXCLUDE_FROM_ALL tests/opengl_npr_smoke.cpp)
    target_link_libraries(opengl_npr_smoke PRIVATE
        renderer_core renderer_opengl renderer_glad ${RENDERER_SDL_TARGET})
    target_compile_definitions(opengl_npr_smoke PRIVATE
        RENDERER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
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

add_executable(realtime_tests tests/realtime_tests.cpp tests/test_main.cpp)
target_link_libraries(realtime_tests PRIVATE renderer_cuda)

set(RENDERER_TEST_TARGETS
    renderer_tests
    scene_tests
    sampling_tests
    document_tests
    opengl_contract_tests
    opengl_shader_lint_tests
    cuda_contract_tests
    realtime_tests)

if(RENDERER_BUILD_VIEWER)
    add_executable(viewer_ui_tests tests/viewer_ui_tests.cpp tests/test_main.cpp)
    target_include_directories(viewer_ui_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(viewer_ui_tests PRIVATE renderer_imgui)
    list(APPEND RENDERER_TEST_TARGETS viewer_ui_tests)
    add_executable(opengl_transport_tests
        tests/opengl_transport_tests.cpp tests/test_main.cpp)
    target_link_libraries(opengl_transport_tests PRIVATE
        renderer_core renderer_opengl renderer_glad ${RENDERER_SDL_TARGET})
    target_compile_definitions(opengl_transport_tests PRIVATE
        RENDERER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    list(APPEND RENDERER_TEST_TARGETS opengl_transport_tests)
    add_executable(optix_interop_tests tests/optix_interop_tests.cpp tests/test_main.cpp)
    add_executable(ddgi_tests tests/ddgi_tests.cpp tests/test_main.cpp)
    target_link_libraries(ddgi_tests PRIVATE renderer_core renderer_opengl renderer_ddgi renderer_glad ${RENDERER_SDL_TARGET})
    target_compile_definitions(ddgi_tests PRIVATE RENDERER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    list(APPEND RENDERER_TEST_TARGETS ddgi_tests)
    target_link_libraries(optix_interop_tests PRIVATE renderer_core renderer_opengl renderer_glad ${RENDERER_SDL_TARGET})
    target_compile_definitions(optix_interop_tests PRIVATE RENDERER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    list(APPEND RENDERER_TEST_TARGETS optix_interop_tests)
endif()

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
        # Diagnostics variant: only the path tracer translation unit is
        # recompiled with RENDERER_BENCHMARK_DIAGNOSTICS; the rest of the
        # core is shared with the normal build.
        add_library(renderer_cuda_pathtracer_diagnostics STATIC
            src/render/pathtracer/cuda_pathtracer.cu)
        target_include_directories(renderer_cuda_pathtracer_diagnostics PUBLIC
            ${RENDERER_PUBLIC_INCLUDE_DIRECTORIES})
        target_link_libraries(renderer_cuda_pathtracer_diagnostics PUBLIC
            renderer_scene
            renderer_render_common
            renderer_cuda_api)
        target_link_libraries(renderer_cuda_pathtracer_diagnostics PRIVATE
            CUDA::cudart)
        target_compile_definitions(renderer_cuda_pathtracer_diagnostics PUBLIC
            RENDERER_HAS_CUDA=1
            RENDERER_BENCHMARK_DIAGNOSTICS=1)
        set_target_properties(renderer_cuda_pathtracer_diagnostics PROPERTIES
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
            renderer_scene
            renderer_render_common
            renderer_interactive
            renderer_cuda_api
            renderer_path_session
            renderer_cuda_pathtracer_diagnostics
            renderer_opengl
            renderer_glad
            ${RENDERER_SDL_TARGET}
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
            renderer_scene
            renderer_render_common
            renderer_cuda_api
            renderer_path_session
            renderer_cuda_pathtracer_diagnostics
            CUDA::cudart
        )
        target_compile_definitions(benchmark_diagnostics_tests PRIVATE
            RENDERER_HAS_CUDA=1
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
    renderer_cuda_api
    renderer_cuda_pathtracer
    renderer_realtime
    renderer_path_session
    renderer_interactive
    renderer
    ${RENDERER_TEST_TARGETS})
if(RENDERER_BUILD_VIEWER)
    list(APPEND RENDERER_NATIVE_TARGETS renderer_opengl viewer opengl_npr_smoke)
endif()
if(RENDERER_BUILD_BENCHMARKS)
    list(APPEND RENDERER_NATIVE_TARGETS
        renderer_benchmark_common viewer_benchmark benchmark_tests)
    if(RENDERER_HAS_CUDA)
        list(APPEND RENDERER_NATIVE_TARGETS
            renderer_cuda_pathtracer_diagnostics viewer_benchmark_diagnostics
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
if(RENDERER_BUILD_VIEWER)
    add_test(NAME viewer_ui_tests COMMAND viewer_ui_tests)
    add_test(NAME opengl_transport_tests COMMAND opengl_transport_tests)
    add_test(NAME ddgi_tests COMMAND ddgi_tests)
    add_test(NAME optix_interop_tests COMMAND optix_interop_tests)
endif()
add_test(NAME cuda_contract_tests COMMAND cuda_contract_tests)
add_test(NAME realtime_tests COMMAND realtime_tests)

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
target_enable_warnings(renderer_cuda_api)
target_enable_warnings(renderer_cuda_pathtracer)
target_enable_warnings(renderer_realtime)
target_enable_warnings(renderer_path_session)
target_enable_warnings(renderer_interactive)
foreach(test_target IN LISTS RENDERER_TEST_TARGETS)
    target_enable_warnings(${test_target})
endforeach()
if(RENDERER_BUILD_BENCHMARKS)
    target_enable_warnings(renderer_benchmark_common)
    target_enable_warnings(viewer_benchmark)
    target_enable_warnings(benchmark_tests)
    if(RENDERER_HAS_CUDA)
        target_enable_warnings(renderer_cuda_pathtracer_diagnostics)
        target_enable_warnings(viewer_benchmark_diagnostics)
        target_enable_warnings(benchmark_diagnostics_tests)
    endif()
endif()
if(RENDERER_BUILD_VIEWER)
    target_enable_warnings(renderer_opengl)
    target_enable_warnings(viewer)
    target_enable_warnings(opengl_npr_smoke)
endif()

# ---------------------------------------------------------------------------
# Install, export, and packaging.
#
# Executables, headers, shaders, and docs install into a standard prefix.
# Static libraries are exported for build-tree consumers only: they link
# FetchContent-provided third-party targets (Eigen3, nlohmann_json, ...) that
# are not part of an installable package, so an installed config-file package
# would not be relocatable. Downstream CMake projects may add_subdirectory()
# this repository and use the exported RendererTargets from the build tree.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(FILES src/render/realtime/PBRT_LICENSE.txt
    DESTINATION "${CMAKE_INSTALL_DATADIR}/renderer/licenses")
if(TARGET renderer_rtrt_optix)
    install(FILES "${optix_headers_SOURCE_DIR}/LICENSE.txt" "${optix_headers_SOURCE_DIR}/license_info.txt"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/OPTIX_BSD_NOTICE.txt"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/renderer/licenses/optix")
endif()

set(RENDERER_INSTALL_EXECUTABLES renderer)
if(RENDERER_BUILD_VIEWER)
    list(APPEND RENDERER_INSTALL_EXECUTABLES viewer)
    install(DIRECTORY shaders/opengl
        DESTINATION "${CMAKE_INSTALL_DATADIR}/renderer/shaders"
        PATTERN "LTC_LICENSE.txt" EXCLUDE)
    install(FILES shaders/opengl/LTC_LICENSE.txt
        DESTINATION "${CMAKE_INSTALL_DATADIR}/renderer/shaders/opengl")
endif()

install(TARGETS ${RENDERER_INSTALL_EXECUTABLES}
    EXPORT RendererTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
export(EXPORT RendererTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/RendererTargets.cmake"
    NAMESPACE renderer::)

install(DIRECTORY src/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/renderer"
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cuh")
install(DIRECTORY external/stb external/tinyobjloader
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/renderer"
    FILES_MATCHING PATTERN "*.h")
install(FILES README.md
    DESTINATION "${CMAKE_INSTALL_DOCDIR}")

set(CPACK_PACKAGE_NAME "renderer")
set(CPACK_PACKAGE_VENDOR "Zaurry")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "C++20 teaching 3D renderer: OpenGL real-time preview and CUDA path tracing")
set(CPACK_GENERATOR "ZIP")
include(CPack)
