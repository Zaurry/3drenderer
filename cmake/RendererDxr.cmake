set(RENDERER_DXR "AUTO" CACHE STRING "DirectX ray tracing: AUTO, ON, OFF")
set_property(CACHE RENDERER_DXR PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${RENDERER_DXR}" RENDERER_DXR_MODE)
if(NOT RENDERER_DXR_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "RENDERER_DXR must be AUTO, ON, or OFF")
endif()
set(RENDERER_HAS_DXR OFF)
if(WIN32 AND NOT RENDERER_DXR_MODE STREQUAL "OFF")
    set(RENDERER_HAS_DXR ON)
elseif(RENDERER_DXR_MODE STREQUAL "ON")
    message(FATAL_ERROR "RENDERER_DXR=ON requires Windows")
endif()

if(RENDERER_HAS_DXR)
    set(RENDERER_DXR_SDK_ROOT "${CMAKE_SOURCE_DIR}/build/dxr-sdk" CACHE PATH "Optional pre-fetched DXR SDK cache")
    function(renderer_dxr_dependency name relative url hash)
        if(EXISTS "${RENDERER_DXR_SDK_ROOT}/${relative}")
            string(REGEX REPLACE "/.*" "" archive_name "${relative}")
            set(archive "${RENDERER_DXR_SDK_ROOT}/${archive_name}.zip")
            if(NOT EXISTS "${archive}")
                message(FATAL_ERROR "Cannot verify ${name}: missing ${archive}. Run tools/fetch_dxr_sdks.ps1 or use an empty RENDERER_DXR_SDK_ROOT.")
            endif()
            file(SHA256 "${archive}" actual_hash)
            if(NOT actual_hash STREQUAL hash)
                message(FATAL_ERROR "Pinned DXR dependency checksum mismatch: ${archive}")
            endif()
            set(${name}_SOURCE_DIR "${RENDERER_DXR_SDK_ROOT}/${relative}" PARENT_SCOPE)
        else()
            FetchContent_Declare(${name} URL "${url}" URL_HASH "SHA256=${hash}"
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR _download_only)
            FetchContent_MakeAvailable(${name})
            set(${name}_SOURCE_DIR "${${name}_SOURCE_DIR}" PARENT_SCOPE)
        endif()
    endfunction()
    renderer_dxr_dependency(dxr_agility "agility"
        "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/1.619.6/microsoft.direct3d.d3d12.1.619.6.nupkg"
        08f0489281401aa430fc37322d6c3fc98a8025175aacd714c10d562f4963f1e9)
    renderer_dxr_dependency(dxr_dxc "dxc"
        "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2609/dxc_2026_09_29.zip"
        ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1)
    renderer_dxr_dependency(dxr_rtxdi "rtxdi-library/RTXDI-Library-f12037fa8e97ebc08e9e3edfd2de528ed1772a4b"
        "https://github.com/NVIDIA-RTX/RTXDI-Library/archive/f12037fa8e97ebc08e9e3edfd2de528ed1772a4b.zip"
        f91de92d7c27f9824915ee5fcf62b0da664a6eb294edb659056ff3c462823f75)
    renderer_dxr_dependency(dxr_nrd "nrd/NRD-4.17.3"
        "https://github.com/NVIDIA-RTX/NRD/archive/refs/tags/v4.17.3.zip"
        c3a71eb0c3577f664f6f8bf3be7c585a39911fbf2895c144af10c58b861ad6ba)
    renderer_dxr_dependency(dxr_streamline "streamline"
        "https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.14.1/streamline-sdk-v2.14.1.zip"
        92c4d954631a1710da86ca3fa8d5034f2b9503838c95fc4ae977ae149319781b)
    set(RENDERER_DXC "${dxr_dxc_SOURCE_DIR}/bin/x64/dxc.exe")
    renderer_dxr_dependency(dxr_shadermake "shadermake/ShaderMake-18f5a344e7ca8fa65daaf079d07bc8ce38453e05"
        "https://github.com/NVIDIA-RTX/ShaderMake/archive/18f5a344e7ca8fa65daaf079d07bc8ce38453e05.zip"
        35de2547e28cf10f18a0e2782cf154a3d1610212d00d829fa8e73a18210623b6)
    renderer_dxr_dependency(dxr_mathlib "mathlib/MathLib-11"
        "https://github.com/NVIDIA-RTX/MathLib/archive/refs/tags/v11.zip"
        4a0772103fd7ef832f8c2fb3ec7fb6bf5b3bebbd41ecafc476671d11eca28a06)
    set(SHADERMAKE_FIND_COMPILERS OFF CACHE BOOL "Use the pinned DXC" FORCE)
    set(SHADERMAKE_TOOL OFF CACHE BOOL "Build ShaderMake in this project" FORCE)
    set(DXC_PATH "${RENDERER_DXC}")
    add_subdirectory("${dxr_shadermake_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/dxr/shadermake" EXCLUDE_FROM_ALL)
    add_subdirectory("${dxr_mathlib_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/dxr/mathlib" EXCLUDE_FROM_ALL)
    set(SHADERMAKE_DXC_PATH "${RENDERER_DXC}")
    set(NRD_STATIC_LIBRARY ON CACHE BOOL "Static NRD host library" FORCE)
    set(NRD_NRI OFF CACHE BOOL "Native D3D12 integration" FORCE)
    set(NRD_EMBEDS_DXIL_SHADERS ON CACHE BOOL "Compile native DXIL" FORCE)
    set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "No legacy bytecode" FORCE)
    set(NRD_EMBEDS_SPIRV_SHADERS OFF CACHE BOOL "No Vulkan backend" FORCE)
    set(NRD_NORMAL_ENCODING 4 CACHE STRING "Signed normals in FP16" FORCE)
    # SDK sources are shared by the CUDA and no-CUDA presets. ShaderMake writes
    # temporary files as well as headers; each build needs its own output tree.
    set(NRD_SHADERS_PATH "${CMAKE_BINARY_DIR}/generated/nrd" CACHE STRING "Per-build NRD shader output" FORCE)
    add_subdirectory("${dxr_nrd_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/dxr/nrd" EXCLUDE_FROM_ALL)
    add_library(renderer_dxr_runtime STATIC src/platform/d3d12/d3d12_context.cpp src/render/dxr/dxr_streamline.cpp)
    set(runtime_manifest "#pragma once\nstruct DxrPinnedRuntime {const char* name;const char* sha256;};\ninline constexpr DxrPinnedRuntime dxr_pinned_runtimes[]={\n")
    foreach(runtime sl.interposer sl.common sl.dlss sl.dlss_d nvngx_dlss nvngx_dlssd)
        file(SHA256 "${dxr_streamline_SOURCE_DIR}/bin/x64/${runtime}.dll" runtime_hash)
        string(APPEND runtime_manifest "{\"${runtime}.dll\",\"${runtime_hash}\"},\n")
    endforeach()
    string(APPEND runtime_manifest "};\n")
    file(WRITE "${CMAKE_BINARY_DIR}/generated/dxr_runtime_manifest.h" "${runtime_manifest}")
    target_include_directories(renderer_dxr_runtime BEFORE PUBLIC
        "${dxr_agility_SOURCE_DIR}/build/native/include" "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(renderer_dxr_runtime PUBLIC RENDERER_HAS_DXR=1 NOMINMAX WIN32_LEAN_AND_MEAN)
    target_include_directories(renderer_dxr_runtime PRIVATE "${dxr_streamline_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/generated")
    target_link_libraries(renderer_dxr_runtime PUBLIC d3d12 dxgi dxguid PRIVATE Eigen3::Eigen bcrypt version)
    add_executable(dxr_probe tools/dxr_probe.cpp)
    target_link_libraries(dxr_probe PRIVATE renderer_dxr_runtime)
    function(renderer_deploy_dxr target)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/D3D12"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dxr_agility_SOURCE_DIR}/build/native/bin/x64/D3D12Core.dll"
                "${dxr_agility_SOURCE_DIR}/build/native/bin/x64/d3d12SDKLayers.dll"
                "$<TARGET_FILE_DIR:${target}>/D3D12"
            VERBATIM)
        foreach(runtime sl.interposer sl.common sl.dlss sl.dlss_d nvngx_dlss nvngx_dlssd)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/Streamline"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dxr_streamline_SOURCE_DIR}/bin/x64/${runtime}.dll" "$<TARGET_FILE_DIR:${target}>/Streamline/"
                VERBATIM)
        endforeach()
    endfunction()
    renderer_deploy_dxr(dxr_probe)
endif()
