#pragma once

#include <array>
#include <string_view>

namespace renderer {

// Shared C++/GLSL constants. The GLSL sources must match these values; the
// source-text lint (tests/opengl_shader_lint_tests.cpp) verifies that.
inline constexpr int kOpenGlMaxShadowLightSlots = 32;
inline constexpr int kOpenGlMaxPcssSamples = 64;
inline constexpr int kOpenGlShadowResolutionMin = 128;
inline constexpr int kOpenGlShadowResolutionMax = 4096;
inline constexpr int kOpenGlMaxSsrRays = 8;
inline constexpr int kOpenGlMaxSsrSteps = 256;
inline constexpr int kOpenGlMaxSsrDenoisePasses = 4;

struct OpenGlShaderBinding {
    int location = -1;
    std::string_view name;
};

inline constexpr std::array<OpenGlShaderBinding, 5>
    kOpenGlFragmentOutputContract{{
        {0, "out_linear_color"},
        {1, "out_transparency_accum"},
        {2, "out_transparency_reveal"},
        {3, "out_direct_lighting"},
        {4, "out_ray_radiance"},
    }};

inline constexpr std::array<OpenGlShaderBinding, 11>
    kOpenGlSsrTraceTextureContract{{
        {0, "u_ray_radiance"},
        {1, "u_linear_depth"},
        {2, "u_view_normal"},
        {3, "u_ssr_material"},
        {4, "u_hiz"},
        {5, "u_ambient_occlusion"},
        {6, "u_environment_radiance"},
        {7, "u_ssr_pbr"},
        {8, "u_ssr_pbr_aux"},
        {9, "u_diffuse_fresnel"},
        {10, "u_environment_brdf_lut"},
    }};

inline constexpr std::array<OpenGlShaderBinding, 6>
    kOpenGlVertexAttributeContract{{
        {0, "a_position"},
        {1, "a_normal"},
        {2, "a_uv"},
        {3, "a_tangent"},
        {4, "a_uv1"},
        {5, "a_color"},
    }};

inline constexpr std::array<OpenGlShaderBinding, 16>
    kOpenGlTextureBindingContract{{
        {0, "u_base_color_texture"},
        {1, "u_opacity_texture"},
        {2, "u_normal_or_bump_texture"},
        {3, "u_metallic_roughness_texture"},
        {4, "u_occlusion_texture"},
        {5, "u_emissive_texture"},
        {6, "u_specular_texture"},
        {7, "u_specular_color_texture"},
        {8, "u_specular_glossiness_texture"},
        {9, "u_environment_prefilter"},
        {10, "u_environment_brdf_lut"},
        {11, "u_ltc_matrix_lut"},
        {12, "u_ltc_amplitude_lut"},
        {13, "u_shadow_maps_2d"},
        {14, "u_shadow_maps_cube"},
        {15, "u_ambient_occlusion_texture"},
    }};

inline constexpr std::array<std::string_view, 18>
    kOpenGlMaterialUniformContract{{
        "u_material_type",
        "u_pbr_workflow",
        "u_base_color",
        "u_emission",
        "u_ior",
        "u_specular_color",
        "u_specular_factor",
        "u_glossiness",
        "u_metallic",
        "u_roughness",
        "u_opacity",
        "u_alpha_cutoff",
        "u_bump_scale",
        "u_normal_scale",
        "u_occlusion_strength",
        "u_alpha_mode",
        "u_two_sided",
        "u_texture_texcoord",
    }};

inline constexpr std::array<OpenGlShaderBinding, 4>
    kOpenGlLightBufferContract{{
        {0, "DirectionalLightBuffer"},
        {1, "PointLightBuffer"},
        {2, "SpotLightBuffer"},
        {3, "RectAreaLightBuffer"},
    }};

}  // namespace renderer
