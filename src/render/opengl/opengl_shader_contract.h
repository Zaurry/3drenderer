#pragma once

#include <array>
#include <string_view>

namespace renderer {

struct OpenGlShaderBinding {
    int location = -1;
    std::string_view name;
};

inline constexpr std::array<OpenGlShaderBinding, 6>
    kOpenGlVertexAttributeContract{{
        {0, "a_position"},
        {1, "a_normal"},
        {2, "a_uv"},
        {3, "a_tangent"},
        {4, "a_uv1"},
        {5, "a_color"},
    }};

inline constexpr std::array<OpenGlShaderBinding, 11>
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

inline constexpr std::array<OpenGlShaderBinding, 3>
    kOpenGlLightBufferContract{{
        {0, "DirectionalLightBuffer"},
        {1, "PointLightBuffer"},
        {2, "SpotLightBuffer"},
    }};

}  // namespace renderer
