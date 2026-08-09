#pragma once

#include "core/color.h"
#include "core/math/types.h"

namespace renderer {

enum class MaterialType { Diffuse, Metal, Dielectric, Emissive, Pbr };

enum class AlphaMode { Opaque, Mask, Blend };

enum class PbrWorkflow { MetallicRoughness, SpecularGlossiness };

struct TextureTransform {
    Vec2 offset = Vec2::Zero();
    Vec2 scale = Vec2::Ones();
    float rotation = 0.0f;
    int texcoord = 0;
};

struct Material {
    MaterialType type = MaterialType::Diffuse;
    Color base_color = Color(0.8f, 0.8f, 0.8f);
    Color emission = Color(0, 0, 0);
    float roughness = 0.0f;
    float metallic = 0.0f;
    float ior = 1.5f;
    float opacity = 1.0f;
    float alpha_cutoff = 0.5f;
    float bump_scale = 1.0f;
    float normal_scale = 1.0f;
    float occlusion_strength = 1.0f;
    AlphaMode alpha_mode = AlphaMode::Opaque;
    PbrWorkflow pbr_workflow = PbrWorkflow::MetallicRoughness;
    bool two_sided = true;
    Color specular_color = Color::Ones();
    float specular_factor = 1.0f;
    float glossiness = 1.0f;
    int diffuse_texture_id = -1;
    int opacity_texture_id = -1;
    int bump_texture_id = -1;
    int base_color_texture_id = -1;
    int metallic_roughness_texture_id = -1;
    int normal_texture_id = -1;
    int occlusion_texture_id = -1;
    int emissive_texture_id = -1;
    int specular_texture_id = -1;
    int specular_color_texture_id = -1;
    int specular_glossiness_texture_id = -1;
    TextureTransform base_color_texture_transform;
    TextureTransform metallic_roughness_texture_transform;
    TextureTransform normal_texture_transform;
    TextureTransform occlusion_texture_transform;
    TextureTransform emissive_texture_transform;
    TextureTransform specular_texture_transform;
    TextureTransform specular_color_texture_transform;
    TextureTransform specular_glossiness_texture_transform;
};

inline Material diagnostic_material() {
    Material material;
    material.type = MaterialType::Diffuse;
    material.base_color = Color(1.0f, 0.0f, 1.0f);
    material.roughness = 1.0f;
    material.two_sided = true;
    return material;
}

}  // namespace renderer
