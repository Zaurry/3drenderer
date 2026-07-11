#pragma once

#include "core/color.h"

namespace renderer {

enum class MaterialType { Diffuse, Metal, Dielectric, Emissive };

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
    bool two_sided = true;
    int diffuse_texture_id = -1;
    int opacity_texture_id = -1;
    int bump_texture_id = -1;
};

}  // namespace renderer
