#pragma once

#include "core/color.h"

namespace renderer {

enum class MaterialType { Diffuse, Metal, Dielectric, Emissive };

struct Material {
    MaterialType type = MaterialType::Diffuse;
    Color base_color = Color(0.8f, 0.8f, 0.8f);
    Color emission = Color(0, 0, 0);
    double roughness = 0.0;
    double metallic = 0.0;
    double ior = 1.5;
    double opacity = 1.0;
    double alpha_cutoff = 0.5;
    double bump_scale = 1.0;
    bool two_sided = true;
    int diffuse_texture_id = -1;
    int opacity_texture_id = -1;
    int bump_texture_id = -1;
};

}  // namespace renderer
