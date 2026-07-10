#pragma once

#include "core/color.h"

namespace renderer {

enum class MaterialType { Diffuse, Metal, Dielectric, Emissive };

struct Material {
    MaterialType type = MaterialType::Diffuse;
    Color base_color = Color(0.8, 0.8, 0.8);
    Color emission = Color(0, 0, 0);
    double roughness = 0.0;
    double metallic = 0.0;
    double ior = 1.5;
    int diffuse_texture_id = -1;
};

}  // namespace renderer
