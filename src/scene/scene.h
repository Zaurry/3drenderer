#pragma once

#include "core/color.h"
#include "scene/light.h"
#include "scene/material.h"
#include "scene/primitive.h"
#include "scene/texture.h"

#include <vector>

namespace renderer {

struct Scene {
    std::vector<Material> materials;
    std::vector<ImageTexture> textures;
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<PointLight> point_lights;
    std::vector<DirectionalLight> directional_lights;
    Color environment = Color(0.02, 0.03, 0.05);
};

Scene make_gradient_sphere_scene();
Scene make_raster_triangle_scene();
Scene make_mirror_spheres_scene();
Scene make_cornell_box_scene();

}  // namespace renderer
