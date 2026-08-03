#pragma once

#include "core/color.h"
#include "scene/light.h"
#include "scene/environment.h"
#include "scene/material.h"
#include "scene/primitive.h"
#include "scene/texture.h"

#include <vector>
#include <memory>

namespace renderer {

struct Scene {
    std::vector<Material> materials;
    std::vector<ImageTexture> textures;
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<PointLight> point_lights;
    std::vector<DirectionalLight> directional_lights;
    std::vector<SpotLight> spot_lights;
    Color environment = Color(0.02f, 0.03f, 0.05f);
    std::shared_ptr<const EnvironmentMap> environment_map;
    float environment_intensity = 1.0f;
    float environment_rotation_degrees = 0.0f;
    bool environment_background_visible = true;
};

Scene make_gradient_sphere_scene();
Scene make_triangle_scene();
Scene make_mirror_spheres_scene();
Scene make_cornell_box_scene();

}  // namespace renderer
