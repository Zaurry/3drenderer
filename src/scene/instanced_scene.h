#pragma once

#include "core/math/bounds.h"
#include "core/math/types.h"
#include "scene/light.h"
#include "scene/material.h"
#include "scene/scene.h"

#include <cstdint>
#include <vector>

namespace renderer {

struct InstancedSceneAssetView {
    std::uint64_t asset_id = 0;
    const Scene* local_scene = nullptr;
    Bounds3 local_bounds;
};

struct InstancedSceneInstanceView {
    std::uint64_t object_id = 0;
    int asset_index = -1;
    Mat4 object_to_world = Mat4::Identity();
    Mat4 world_to_object = Mat4::Identity();
    Mat3 normal_to_world = Mat3::Identity();
    Bounds3 world_bounds;
    std::vector<Material> materials;
};

struct InstancedSceneView {
    std::vector<InstancedSceneAssetView> assets;
    std::vector<InstancedSceneInstanceView> instances;
    std::vector<PointLight> point_lights;
    std::vector<DirectionalLight> directional_lights;
    Color environment = Color(0.02f, 0.03f, 0.05f);
};

}  // namespace renderer
