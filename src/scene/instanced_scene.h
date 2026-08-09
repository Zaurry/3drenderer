#pragma once

#include "core/math/bounds.h"
#include "core/math/types.h"
#include "scene/light.h"
#include "scene/material.h"
#include "scene/scene.h"
#include "scene/scene_revision.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace renderer {

class MaterialSlot {
public:
    static MaterialSlot missing() {
        return MaterialSlot();
    }

    static MaterialSlot bound(std::uint32_t index) {
        MaterialSlot slot;
        slot.index_ = index;
        return slot;
    }

    bool has_value() const {
        return index_.has_value();
    }

    std::uint32_t value() const {
        if (!index_) {
            throw std::logic_error("missing material slot has no index");
        }
        return *index_;
    }

    int device_value() const {
        return index_
            ? static_cast<int>(*index_)
            : -1;
    }

    bool operator==(const MaterialSlot&) const = default;

private:
    std::optional<std::uint32_t> index_;
};

struct RenderSceneAssetSnapshot {
    std::uint64_t asset_id = 0;
    std::uint64_t geometry_revision = 0;
    std::shared_ptr<const Scene> local_scene;
    Bounds3 local_bounds;
    std::vector<MaterialSlot> sphere_material_slots;
    std::vector<MaterialSlot> triangle_material_slots;
};

struct RenderSceneInstanceSnapshot {
    std::uint64_t object_id = 0;
    int asset_index = -1;
    Mat4 object_to_world = Mat4::Identity();
    Mat4 world_to_object = Mat4::Identity();
    Mat3 normal_to_world = Mat3::Identity();
    Bounds3 world_bounds;
    std::vector<Material> materials;
};

struct RenderSceneSnapshot {
    // Identifies the snapshot lineage independently from per-domain revisions.
    // Revisions are only comparable when this value is the same.
    std::uint64_t source_id = 0;
    SceneRevisions revisions;
    std::vector<RenderSceneAssetSnapshot> assets;
    std::vector<RenderSceneInstanceSnapshot> instances;
    std::vector<ImageTexture> textures;
    std::vector<PointLight> point_lights;
    std::vector<DirectionalLight> directional_lights;
    std::vector<SpotLight> spot_lights;
    Color environment = Color(0.02f, 0.03f, 0.05f);
    std::shared_ptr<const EnvironmentMap> environment_map;
    float environment_intensity = 1.0f;
    float environment_rotation_degrees = 0.0f;
    bool environment_background_visible = true;
};

std::uint64_t allocate_render_scene_source_id();

Scene flatten_render_scene_snapshot(
    const RenderSceneSnapshot& snapshot);

RenderSceneSnapshot make_render_scene_snapshot(Scene scene);

const std::shared_ptr<const Scene>& canonical_unit_sphere_geometry();

}  // namespace renderer
