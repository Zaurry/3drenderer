#pragma once

#include "core/math/bounds.h"
#include "core/math/ray.h"
#include "core/math/types.h"
#include "scene/light.h"
#include "scene/instanced_scene.h"
#include "scene/scene.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace renderer {

struct LoadedScene;

using ObjectId = std::uint64_t;
using AssetId = std::uint64_t;

constexpr ObjectId kInvalidObjectId = 0;
constexpr AssetId kInvalidAssetId = 0;

enum class SceneObjectType {
    Group,
    Mesh,
    PointLight,
    DirectionalLight,
    SpotLight,
    Camera,
};

enum class SceneCameraProjection { Perspective, Orthographic };

struct SceneTransform {
    Vec3 translation = Vec3::Zero();
    Vec3 rotation_degrees = Vec3::Zero();
    Vec3 scale = Vec3::Ones();

    Mat4 matrix() const;
    bool valid() const;
};

struct SceneMeshAsset {
    AssetId id = kInvalidAssetId;
    std::filesystem::path source_path;
    std::string builtin_id;
    int source_mesh_index = -1;
    Scene local_scene;
    Bounds3 local_bounds;
    std::vector<std::string> warnings;
    std::vector<std::string> material_names;
};

struct SceneMaterialOverride {
    std::uint32_t material_slot = 0;
    MaterialType type = MaterialType::Diffuse;
    Color base_color = Color(0.8f, 0.8f, 0.8f);
    Color emission = Color::Zero();
    float roughness = 0.0f;
    float metallic = 0.0f;
    float ior = 1.5f;
    float opacity = 1.0f;
    float alpha_cutoff = 0.5f;
    float bump_scale = 1.0f;
    float normal_scale = 1.0f;
    float occlusion_strength = 1.0f;
    AlphaMode alpha_mode = AlphaMode::Opaque;
    bool two_sided = true;
    bool use_diffuse_texture = true;
    bool use_opacity_texture = true;
    bool use_bump_texture = true;
    bool use_base_color_texture = true;
    bool use_metallic_roughness_texture = true;
    bool use_normal_texture = true;
    bool use_occlusion_texture = true;
    bool use_emissive_texture = true;
};

struct SceneObject {
    ObjectId id = kInvalidObjectId;
    ObjectId parent_id = kInvalidObjectId;
    std::string name;
    SceneObjectType type = SceneObjectType::Group;
    SceneTransform transform;
    bool visible = true;
    bool locked = false;
    AssetId asset_id = kInvalidAssetId;
    Color light_color = Color(25.0f, 25.0f, 25.0f);
    float light_range = 0.0f;
    float spot_inner_cone_radians = 0.0f;
    float spot_outer_cone_radians = 0.7853981634f;
    SceneCameraProjection camera_projection = SceneCameraProjection::Perspective;
    float camera_vertical_fov_degrees = 45.0f;
    float camera_aspect_ratio = 0.0f;
    float camera_x_magnification = 1.0f;
    float camera_y_magnification = 1.0f;
    float camera_near_plane = 0.01f;
    float camera_far_plane = 1000.0f;
    std::vector<SceneMaterialOverride> material_overrides;
};

struct ScenePickResult {
    ObjectId object_id = kInvalidObjectId;
    float distance = 0.0f;
};

class SceneDocument {
public:
    SceneDocument();

    static SceneDocument from_scene(
        Scene scene,
        std::string name,
        std::string builtin_id = {});
    static SceneDocument load(const std::filesystem::path& path, int width, int height);
    static SceneDocument from_session_snapshot(
        const nlohmann::json& snapshot,
        int width,
        int height);

    void save(const std::filesystem::path& path);
    nlohmann::json session_snapshot() const;
    void restore_file_state(const std::filesystem::path& path, bool dirty);
    std::vector<ObjectId> import_path(const std::filesystem::path& path, int width, int height);

    const std::vector<SceneObject>& objects() const;
    std::vector<SceneObject>& objects();
    const std::vector<std::shared_ptr<SceneMeshAsset>>& assets() const;
    const SceneMeshAsset* asset_for_object(ObjectId id) const;
    const SceneObject* find(ObjectId id) const;
    SceneObject* find(ObjectId id);
    std::vector<ObjectId> children(ObjectId parent_id) const;

    ObjectId create_group(std::string name, ObjectId parent_id = kInvalidObjectId);
    ObjectId create_point_light(
        std::string name,
        const Vec3& position,
        const Color& intensity,
        ObjectId parent_id = kInvalidObjectId);
    ObjectId create_directional_light(
        std::string name,
        const Vec3& direction,
        const Color& radiance,
        ObjectId parent_id = kInvalidObjectId);
    ObjectId create_spot_light(
        std::string name,
        const Vec3& position,
        const Vec3& direction,
        const Color& intensity,
        float range,
        float inner_cone_radians,
        float outer_cone_radians,
        ObjectId parent_id = kInvalidObjectId);
    ObjectId create_camera(
        std::string name,
        SceneCameraProjection projection,
        ObjectId parent_id = kInvalidObjectId);
    ObjectId duplicate_subtree(ObjectId id);
    bool erase_subtree(ObjectId id);
    bool reparent(ObjectId id, ObjectId new_parent_id);
    bool set_world_matrix(ObjectId id, const Mat4& world);
    std::optional<SceneMaterialOverride> material_properties(
        ObjectId id,
        std::size_t material_slot) const;
    const SceneMaterialOverride* material_override(
        ObjectId id,
        std::size_t material_slot) const;
    bool set_material_override(
        ObjectId id,
        const SceneMaterialOverride& material_override);
    bool clear_material_override(ObjectId id, std::size_t material_slot);

    Mat4 world_matrix(ObjectId id) const;
    Bounds3 world_bounds(ObjectId id) const;
    Bounds3 scene_bounds() const;
    std::optional<ScenePickResult> pick(const Ray& ray) const;

    const Scene& render_scene() const;
    const InstancedSceneView& instanced_render_scene() const;
    bool rebuild_render_scene();
    Color& environment();
    const Color& environment() const;
    void set_environment_map(const std::filesystem::path& path);
    void clear_environment_map();
    const std::shared_ptr<const EnvironmentMap>& environment_map() const;
    const std::filesystem::path& environment_path() const;
    float& environment_intensity();
    float environment_intensity() const;
    float& environment_rotation_degrees();
    float environment_rotation_degrees() const;
    bool& environment_background_visible();
    bool environment_background_visible() const;

    void checkpoint();
    bool undo();
    bool redo();
    bool can_undo() const;
    bool can_redo() const;
    bool dirty() const;
    void mark_saved();

    const std::filesystem::path& file_path() const;
    const std::vector<std::string>& warnings() const;

private:
    struct State {
        std::vector<SceneObject> objects;
        Color environment = Color(0.02f, 0.025f, 0.03f);
        std::shared_ptr<const EnvironmentMap> environment_map;
        std::filesystem::path environment_path;
        float environment_intensity = 1.0f;
        float environment_rotation_degrees = 0.0f;
        bool environment_background_visible = true;
    };

    State state_;
    std::vector<std::shared_ptr<SceneMeshAsset>> assets_;
    mutable Scene render_scene_;
    mutable InstancedSceneView instanced_scene_;
    mutable Bounds3 render_bounds_;
    mutable bool render_dirty_ = true;
    mutable bool instanced_dirty_ = true;
    ObjectId next_object_id_ = 1;
    AssetId next_asset_id_ = 1;
    std::vector<State> history_;
    std::size_t history_cursor_ = 0;
    std::optional<std::size_t> saved_cursor_ = 0;
    std::filesystem::path file_path_;
    std::vector<std::string> warnings_;

    std::shared_ptr<SceneMeshAsset> find_asset(AssetId id) const;
    std::shared_ptr<SceneMeshAsset> load_asset(
        const std::filesystem::path& path,
        int width,
        int height,
        int source_mesh_index = -1);
    std::shared_ptr<SceneMeshAsset> store_loaded_asset(
        LoadedScene loaded,
        const std::filesystem::path& normalized_path,
        int source_mesh_index);
    ObjectId import_obj(
        const std::filesystem::path& path,
        ObjectId parent_id,
        int width,
        int height);
    ObjectId import_gltf(
        const std::filesystem::path& path,
        ObjectId parent_id,
        int width,
        int height);
    ObjectId clone_subtree(ObjectId source_id, ObjectId parent_id);
    bool is_descendant(ObjectId candidate, ObjectId ancestor) const;
    bool is_effectively_visible(ObjectId id) const;
    Mat4 world_matrix_recursive(ObjectId id, int depth) const;
    static bool decompose_matrix(const Mat4& matrix, SceneTransform& transform);
    nlohmann::json serialize_document(
        const std::filesystem::path& base,
        bool session_snapshot) const;
    static SceneDocument deserialize_document(
        const nlohmann::json& root,
        const std::filesystem::path& document_path,
        int width,
        int height,
        bool session_snapshot);
    void ensure_render_scene() const;
    void ensure_instanced_scene() const;
};

}  // namespace renderer
