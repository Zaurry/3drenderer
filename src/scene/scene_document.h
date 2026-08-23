#pragma once

#include "core/math/bounds.h"
#include "core/math/ray.h"
#include "core/math/types.h"
#include "scene/light.h"
#include "scene/instanced_scene.h"
#include "scene/scene.h"
#include "scene/scene_revision.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace renderer {

struct LoadedScene;
class SceneEditTransaction;
class SceneIntersector;

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
    RectAreaLight,
    Camera,
};

enum class SceneCameraProjection { Perspective, Orthographic };

struct SceneTrs {
    Vec3 translation = Vec3::Zero();
    Vec3 rotation_degrees = Vec3::Zero();
    Vec3 scale = Vec3::Ones();

    Mat4 matrix() const;
    bool valid() const;
};

struct SceneTransform {
    Mat4 local_matrix = Mat4::Identity();

    Mat4 matrix() const;
    bool valid() const;
    static SceneTransform from_trs(const SceneTrs& trs);
    std::optional<SceneTrs> trs() const;
};

struct SceneMeshAsset {
    AssetId id = kInvalidAssetId;
    std::uint64_t geometry_revision = 1;
    std::filesystem::path source_path;
    std::string builtin_id;
    int source_mesh_index = -1;
    Scene local_scene;
    // Analytic lights extracted from mesh geometry are instantiated as child
    // document objects for every import of this asset.
    std::vector<RectAreaLight> rect_area_lights;
    std::shared_ptr<const Scene> render_geometry;
    std::vector<Sphere> procedural_spheres;
    std::shared_ptr<const SceneIntersector> picking_intersector;
    Bounds3 local_bounds;
    std::vector<std::string> warnings;
    std::vector<std::string> material_names;
};

// Per-object material override, stored per material slot.
//
// REDESIGN TODO (docs/design-review.md §1.1): this is a hand-maintained
// partial copy of Material. Field mapping is duplicated across validation,
// copy-from-source, application, and (de)serialization; specular-glossiness
// parameters and texture transforms are not overridable, and the use_*
// booleans can only disable textures (true = keep the source material's
// texture). Planned: replace with a full Material plus a slot and migrate
// the .rscene format.
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
    float light_source_radius = 0.05f;
    float directional_angular_radius_radians = 0.00464257581f;
    bool light_casts_shadows = true;
    int light_shadow_priority = 0;
    float area_width = 1.0f;
    float area_height = 1.0f;
    bool light_two_sided = false;
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

struct SceneCameraProperties {
    SceneCameraProjection projection = SceneCameraProjection::Perspective;
    float vertical_fov_degrees = 45.0f;
    float aspect_ratio = 0.0f;
    float x_magnification = 1.0f;
    float y_magnification = 1.0f;
    float near_plane = 0.01f;
    float far_plane = 1000.0f;
};

struct SceneLightProperties {
    Color color = Color(25.0f, 25.0f, 25.0f);
    float range = 0.0f;
    float spot_inner_cone_radians = 0.0f;
    float spot_outer_cone_radians = 0.7853981634f;
    float source_radius = 0.05f;
    float directional_angular_radius_radians = 0.00464257581f;
    bool casts_shadows = true;
    int shadow_priority = 0;
    float area_width = 1.0f;
    float area_height = 1.0f;
    bool two_sided = false;
};

// Single-threaded editing document.
//
// Threading contract: SceneDocument is not thread-safe. All mutation and all
// reads (including render_scene_snapshot()) must happen on one thread. Several
// accessors lazily rebuild mutable caches through const methods, and the
// snapshot accessor returns a reference into the document's own storage;
// introduce locking here only together with a snapshot ownership redesign.
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
    std::vector<std::shared_ptr<const SceneMeshAsset>> assets() const;
    const SceneMeshAsset* asset_for_object(ObjectId id) const;
    const SceneObject* find(ObjectId id) const;
    std::vector<ObjectId> children(ObjectId parent_id) const;

    bool set_object_name(ObjectId id, std::string name);
    bool set_object_visible(ObjectId id, bool visible);
    bool set_object_locked(ObjectId id, bool locked);
    bool set_camera_properties(
        ObjectId id,
        const SceneCameraProperties& properties);
    bool set_light_properties(
        ObjectId id,
        const SceneLightProperties& properties);

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
    ObjectId create_rect_area_light(
        std::string name,
        const Vec3& position,
        const Vec3& direction,
        const Color& radiance,
        float width = 1.0f,
        float height = 1.0f,
        bool two_sided = false,
        ObjectId parent_id = kInvalidObjectId);
    ObjectId create_camera(
        std::string name,
        SceneCameraProjection projection,
        ObjectId parent_id = kInvalidObjectId);
    ObjectId duplicate_subtree(ObjectId id);
    bool erase_subtree(ObjectId id);
    bool reparent(ObjectId id, ObjectId new_parent_id);
    std::size_t prune_unreferenced_assets();
    bool set_world_matrix(ObjectId id, const Mat4& world);
    std::optional<SceneTrs> local_trs(ObjectId id) const;
    bool set_local_trs(ObjectId id, const SceneTrs& trs);
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

    const RenderSceneSnapshot& render_scene_snapshot() const;
    const SceneRevisions& revisions() const;
    const Color& environment() const;
    void set_environment(Color color);
    void set_environment_map(const std::filesystem::path& path);
    void clear_environment_map();
    const std::shared_ptr<const EnvironmentMap>& environment_map() const;
    const std::filesystem::path& environment_path() const;
    float environment_intensity() const;
    void set_environment_intensity(float intensity);
    float environment_rotation_degrees() const;
    void set_environment_rotation_degrees(float rotation_degrees);
    bool environment_background_visible() const;
    void set_environment_background_visible(bool visible);

    void checkpoint();
    SceneEditTransaction begin_edit(std::string merge_key = {});
    bool undo();
    bool redo();
    bool can_undo() const;
    bool can_redo() const;
    bool dirty() const;
    void mark_saved();

    const std::filesystem::path& file_path() const;
    const std::vector<std::string>& warnings() const;

private:
    friend class SceneEditTransaction;

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
    mutable RenderSceneSnapshot render_scene_snapshot_;
    mutable bool snapshot_dirty_ = true;
    mutable bool object_index_dirty_ = true;
    mutable bool spatial_cache_dirty_ = true;
    mutable bool asset_index_dirty_ = true;
    mutable std::unordered_map<ObjectId, std::size_t> object_indices_;
    mutable std::unordered_map<ObjectId, std::vector<ObjectId>> children_by_parent_;
    mutable std::unordered_map<ObjectId, Mat4> world_matrices_;
    mutable std::unordered_map<ObjectId, Bounds3> world_bounds_;
    mutable std::unordered_map<AssetId, std::size_t> asset_indices_;
    std::uint64_t render_source_id_ = 0;
    SceneRevisions revisions_;
    bool uncheckpointed_changes_ = false;
    ObjectId next_object_id_ = 1;
    AssetId next_asset_id_ = 1;
    std::vector<State> history_;
    // Parallel to history_: assets detached when an undo/redo move ended at
    // the corresponding slot (referenced by no object in that state). They
    // are re-inserted when the slot is reached again, so redo can resurrect
    // geometry that an undo pruned.
    std::vector<std::vector<std::shared_ptr<SceneMeshAsset>>> history_pruned_assets_;
    std::size_t history_cursor_ = 0;
    std::optional<std::size_t> saved_cursor_ = 0;
    std::filesystem::path file_path_;
    std::vector<std::string> warnings_;
    bool edit_transaction_active_ = false;
    std::string last_checkpoint_merge_key_;

    std::shared_ptr<SceneMeshAsset> find_asset(AssetId id) const;
    SceneObject* mutable_object_for_edit(ObjectId id);
    std::vector<std::shared_ptr<SceneMeshAsset>> detach_unreferenced_assets();
    void restore_pruned_assets_at(std::size_t history_index);
    void rebuild_object_index() const;
    void rebuild_spatial_cache() const;
    void mark_changed(
        SceneRevisionDomain domains = SceneRevisionDomain::All,
        bool hierarchy_changed = false);
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
    ObjectId create_rect_area_light_from_source(
        std::string name,
        const RectAreaLight& source,
        ObjectId parent_id);
    ObjectId clone_subtree(ObjectId source_id, ObjectId parent_id);
    bool is_descendant(ObjectId candidate, ObjectId ancestor) const;
    bool is_effectively_visible(ObjectId id) const;
    nlohmann::json serialize_document(
        const std::filesystem::path& base,
        bool session_snapshot) const;
    static SceneDocument deserialize_document(
        const nlohmann::json& root,
        const std::filesystem::path& document_path,
        int width,
        int height,
        bool session_snapshot);
    void ensure_render_scene_snapshot() const;
};

class SceneEditTransaction {
public:
    SceneEditTransaction(SceneEditTransaction&& other) noexcept;
    SceneEditTransaction& operator=(SceneEditTransaction&& other) noexcept;
    SceneEditTransaction(const SceneEditTransaction&) = delete;
    SceneEditTransaction& operator=(const SceneEditTransaction&) = delete;
    ~SceneEditTransaction();

    void commit();
    void cancel();
    bool active() const;

private:
    friend class SceneDocument;
    struct Backup;

    SceneEditTransaction(
        SceneDocument& document,
        std::string merge_key);

    SceneDocument* document_ = nullptr;
    std::unique_ptr<Backup> backup_;
    std::string merge_key_;
};

}  // namespace renderer
