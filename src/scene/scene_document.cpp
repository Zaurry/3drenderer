#include "scene/scene_document.h"

#include "core/io/atomic_file.h"
#include "render/scene_intersector.h"
#include "scene/scene_asset_loader.h"
#include "scene/gltf_loader.h"

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace renderer {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kMinimumScale = 1.0e-6f;
constexpr int kMaximumHierarchyDepth = 1024;
constexpr std::size_t kMaximumHistory = 256;

void build_picking_acceleration(
    const std::shared_ptr<SceneMeshAsset>& asset) {
    asset->picking_intersector =
        std::make_shared<SceneIntersector>(asset->local_scene);
}

constexpr int Material::* kMaterialTextureIds[] = {
    &Material::diffuse_texture_id,
    &Material::opacity_texture_id,
    &Material::bump_texture_id,
    &Material::base_color_texture_id,
    &Material::metallic_roughness_texture_id,
    &Material::normal_texture_id,
    &Material::occlusion_texture_id,
    &Material::emissive_texture_id,
    &Material::specular_texture_id,
    &Material::specular_color_texture_id,
    &Material::specular_glossiness_texture_id,
};

int append_unique_texture(
    std::vector<ImageTexture>& textures,
    const ImageTexture& texture) {
    const auto found = std::find_if(
        textures.begin(),
        textures.end(),
        [&texture](const ImageTexture& candidate) {
            return candidate.same_resource_view(texture);
        });
    if (found != textures.end()) {
        return static_cast<int>(std::distance(textures.begin(), found));
    }
    const int id = static_cast<int>(textures.size());
    textures.push_back(texture);
    return id;
}

std::vector<int> append_unique_textures(
    std::vector<ImageTexture>& textures,
    const std::vector<ImageTexture>& local_textures) {
    std::vector<int> remap;
    remap.reserve(local_textures.size());
    for (const ImageTexture& texture : local_textures) {
        remap.push_back(append_unique_texture(textures, texture));
    }
    return remap;
}

void remap_material_textures(
    Material& material,
    const std::vector<int>& texture_remap) {
    for (int Material::* field : kMaterialTextureIds) {
        int& texture_id = material.*field;
        if (texture_id < 0) {
            continue;
        }
        if (static_cast<std::size_t>(texture_id) >= texture_remap.size()) {
            throw std::runtime_error("material texture index is out of range");
        }
        texture_id = texture_remap[static_cast<std::size_t>(texture_id)];
    }
}

Vec3 transform_point(const Mat4& matrix, const Vec3& point) {
    const Vec4 transformed = matrix * Vec4(point.x(), point.y(), point.z(), 1.0f);
    return transformed.head<3>() / transformed.w();
}

Bounds3 transform_bounds(const Bounds3& bounds, const Mat4& matrix) {
    Bounds3 transformed;
    for (int corner = 0; corner < 8; ++corner) {
        transformed.expand(transform_point(
            matrix,
            Vec3(
                (corner & 1) != 0 ? bounds.max.x() : bounds.min.x(),
                (corner & 2) != 0 ? bounds.max.y() : bounds.min.y(),
                (corner & 4) != 0 ? bounds.max.z() : bounds.min.z())));
    }
    return transformed;
}

bool finite_bounds(const Bounds3& bounds) {
    return bounds.min.allFinite() && bounds.max.allFinite() &&
        (bounds.max.array() >= bounds.min.array()).all();
}

std::string object_type_name(SceneObjectType type) {
    switch (type) {
        case SceneObjectType::Group:
            return "group";
        case SceneObjectType::Mesh:
            return "mesh";
        case SceneObjectType::PointLight:
            return "point_light";
        case SceneObjectType::DirectionalLight:
            return "directional_light";
        case SceneObjectType::SpotLight:
            return "spot_light";
        case SceneObjectType::Camera:
            return "camera";
    }
    return "group";
}

SceneObjectType parse_object_type(const std::string& value) {
    if (value == "group") {
        return SceneObjectType::Group;
    }
    if (value == "mesh") {
        return SceneObjectType::Mesh;
    }
    if (value == "point_light") {
        return SceneObjectType::PointLight;
    }
    if (value == "directional_light") {
        return SceneObjectType::DirectionalLight;
    }
    if (value == "spot_light") {
        return SceneObjectType::SpotLight;
    }
    if (value == "camera") {
        return SceneObjectType::Camera;
    }
    throw std::runtime_error("unknown scene object type: " + value);
}

std::string material_type_name(MaterialType type) {
    switch (type) {
        case MaterialType::Diffuse:
            return "diffuse";
        case MaterialType::Metal:
            return "metal";
        case MaterialType::Dielectric:
            return "dielectric";
        case MaterialType::Emissive:
            return "emissive";
        case MaterialType::Pbr:
            return "pbr";
    }
    return "diffuse";
}

MaterialType parse_material_type(const std::string& value) {
    if (value == "diffuse") {
        return MaterialType::Diffuse;
    }
    if (value == "metal") {
        return MaterialType::Metal;
    }
    if (value == "dielectric") {
        return MaterialType::Dielectric;
    }
    if (value == "emissive") {
        return MaterialType::Emissive;
    }
    if (value == "pbr") {
        return MaterialType::Pbr;
    }
    throw std::runtime_error("unknown material type: " + value);
}

bool valid_material_type(MaterialType type) {
    switch (type) {
        case MaterialType::Diffuse:
        case MaterialType::Metal:
        case MaterialType::Dielectric:
        case MaterialType::Emissive:
        case MaterialType::Pbr:
            return true;
    }
    return false;
}

bool valid_material_override(const SceneMaterialOverride& material_override) {
    return valid_material_type(material_override.type) &&
        material_override.base_color.allFinite() &&
        material_override.emission.allFinite() &&
        (material_override.base_color.array() >= 0.0f).all() &&
        (material_override.emission.array() >= 0.0f).all() &&
        std::isfinite(material_override.roughness) &&
        material_override.roughness >= 0.0f &&
        material_override.roughness <= 1.0f &&
        std::isfinite(material_override.metallic) &&
        material_override.metallic >= 0.0f &&
        material_override.metallic <= 1.0f &&
        std::isfinite(material_override.ior) &&
        material_override.ior > 0.0f &&
        std::isfinite(material_override.opacity) &&
        material_override.opacity >= 0.0f &&
        material_override.opacity <= 1.0f &&
        std::isfinite(material_override.alpha_cutoff) &&
        material_override.alpha_cutoff >= 0.0f &&
        material_override.alpha_cutoff <= 1.0f &&
        std::isfinite(material_override.bump_scale) &&
        std::isfinite(material_override.normal_scale) &&
        std::isfinite(material_override.occlusion_strength) &&
        material_override.occlusion_strength >= 0.0f &&
        material_override.occlusion_strength <= 1.0f;
}

SceneMaterialOverride material_override_from_source(
    std::size_t material_slot,
    const Material& material) {
    SceneMaterialOverride result;
    result.material_slot = static_cast<std::uint32_t>(material_slot);
    result.type = material.type;
    result.base_color = material.base_color;
    result.emission = material.emission;
    result.roughness = material.roughness;
    result.metallic = material.metallic;
    result.ior = material.ior;
    result.opacity = material.opacity;
    result.alpha_cutoff = material.alpha_cutoff;
    result.bump_scale = material.bump_scale;
    result.normal_scale = material.normal_scale;
    result.occlusion_strength = material.occlusion_strength;
    result.alpha_mode = material.alpha_mode;
    result.two_sided = material.two_sided;
    result.use_diffuse_texture = material.diffuse_texture_id >= 0;
    result.use_opacity_texture = material.opacity_texture_id >= 0;
    result.use_bump_texture = material.bump_texture_id >= 0;
    result.use_base_color_texture = material.base_color_texture_id >= 0;
    result.use_metallic_roughness_texture = material.metallic_roughness_texture_id >= 0;
    result.use_normal_texture = material.normal_texture_id >= 0;
    result.use_occlusion_texture = material.occlusion_texture_id >= 0;
    result.use_emissive_texture = material.emissive_texture_id >= 0;
    return result;
}

void apply_material_override(
    Material& material,
    const SceneMaterialOverride& material_override) {
    material.type = material_override.type;
    material.base_color = material_override.base_color;
    material.emission = material_override.emission;
    material.roughness = material_override.roughness;
    material.metallic = material_override.metallic;
    if (material.pbr_workflow == PbrWorkflow::SpecularGlossiness &&
        material_override.type == MaterialType::Pbr) {
        material.glossiness = 1.0f - material_override.roughness;
    }
    material.ior = material_override.ior;
    material.opacity = material_override.opacity;
    material.alpha_cutoff = material_override.alpha_cutoff;
    material.bump_scale = material_override.bump_scale;
    material.normal_scale = material_override.normal_scale;
    material.occlusion_strength = material_override.occlusion_strength;
    material.alpha_mode = material_override.alpha_mode;
    material.two_sided = material_override.two_sided;
    if (!material_override.use_diffuse_texture) {
        material.diffuse_texture_id = -1;
    }
    if (!material_override.use_opacity_texture) {
        material.opacity_texture_id = -1;
    }
    if (!material_override.use_bump_texture) {
        material.bump_texture_id = -1;
    }
    if (!material_override.use_base_color_texture) {
        material.base_color_texture_id = -1;
    }
    if (!material_override.use_metallic_roughness_texture) {
        material.metallic_roughness_texture_id = -1;
    }
    if (!material_override.use_normal_texture) {
        material.normal_texture_id = -1;
    }
    if (!material_override.use_occlusion_texture) {
        material.occlusion_texture_id = -1;
    }
    if (!material_override.use_emissive_texture) {
        material.emissive_texture_id = -1;
    }
}

nlohmann::json vec3_json(const Vec3& value) {
    return nlohmann::json::array({value.x(), value.y(), value.z()});
}

Vec3 parse_vec3(const nlohmann::json& value, const char* field) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string(field) + " must contain three numbers");
    }
    Vec3 result(
        value.at(0).get<float>(),
        value.at(1).get<float>(),
        value.at(2).get<float>());
    if (!result.allFinite()) {
        throw std::runtime_error(std::string(field) + " must be finite");
    }
    return result;
}

std::filesystem::path normalized_absolute(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(path, error);
    return (error ? std::filesystem::absolute(path) : canonical).lexically_normal();
}

std::string lowercase_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension;
}

std::string alpha_mode_name(AlphaMode mode) {
    switch (mode) {
        case AlphaMode::Opaque:
            return "opaque";
        case AlphaMode::Mask:
            return "mask";
        case AlphaMode::Blend:
            return "blend";
    }
    return "opaque";
}

AlphaMode parse_alpha_mode(const std::string& value) {
    if (value == "opaque") {
        return AlphaMode::Opaque;
    }
    if (value == "mask") {
        return AlphaMode::Mask;
    }
    if (value == "blend") {
        return AlphaMode::Blend;
    }
    throw std::runtime_error("unknown alpha mode: " + value);
}

const char* camera_projection_name(SceneCameraProjection projection) {
    return projection == SceneCameraProjection::Orthographic
        ? "orthographic"
        : "perspective";
}

SceneCameraProjection parse_camera_projection(const std::string& value) {
    if (value == "perspective") {
        return SceneCameraProjection::Perspective;
    }
    if (value == "orthographic") {
        return SceneCameraProjection::Orthographic;
    }
    throw std::runtime_error("unknown camera projection: " + value);
}

bool asset_extension(const std::filesystem::path& path) {
    const std::string extension = lowercase_extension(path);
    return extension == ".obj" || extension == ".gltf" || extension == ".glb";
}

MaterialSlot validated_material_slot(
    int material_id,
    std::size_t material_count) {
    if (material_id == -1) {
        return MaterialSlot::missing();
    }
    if (material_id < -1) {
        throw std::runtime_error(
            "mesh primitive material slot must be -1 or non-negative");
    }
    if (static_cast<std::size_t>(material_id) >= material_count) {
        throw std::runtime_error(
            "mesh primitive material slot is out of range");
    }
    return MaterialSlot::bound(
        static_cast<std::uint32_t>(material_id));
}

bool decompose_trs_matrix(const Mat4& matrix, SceneTrs& output) {
    if (!matrix.allFinite()) {
        return false;
    }
    const Vec4 affine_row = matrix.row(3).transpose();
    if (!affine_row.isApprox(Vec4(0.0f, 0.0f, 0.0f, 1.0f), 1.0e-6f)) {
        return false;
    }

    SceneTrs candidate;
    candidate.translation = matrix.topRightCorner<3, 1>();
    Mat3 linear = matrix.topLeftCorner<3, 3>();
    candidate.scale = Vec3(
        linear.col(0).norm(),
        linear.col(1).norm(),
        linear.col(2).norm());
    if ((candidate.scale.array() < kMinimumScale).any()) {
        return false;
    }
    if (linear.determinant() < 0.0f) {
        Eigen::Index reflected_axis = 0;
        candidate.scale.cwiseAbs().maxCoeff(&reflected_axis);
        candidate.scale[reflected_axis] = -candidate.scale[reflected_axis];
    }

    Mat3 rotation = linear;
    for (int axis = 0; axis < 3; ++axis) {
        rotation.col(axis) /= candidate.scale[axis];
    }
    if (!rotation.transpose().isApprox(rotation.inverse(), 1.0e-5f) ||
        std::abs(rotation.determinant() - 1.0f) > 1.0e-5f) {
        return false;
    }
    const Vec3 zyx = rotation.eulerAngles(2, 1, 0);
    candidate.rotation_degrees =
        Vec3(zyx.z(), zyx.y(), zyx.x()) * (180.0f / kPi);
    if (!candidate.valid()) {
        return false;
    }
    const float scale = std::max(1.0f, matrix.cwiseAbs().maxCoeff());
    if ((candidate.matrix() - matrix).cwiseAbs().maxCoeff() >
        scale * 1.0e-5f) {
        return false;
    }
    output = candidate;
    return true;
}

}  // namespace

Mat4 SceneTrs::matrix() const {
    const Vec3 radians = rotation_degrees * (kPi / 180.0f);
    const Eigen::Affine3f transform =
        Eigen::Translation3f(translation) *
        Eigen::AngleAxisf(radians.z(), Vec3::UnitZ()) *
        Eigen::AngleAxisf(radians.y(), Vec3::UnitY()) *
        Eigen::AngleAxisf(radians.x(), Vec3::UnitX()) *
        Eigen::Scaling(scale);
    return transform.matrix();
}

bool SceneTrs::valid() const {
    return translation.allFinite() && rotation_degrees.allFinite() && scale.allFinite() &&
        std::abs(scale.x()) >= kMinimumScale &&
        std::abs(scale.y()) >= kMinimumScale &&
        std::abs(scale.z()) >= kMinimumScale;
}

Mat4 SceneTransform::matrix() const {
    return local_matrix;
}

bool SceneTransform::valid() const {
    if (!local_matrix.allFinite() ||
        !local_matrix.row(3).transpose().isApprox(
            Vec4(0.0f, 0.0f, 0.0f, 1.0f),
            1.0e-6f)) {
        return false;
    }
    const float determinant = local_matrix.topLeftCorner<3, 3>().determinant();
    return std::isfinite(determinant) && std::abs(determinant) >= kMinimumScale;
}

SceneTransform SceneTransform::from_trs(const SceneTrs& trs) {
    SceneTransform result;
    result.local_matrix = trs.matrix();
    return result;
}

nlohmann::json matrix_json(const Mat4& matrix) {
    nlohmann::json result = nlohmann::json::array();
    for (int row = 0; row < 4; ++row) {
        result.push_back(nlohmann::json::array({
            matrix(row, 0),
            matrix(row, 1),
            matrix(row, 2),
            matrix(row, 3)}));
    }
    return result;
}

Mat4 parse_matrix(const nlohmann::json& value, const char* field) {
    if (!value.is_array() || value.size() != 4) {
        throw std::runtime_error(
            std::string(field) + " must contain four rows");
    }
    Mat4 result;
    for (int row = 0; row < 4; ++row) {
        const nlohmann::json& source_row = value.at(row);
        if (!source_row.is_array() || source_row.size() != 4) {
            throw std::runtime_error(
                std::string(field) + " rows must contain four numbers");
        }
        for (int column = 0; column < 4; ++column) {
            result(row, column) = source_row.at(column).get<float>();
        }
    }
    if (!result.allFinite()) {
        throw std::runtime_error(std::string(field) + " must be finite");
    }
    return result;
}

std::optional<SceneTrs> SceneTransform::trs() const {
    SceneTrs result;
    if (!decompose_trs_matrix(local_matrix, result)) {
        return std::nullopt;
    }
    return result;
}

SceneDocument::SceneDocument() {
    history_.push_back(state_);
}

SceneDocument SceneDocument::from_scene(
    Scene scene,
    std::string name,
    std::string builtin_id) {
    SceneDocument document;
    const std::vector<Sphere> procedural_spheres = scene.spheres;
    std::shared_ptr<Scene> render_geometry;
    if (!procedural_spheres.empty()) {
        render_geometry = std::make_shared<Scene>(scene);
        render_geometry->spheres.clear();
    }
    // Picking keeps the same canonical triangles as the shared unit-sphere
    // render asset, while snapshots represent center/radius as instances.
    tessellate_spheres(scene);
    document.state_.environment = scene.environment;
    document.state_.environment_map = scene.environment_map;
    document.state_.environment_path = scene.environment_map
        ? scene.environment_map->source_path()
        : std::filesystem::path();
    document.state_.environment_intensity = scene.environment_intensity;
    document.state_.environment_rotation_degrees = scene.environment_rotation_degrees;
    document.state_.environment_background_visible = scene.environment_background_visible;
    std::vector<PointLight> point_lights =
        std::move(scene.point_lights);
    std::vector<DirectionalLight> directional_lights =
        std::move(scene.directional_lights);
    std::vector<SpotLight> spot_lights = std::move(scene.spot_lights);
    scene.point_lights.clear();
    scene.directional_lights.clear();
    scene.spot_lights.clear();
    auto asset = std::make_shared<SceneMeshAsset>();
    asset->id = document.next_asset_id_++;
    asset->source_path.clear();
    asset->builtin_id = std::move(builtin_id);
    asset->local_scene = std::move(scene);
    asset->render_geometry = std::move(render_geometry);
    asset->procedural_spheres = procedural_spheres;
    asset->material_names.reserve(asset->local_scene.materials.size());
    for (std::size_t index = 0; index < asset->local_scene.materials.size(); ++index) {
        asset->material_names.push_back(
            "Material " + std::to_string(index + 1));
    }
    for (const Triangle& triangle : asset->local_scene.triangles) {
        asset->local_bounds.expand(triangle.bounds());
    }
    for (const Sphere& sphere : asset->local_scene.spheres) {
        asset->local_bounds.expand(sphere.bounds());
    }
    build_picking_acceleration(asset);
    document.assets_.push_back(asset);
    SceneObject object;
    object.id = document.next_object_id_++;
    object.name = std::move(name);
    object.type = SceneObjectType::Mesh;
    object.asset_id = asset->id;
    document.state_.objects.push_back(std::move(object));
    for (std::size_t index = 0;
         index < point_lights.size();
         ++index) {
        const ObjectId light_id = document.create_point_light(
            "Point Light " + std::to_string(index + 1),
            point_lights[index].position,
            point_lights[index].intensity);
        document.find_mutable(light_id)->light_range = point_lights[index].range;
    }
    for (std::size_t index = 0; index < spot_lights.size(); ++index) {
        document.create_spot_light(
            "Spot Light " + std::to_string(index + 1),
            spot_lights[index].position,
            spot_lights[index].direction,
            spot_lights[index].intensity,
            spot_lights[index].range,
            spot_lights[index].inner_cone_radians,
            spot_lights[index].outer_cone_radians);
    }
    for (std::size_t index = 0;
         index < directional_lights.size();
         ++index) {
        document.create_directional_light(
            "Directional Light " +
                std::to_string(index + 1),
            directional_lights[index].direction,
            directional_lights[index].radiance);
    }
    document.history_.assign(1, document.state_);
    document.uncheckpointed_changes_ = false;
    document.object_index_dirty_ = true;
    document.spatial_cache_dirty_ = true;
    advance_scene_revisions(
        document.revisions_,
        SceneRevisionDomain::All);
    document.snapshot_dirty_ = true;
    return document;
}

const std::vector<SceneObject>& SceneDocument::objects() const {
    return state_.objects;
}

std::vector<std::shared_ptr<const SceneMeshAsset>> SceneDocument::assets() const {
    std::vector<std::shared_ptr<const SceneMeshAsset>> result;
    result.reserve(assets_.size());
    for (const auto& asset : assets_) {
        result.push_back(asset);
    }
    return result;
}

const SceneMeshAsset* SceneDocument::asset_for_object(ObjectId id) const {
    const SceneObject* object = find(id);
    if (!object || object->type != SceneObjectType::Mesh) {
        return nullptr;
    }
    const auto asset = find_asset(object->asset_id);
    return asset.get();
}

const SceneObject* SceneDocument::find(ObjectId id) const {
    if (object_index_dirty_ || object_indices_.size() != state_.objects.size()) {
        rebuild_object_index();
    }
    const auto found = object_indices_.find(id);
    return found == object_indices_.end()
        ? nullptr
        : &state_.objects[found->second];
}

SceneObject* SceneDocument::find_mutable(ObjectId id) {
    return const_cast<SceneObject*>(
        static_cast<const SceneDocument&>(*this).find(id));
}

void SceneDocument::rebuild_object_index() const {
    object_indices_.clear();
    children_by_parent_.clear();
    object_indices_.reserve(state_.objects.size());
    children_by_parent_.reserve(state_.objects.size());
    for (std::size_t index = 0; index < state_.objects.size(); ++index) {
        const SceneObject& object = state_.objects[index];
        object_indices_.emplace(object.id, index);
        children_by_parent_[object.parent_id].push_back(object.id);
    }
    object_index_dirty_ = false;
}

void SceneDocument::mark_changed(
    SceneRevisionDomain domains,
    bool hierarchy_changed) {
    snapshot_dirty_ = true;
    uncheckpointed_changes_ = true;
    advance_scene_revisions(revisions_, domains);
    if (hierarchy_changed ||
        has_revision_domain(domains, SceneRevisionDomain::Topology)) {
        object_index_dirty_ = true;
    }
    if (hierarchy_changed ||
        has_revision_domain(domains, SceneRevisionDomain::Topology) ||
        has_revision_domain(domains, SceneRevisionDomain::Transforms)) {
        spatial_cache_dirty_ = true;
    }
}

std::vector<ObjectId> SceneDocument::children(ObjectId parent_id) const {
    if (object_index_dirty_ || object_indices_.size() != state_.objects.size()) {
        rebuild_object_index();
    }
    const auto found = children_by_parent_.find(parent_id);
    return found == children_by_parent_.end()
        ? std::vector<ObjectId>()
        : found->second;
}

bool SceneDocument::set_object_name(ObjectId id, std::string name) {
    SceneObject* object = find_mutable(id);
    if (!object || object->name == name) {
        return false;
    }
    object->name = std::move(name);
    uncheckpointed_changes_ = true;
    return true;
}

bool SceneDocument::set_object_visible(ObjectId id, bool visible) {
    SceneObject* object = find_mutable(id);
    if (!object || object->visible == visible) {
        return false;
    }
    object->visible = visible;
    mark_changed(SceneRevisionDomain::Topology);
    return true;
}

bool SceneDocument::set_object_locked(ObjectId id, bool locked) {
    SceneObject* object = find_mutable(id);
    if (!object || object->locked == locked) {
        return false;
    }
    object->locked = locked;
    uncheckpointed_changes_ = true;
    return true;
}

bool SceneDocument::set_camera_properties(
    ObjectId id,
    const SceneCameraProperties& properties) {
    SceneObject* object = find_mutable(id);
    if (!object || object->type != SceneObjectType::Camera || object->locked ||
        !std::isfinite(properties.vertical_fov_degrees) ||
        !std::isfinite(properties.aspect_ratio) ||
        !std::isfinite(properties.x_magnification) ||
        !std::isfinite(properties.y_magnification) ||
        !std::isfinite(properties.near_plane) ||
        !std::isfinite(properties.far_plane) ||
        properties.vertical_fov_degrees < 1.0f ||
        properties.vertical_fov_degrees > 179.0f ||
        properties.x_magnification < 1.0e-4f ||
        properties.y_magnification < 1.0e-4f ||
        properties.near_plane < 1.0e-5f ||
        properties.far_plane <= properties.near_plane) {
        return false;
    }
    object->camera_projection = properties.projection;
    object->camera_vertical_fov_degrees = properties.vertical_fov_degrees;
    object->camera_aspect_ratio = properties.aspect_ratio;
    object->camera_x_magnification = properties.x_magnification;
    object->camera_y_magnification = properties.y_magnification;
    object->camera_near_plane = properties.near_plane;
    object->camera_far_plane = properties.far_plane;
    mark_changed(SceneRevisionDomain::Transforms);
    return true;
}

bool SceneDocument::set_light_properties(
    ObjectId id,
    const SceneLightProperties& properties) {
    SceneObject* object = find_mutable(id);
    if (!object || object->locked ||
        (object->type != SceneObjectType::PointLight &&
         object->type != SceneObjectType::DirectionalLight &&
         object->type != SceneObjectType::SpotLight) ||
        !properties.color.allFinite() ||
        !std::isfinite(properties.range) ||
        !std::isfinite(properties.spot_inner_cone_radians) ||
        !std::isfinite(properties.spot_outer_cone_radians)) {
        return false;
    }
    object->light_color = properties.color.cwiseMax(Color::Zero());
    object->light_range = std::max(0.0f, properties.range);
    object->spot_inner_cone_radians = std::clamp(
        properties.spot_inner_cone_radians, 0.0f, kPi * 0.5f);
    object->spot_outer_cone_radians = std::clamp(
        properties.spot_outer_cone_radians,
        object->spot_inner_cone_radians,
        kPi * 0.5f);
    mark_changed(SceneRevisionDomain::Lighting);
    return true;
}

ObjectId SceneDocument::create_group(std::string name, ObjectId parent_id) {
    SceneObject object;
    object.id = next_object_id_++;
    object.parent_id = parent_id;
    object.name = std::move(name);
    object.type = SceneObjectType::Group;
    state_.objects.push_back(std::move(object));
    mark_changed(SceneRevisionDomain::All, true);
    return state_.objects.back().id;
}

ObjectId SceneDocument::create_point_light(
    std::string name,
    const Vec3& position,
    const Color& intensity,
    ObjectId parent_id) {
    const ObjectId id = create_group(std::move(name), parent_id);
    SceneObject* object = find_mutable(id);
    object->type = SceneObjectType::PointLight;
    SceneTrs transform;
    transform.translation = position;
    object->transform = SceneTransform::from_trs(transform);
    object->light_color = intensity;
    return id;
}

ObjectId SceneDocument::create_directional_light(
    std::string name,
    const Vec3& direction,
    const Color& radiance,
    ObjectId parent_id) {
    const ObjectId id = create_group(std::move(name), parent_id);
    SceneObject* object = find_mutable(id);
    object->type = SceneObjectType::DirectionalLight;
    object->light_color = radiance;
    const Vec3 normalized = usable_direction(direction)
        ? direction.normalized()
        : Vec3(0.0f, -1.0f, 0.0f);
    const Eigen::Quaternionf rotation =
        Eigen::Quaternionf::FromTwoVectors(Vec3(0.0f, 0.0f, -1.0f), normalized);
    const Vec3 zyx = rotation.toRotationMatrix().eulerAngles(2, 1, 0);
    SceneTrs transform;
    transform.rotation_degrees =
        Vec3(zyx.z(), zyx.y(), zyx.x()) * (180.0f / kPi);
    object->transform = SceneTransform::from_trs(transform);
    return id;
}

ObjectId SceneDocument::create_spot_light(
    std::string name,
    const Vec3& position,
    const Vec3& direction,
    const Color& intensity,
    float range,
    float inner_cone_radians,
    float outer_cone_radians,
    ObjectId parent_id) {
    const ObjectId id = create_directional_light(
        std::move(name),
        direction,
        intensity,
        parent_id);
    SceneObject* object = find_mutable(id);
    object->type = SceneObjectType::SpotLight;
    SceneTrs transform = object->transform.trs().value_or(SceneTrs{});
    transform.translation = position;
    object->transform = SceneTransform::from_trs(transform);
    object->light_range = std::max(0.0f, range);
    object->spot_inner_cone_radians = std::max(0.0f, inner_cone_radians);
    object->spot_outer_cone_radians = std::clamp(
        outer_cone_radians,
        object->spot_inner_cone_radians,
        kPi * 0.5f);
    return id;
}

ObjectId SceneDocument::create_camera(
    std::string name,
    SceneCameraProjection projection,
    ObjectId parent_id) {
    const ObjectId id = create_group(std::move(name), parent_id);
    SceneObject* object = find_mutable(id);
    object->type = SceneObjectType::Camera;
    object->camera_projection = projection;
    return id;
}

std::shared_ptr<SceneMeshAsset> SceneDocument::find_asset(AssetId id) const {
    const auto found = std::find_if(
        assets_.begin(),
        assets_.end(),
        [id](const std::shared_ptr<SceneMeshAsset>& asset) { return asset->id == id; });
    return found == assets_.end() ? nullptr : *found;
}

std::shared_ptr<SceneMeshAsset> SceneDocument::load_asset(
    const std::filesystem::path& path,
    int width,
    int height,
    int source_mesh_index) {
    const std::filesystem::path normalized = normalized_absolute(path);
    for (const auto& asset : assets_) {
        if (!asset->source_path.empty() &&
            normalized_absolute(asset->source_path) == normalized &&
            asset->source_mesh_index == source_mesh_index) {
            return asset;
        }
    }

    LoadedScene loaded{
        Scene(),
        Camera(
            Vec3(0.0f, 0.0f, 1.0f),
            Vec3::Zero(),
            Vec3(0.0f, 1.0f, 0.0f),
            45.0f,
            1.0f),
        Bounds3()};
    const std::string extension = lowercase_extension(normalized);
    if (extension == ".gltf" || extension == ".glb") {
        LoadedGltfScene gltf = load_gltf_scene(normalized, width, height);
        const int mesh_index = source_mesh_index >= 0 ? source_mesh_index : 0;
        if (mesh_index < 0 || static_cast<std::size_t>(mesh_index) >= gltf.meshes.size()) {
            throw std::runtime_error("glTF source mesh index is out of range");
        }
        loaded = std::move(gltf.meshes[static_cast<std::size_t>(mesh_index)]);
        source_mesh_index = mesh_index;
        for (const std::string& warning : gltf.warnings) {
            warnings_.push_back(normalized.string() + ": " + warning);
        }
    } else {
        loaded = load_scene_asset(normalized.string(), width, height);
        source_mesh_index = -1;
    }
    return store_loaded_asset(std::move(loaded), normalized, source_mesh_index);
}

std::shared_ptr<SceneMeshAsset> SceneDocument::store_loaded_asset(
    LoadedScene loaded,
    const std::filesystem::path& normalized_path,
    int source_mesh_index) {
    auto asset = std::make_shared<SceneMeshAsset>();
    asset->id = next_asset_id_++;
    asset->source_path = normalized_path;
    asset->source_mesh_index = source_mesh_index;
    asset->local_scene = std::move(loaded.scene);
    asset->local_scene.directional_lights.clear();
    asset->local_scene.point_lights.clear();
    asset->local_scene.spot_lights.clear();
    asset->warnings = std::move(loaded.warnings);
    asset->material_names = std::move(loaded.material_names);
    asset->local_bounds = loaded.bounds;
    build_picking_acceleration(asset);
    assets_.push_back(asset);
    for (const std::string& warning : asset->warnings) {
        warnings_.push_back(normalized_path.string() + ": " + warning);
    }
    return asset;
}

ObjectId SceneDocument::import_gltf(
    const std::filesystem::path& path,
    ObjectId parent_id,
    int width,
    int height) {
    const std::filesystem::path normalized = normalized_absolute(path);
    LoadedGltfScene loaded = load_gltf_scene(normalized, width, height);
    std::vector<AssetId> mesh_assets(loaded.meshes.size(), kInvalidAssetId);
    for (std::size_t mesh_index = 0; mesh_index < loaded.meshes.size(); ++mesh_index) {
        auto asset = std::make_shared<SceneMeshAsset>();
        asset->id = next_asset_id_++;
        asset->source_path = normalized;
        asset->source_mesh_index = static_cast<int>(mesh_index);
        asset->local_scene = std::move(loaded.meshes[mesh_index].scene);
        asset->local_scene.directional_lights.clear();
        asset->local_scene.point_lights.clear();
        asset->local_scene.spot_lights.clear();
        asset->warnings = std::move(loaded.meshes[mesh_index].warnings);
        asset->material_names = std::move(loaded.meshes[mesh_index].material_names);
        asset->local_bounds = loaded.meshes[mesh_index].bounds;
        build_picking_acceleration(asset);
        mesh_assets[mesh_index] = asset->id;
        for (const std::string& warning : asset->warnings) {
            warnings_.push_back(normalized.string() + ": " + warning);
        }
        assets_.push_back(std::move(asset));
    }

    const ObjectId root = create_group(path.stem().string(), parent_id);
    std::vector<ObjectId> node_objects;
    node_objects.reserve(loaded.nodes.size());
    for (std::size_t node_index = 0; node_index < loaded.nodes.size(); ++node_index) {
        const GltfNodeAsset& source = loaded.nodes[node_index];
        const ObjectId node_parent = source.parent_index >= 0
            ? node_objects.at(static_cast<std::size_t>(source.parent_index))
            : root;
        const ObjectId node_id = create_group(source.name, node_parent);
        SceneObject* node = find_mutable(node_id);
        node->transform.local_matrix = source.local_transform;
        if (!node->transform.valid()) {
            throw std::runtime_error(
                "glTF node transform is not a finite invertible affine matrix: " +
                source.name);
        }
        node_objects.push_back(node_id);

        if (source.mesh_index >= 0) {
            if (static_cast<std::size_t>(source.mesh_index) >= mesh_assets.size()) {
                throw std::runtime_error("glTF node mesh index is out of range");
            }
            SceneObject mesh;
            mesh.id = next_object_id_++;
            mesh.parent_id = node_id;
            mesh.name = source.name + " Mesh";
            mesh.type = SceneObjectType::Mesh;
            mesh.asset_id = mesh_assets[static_cast<std::size_t>(source.mesh_index)];
            state_.objects.push_back(std::move(mesh));
            object_index_dirty_ = true;
        }
        if (source.camera_index >= 0) {
            if (static_cast<std::size_t>(source.camera_index) >= loaded.cameras.size()) {
                throw std::runtime_error("glTF node camera index is out of range");
            }
            const GltfCameraAsset& camera_source =
                loaded.cameras[static_cast<std::size_t>(source.camera_index)];
            const ObjectId camera_id = create_camera(
                camera_source.name,
                camera_source.orthographic
                    ? SceneCameraProjection::Orthographic
                    : SceneCameraProjection::Perspective,
                node_id);
            SceneObject* camera = find_mutable(camera_id);
            camera->camera_vertical_fov_degrees =
                camera_source.vertical_fov_radians * (180.0f / kPi);
            camera->camera_aspect_ratio = camera_source.aspect_ratio;
            camera->camera_x_magnification = camera_source.x_magnification;
            camera->camera_y_magnification = camera_source.y_magnification;
            camera->camera_near_plane = camera_source.near_plane;
            camera->camera_far_plane = camera_source.far_plane;
        }
        switch (source.light_type) {
            case GltfNodeAsset::LightType::None:
                break;
            case GltfNodeAsset::LightType::Directional:
                create_directional_light(
                    source.name + " Light",
                    Vec3(0.0f, 0.0f, -1.0f),
                    source.light_color * source.light_intensity,
                    node_id);
                break;
            case GltfNodeAsset::LightType::Point: {
                const ObjectId light_id = create_point_light(
                    source.name + " Light",
                    Vec3::Zero(),
                    source.light_color * source.light_intensity,
                    node_id);
                find_mutable(light_id)->light_range = source.light_range;
                break;
            }
            case GltfNodeAsset::LightType::Spot:
                create_spot_light(
                    source.name + " Light",
                    Vec3::Zero(),
                    Vec3(0.0f, 0.0f, -1.0f),
                    source.light_color * source.light_intensity,
                    source.light_range,
                    source.spot_inner_cone_radians,
                    source.spot_outer_cone_radians,
                    node_id);
                break;
        }
    }
    for (const std::string& warning : loaded.warnings) {
        warnings_.push_back(normalized.string() + ": " + warning);
    }
    mark_changed(SceneRevisionDomain::All, true);
    return root;
}

ObjectId SceneDocument::import_obj(
    const std::filesystem::path& path,
    ObjectId parent_id,
    int width,
    int height) {
    const auto asset = load_asset(path, width, height);
    SceneObject object;
    object.id = next_object_id_++;
    object.parent_id = parent_id;
    object.name = path.stem().string();
    object.type = SceneObjectType::Mesh;
    object.asset_id = asset->id;
    state_.objects.push_back(std::move(object));
    mark_changed(SceneRevisionDomain::All, true);
    return state_.objects.back().id;
}

std::vector<ObjectId> SceneDocument::import_path(
    const std::filesystem::path& path,
    int width,
    int height) {
    const std::filesystem::path normalized = normalized_absolute(path);
    if (!std::filesystem::exists(normalized)) {
        throw std::runtime_error("asset path does not exist: " + normalized.string());
    }

    const State previous_state = state_;
    const std::size_t previous_asset_count = assets_.size();
    const std::size_t previous_warning_count = warnings_.size();
    const ObjectId previous_next_object_id = next_object_id_;
    const AssetId previous_next_asset_id = next_asset_id_;
    const bool previous_uncheckpointed_changes =
        uncheckpointed_changes_;
    try {
    std::vector<ObjectId> imported;
    if (std::filesystem::is_regular_file(normalized)) {
        if (!asset_extension(normalized)) {
            throw std::runtime_error("asset file is not OBJ, glTF, or GLB: " + normalized.string());
        }
        const std::string extension = lowercase_extension(normalized);
        if (extension == ".gltf" || extension == ".glb") {
            imported.push_back(import_gltf(normalized, kInvalidObjectId, width, height));
        } else {
            imported.push_back(import_obj(normalized, kInvalidObjectId, width, height));
        }
    } else if (std::filesystem::is_directory(normalized)) {
        std::vector<std::filesystem::path> files;
        std::error_code error;
        const auto options = std::filesystem::directory_options::skip_permission_denied;
        for (std::filesystem::recursive_directory_iterator iterator(normalized, options, error), end;
             iterator != end;
             iterator.increment(error)) {
            if (error) {
                warnings_.push_back("directory scan warning: " + error.message());
                error.clear();
                continue;
            }
            if (iterator->is_symlink(error) && iterator->is_directory(error)) {
                iterator.disable_recursion_pending();
            } else if (iterator->is_regular_file(error) && asset_extension(iterator->path())) {
                files.push_back(iterator->path());
            }
        }
        std::sort(files.begin(), files.end(), [&normalized](const auto& lhs, const auto& rhs) {
            return lhs.lexically_relative(normalized).generic_string() <
                rhs.lexically_relative(normalized).generic_string();
        });
        if (files.empty()) {
            throw std::runtime_error("asset directory contains no OBJ, glTF, or GLB files: " + normalized.string());
        }

        const ObjectId root = create_group(normalized.filename().string());
        std::unordered_map<std::string, ObjectId> groups;
        groups.emplace("", root);
        for (const std::filesystem::path& file : files) {
            const std::filesystem::path relative_parent =
                file.parent_path().lexically_relative(normalized);
            ObjectId parent = root;
            std::filesystem::path accumulated;
            for (const auto& component : relative_parent) {
                if (component.empty() || component == ".") {
                    continue;
                }
                accumulated /= component;
                const std::string key = accumulated.generic_string();
                const auto found = groups.find(key);
                if (found != groups.end()) {
                    parent = found->second;
                } else {
                    parent = create_group(component.string(), parent);
                    groups.emplace(key, parent);
                }
            }
            const std::string extension = lowercase_extension(file);
            imported.push_back(
                extension == ".gltf" || extension == ".glb"
                    ? import_gltf(file, parent, width, height)
                    : import_obj(file, parent, width, height));
        }
    } else {
        throw std::runtime_error("asset path is neither a file nor directory: " + normalized.string());
    }

    if (std::none_of(state_.objects.begin(), state_.objects.end(), [](const SceneObject& object) {
            return object.type == SceneObjectType::DirectionalLight ||
                object.type == SceneObjectType::PointLight ||
                object.type == SceneObjectType::SpotLight;
        })) {
        create_directional_light(
            "Sun",
            Vec3(-0.5f, -1.0f, -0.25f),
            Color(25.0f, 25.0f, 25.0f));
    }
    checkpoint();
    return imported;
    } catch (...) {
        state_ = previous_state;
        assets_.resize(previous_asset_count);
        warnings_.resize(previous_warning_count);
        next_object_id_ = previous_next_object_id;
        next_asset_id_ = previous_next_asset_id;
        object_index_dirty_ = true;
        spatial_cache_dirty_ = true;
        uncheckpointed_changes_ = previous_uncheckpointed_changes;
        snapshot_dirty_ = true;
        throw;
    }
}

ObjectId SceneDocument::clone_subtree(ObjectId source_id, ObjectId parent_id) {
    const SceneObject* source = find(source_id);
    if (!source) {
        return kInvalidObjectId;
    }
    SceneObject copy = *source;
    copy.id = next_object_id_++;
    copy.parent_id = parent_id;
    copy.name += " Copy";
    state_.objects.push_back(copy);
    object_index_dirty_ = true;
    const ObjectId copy_id = copy.id;
    for (ObjectId child_id : children(source_id)) {
        clone_subtree(child_id, copy_id);
    }
    return copy_id;
}

ObjectId SceneDocument::duplicate_subtree(ObjectId id) {
    const SceneObject* source = find(id);
    if (!source) {
        return kInvalidObjectId;
    }
    const ObjectId result = clone_subtree(id, source->parent_id);
    if (result != kInvalidObjectId) {
        mark_changed(SceneRevisionDomain::All, true);
        checkpoint();
    }
    return result;
}

bool SceneDocument::erase_subtree(ObjectId id) {
    if (!find(id)) {
        return false;
    }
    std::unordered_set<ObjectId> remove{id};
    bool changed = true;
    while (changed) {
        changed = false;
        for (const SceneObject& object : state_.objects) {
            if (remove.contains(object.parent_id) && remove.insert(object.id).second) {
                changed = true;
            }
        }
    }
    std::erase_if(state_.objects, [&remove](const SceneObject& object) {
        return remove.contains(object.id);
    });
    mark_changed(SceneRevisionDomain::All, true);
    checkpoint();
    return true;
}

bool SceneDocument::is_descendant(ObjectId candidate, ObjectId ancestor) const {
    ObjectId current = candidate;
    for (int depth = 0; depth < kMaximumHierarchyDepth && current != kInvalidObjectId; ++depth) {
        if (current == ancestor) {
            return true;
        }
        const SceneObject* object = find(current);
        current = object ? object->parent_id : kInvalidObjectId;
    }
    return false;
}

bool SceneDocument::is_effectively_visible(ObjectId id) const {
    ObjectId current = id;
    for (int depth = 0; depth < kMaximumHierarchyDepth && current != kInvalidObjectId; ++depth) {
        const SceneObject* object = find(current);
        if (!object || !object->visible) {
            return false;
        }
        current = object->parent_id;
    }
    return current == kInvalidObjectId;
}

bool SceneDocument::reparent(ObjectId id, ObjectId new_parent_id) {
    SceneObject* object = find_mutable(id);
    if (!object || id == new_parent_id ||
        (new_parent_id != kInvalidObjectId && !find(new_parent_id)) ||
        is_descendant(new_parent_id, id)) {
        return false;
    }
    const Mat4 old_world = world_matrix(id);
    const Mat4 parent_world = new_parent_id == kInvalidObjectId
        ? Mat4::Identity()
        : world_matrix(new_parent_id);
    SceneTransform next_transform;
    next_transform.local_matrix = parent_world.inverse() * old_world;
    if (!next_transform.valid()) {
        return false;
    }
    object->parent_id = new_parent_id;
    object->transform = next_transform;
    mark_changed(
        SceneRevisionDomain::Topology | SceneRevisionDomain::Transforms,
        true);
    checkpoint();
    return true;
}

bool SceneDocument::set_world_matrix(ObjectId id, const Mat4& world) {
    SceneObject* object = find_mutable(id);
    if (!object || object->locked) {
        return false;
    }
    const Mat4 parent_world = object->parent_id == kInvalidObjectId
        ? Mat4::Identity()
        : world_matrix(object->parent_id);
    SceneTransform next_transform;
    next_transform.local_matrix = parent_world.inverse() * world;
    if (!next_transform.valid()) {
        return false;
    }
    object->transform = next_transform;
    mark_changed(
        SceneRevisionDomain::Transforms | SceneRevisionDomain::Lighting);
    return true;
}

std::optional<SceneTrs> SceneDocument::local_trs(ObjectId id) const {
    const SceneObject* object = find(id);
    return object ? object->transform.trs() : std::nullopt;
}

bool SceneDocument::set_local_trs(ObjectId id, const SceneTrs& trs) {
    SceneObject* object = find_mutable(id);
    if (!object || object->locked || !trs.valid()) {
        return false;
    }
    object->transform = SceneTransform::from_trs(trs);
    mark_changed(
        SceneRevisionDomain::Transforms | SceneRevisionDomain::Lighting);
    return true;
}

const SceneMaterialOverride* SceneDocument::material_override(
    ObjectId id,
    std::size_t material_slot) const {
    const SceneObject* object = find(id);
    if (!object || object->type != SceneObjectType::Mesh) {
        return nullptr;
    }
    const auto found = std::find_if(
        object->material_overrides.begin(),
        object->material_overrides.end(),
        [material_slot](const SceneMaterialOverride& candidate) {
            return candidate.material_slot == material_slot;
        });
    return found == object->material_overrides.end() ? nullptr : &*found;
}

std::optional<SceneMaterialOverride> SceneDocument::material_properties(
    ObjectId id,
    std::size_t material_slot) const {
    const SceneMeshAsset* asset = asset_for_object(id);
    if (!asset || material_slot >= asset->local_scene.materials.size()) {
        return std::nullopt;
    }
    if (const SceneMaterialOverride* existing =
            material_override(id, material_slot)) {
        return *existing;
    }
    return material_override_from_source(
        material_slot,
        asset->local_scene.materials[material_slot]);
}

bool SceneDocument::set_material_override(
    ObjectId id,
    const SceneMaterialOverride& material_override_value) {
    SceneObject* object = find_mutable(id);
    const auto asset = object ? find_asset(object->asset_id) : nullptr;
    if (!object ||
        object->type != SceneObjectType::Mesh ||
        object->locked ||
        !asset ||
        material_override_value.material_slot >=
            asset->local_scene.materials.size() ||
        !valid_material_override(material_override_value)) {
        return false;
    }
    const auto found = std::find_if(
        object->material_overrides.begin(),
        object->material_overrides.end(),
        [&material_override_value](const SceneMaterialOverride& candidate) {
            return candidate.material_slot ==
                material_override_value.material_slot;
        });
    if (found == object->material_overrides.end()) {
        object->material_overrides.push_back(material_override_value);
        std::sort(
            object->material_overrides.begin(),
            object->material_overrides.end(),
            [](const SceneMaterialOverride& lhs, const SceneMaterialOverride& rhs) {
                return lhs.material_slot < rhs.material_slot;
            });
    } else {
        *found = material_override_value;
    }
    mark_changed(
        SceneRevisionDomain::MaterialBindings |
        SceneRevisionDomain::Materials |
        SceneRevisionDomain::Textures);
    return true;
}

bool SceneDocument::clear_material_override(
    ObjectId id,
    std::size_t material_slot) {
    SceneObject* object = find_mutable(id);
    if (!object ||
        object->type != SceneObjectType::Mesh ||
        object->locked) {
        return false;
    }
    const std::size_t previous_size = object->material_overrides.size();
    std::erase_if(
        object->material_overrides,
        [material_slot](const SceneMaterialOverride& candidate) {
            return candidate.material_slot == material_slot;
        });
    if (object->material_overrides.size() == previous_size) {
        return false;
    }
    mark_changed(
        SceneRevisionDomain::MaterialBindings |
        SceneRevisionDomain::Materials |
        SceneRevisionDomain::Textures);
    return true;
}

void SceneDocument::rebuild_spatial_cache() const {
    if (object_index_dirty_ || object_indices_.size() != state_.objects.size()) {
        rebuild_object_index();
    }
    world_matrices_.clear();
    world_bounds_.clear();
    world_matrices_.reserve(state_.objects.size());
    world_bounds_.reserve(state_.objects.size());

    std::vector<ObjectId> topology;
    topology.reserve(state_.objects.size());
    std::vector<ObjectId> stack;
    const auto roots = children(kInvalidObjectId);
    stack.insert(stack.end(), roots.rbegin(), roots.rend());
    while (!stack.empty()) {
        const ObjectId id = stack.back();
        stack.pop_back();
        const SceneObject* object = find(id);
        if (!object) {
            continue;
        }
        const Mat4 parent_world = object->parent_id == kInvalidObjectId
            ? Mat4::Identity()
            : world_matrices_.at(object->parent_id);
        world_matrices_[id] = parent_world * object->transform.matrix();
        topology.push_back(id);
        const auto descendants = children(id);
        stack.insert(stack.end(), descendants.rbegin(), descendants.rend());
        if (topology.size() > state_.objects.size()) {
            throw std::runtime_error("scene hierarchy contains a cycle");
        }
    }
    if (topology.size() != state_.objects.size()) {
        throw std::runtime_error("scene hierarchy contains an orphan or cycle");
    }

    for (ObjectId id : topology) {
        Bounds3 bounds;
        const SceneObject* object = find(id);
        if (object && object->type == SceneObjectType::Mesh) {
            const auto asset = find_asset(object->asset_id);
            if (asset && finite_bounds(asset->local_bounds)) {
                bounds.expand(transform_bounds(
                    asset->local_bounds,
                    world_matrices_.at(id)));
            }
        }
        world_bounds_.emplace(id, bounds);
    }
    for (auto iterator = topology.rbegin(); iterator != topology.rend(); ++iterator) {
        const SceneObject* object = find(*iterator);
        if (!object || object->parent_id == kInvalidObjectId) {
            continue;
        }
        const Bounds3 bounds = world_bounds_.at(*iterator);
        if (finite_bounds(bounds)) {
            world_bounds_.at(object->parent_id).expand(bounds);
        }
    }
    spatial_cache_dirty_ = false;
}

Mat4 SceneDocument::world_matrix_recursive(ObjectId id, int) const {
    return world_matrix(id);
}

Mat4 SceneDocument::world_matrix(ObjectId id) const {
    if (spatial_cache_dirty_) {
        rebuild_spatial_cache();
    }
    const auto found = world_matrices_.find(id);
    return found == world_matrices_.end() ? Mat4::Identity() : found->second;
}

Bounds3 SceneDocument::world_bounds(ObjectId id) const {
    if (spatial_cache_dirty_) {
        rebuild_spatial_cache();
    }
    const auto found = world_bounds_.find(id);
    return found == world_bounds_.end() ? Bounds3() : found->second;
}

Bounds3 SceneDocument::scene_bounds() const {
    if (spatial_cache_dirty_) {
        rebuild_spatial_cache();
    }
    Bounds3 bounds;
    for (const SceneObject& object : state_.objects) {
        if (object.type != SceneObjectType::Mesh ||
            !is_effectively_visible(object.id)) {
            continue;
        }
        const auto found = world_bounds_.find(object.id);
        if (found != world_bounds_.end() && finite_bounds(found->second)) {
            bounds.expand(found->second);
        }
    }
    if (!finite_bounds(bounds)) {
        return Bounds3(
            Vec3(-0.5f, -0.5f, -0.5f),
            Vec3(0.5f, 0.5f, 0.5f));
    }
    return bounds;
}

std::optional<ScenePickResult> SceneDocument::pick(const Ray& ray) const {
    std::optional<ScenePickResult> best;
    for (const SceneObject& object : state_.objects) {
        if (!is_effectively_visible(object.id) ||
            object.type != SceneObjectType::Mesh) {
            continue;
        }
        const auto asset = find_asset(object.asset_id);
        if (!asset) {
            continue;
        }
        const Mat4 world = world_matrix(object.id);
        const Mat4 inverse = world.inverse();
        const Vec3 local_origin = transform_point(inverse, ray.origin);
        const Vec4 direction4 = inverse * Vec4(
            ray.direction.x(), ray.direction.y(), ray.direction.z(), 0.0f);
        const Ray local_ray(local_origin, direction4.head<3>());
        if (!asset->picking_intersector) {
            continue;
        }
        HitRecord hit;
        if (asset->picking_intersector->intersect(
                local_ray,
                0.0f,
                best ? best->distance : std::numeric_limits<float>::infinity(),
                hit)) {
            best = ScenePickResult{object.id, hit.t};
        }
    }
    return best;
}

void SceneDocument::ensure_render_scene_snapshot() const {
    if (!snapshot_dirty_) {
        return;
    }

    RenderSceneSnapshot result;
    result.revisions = revisions_;
    result.environment = state_.environment;
    result.environment_map = state_.environment_map;
    result.environment_intensity = state_.environment_intensity;
    result.environment_rotation_degrees = state_.environment_rotation_degrees;
    result.environment_background_visible = state_.environment_background_visible;
    result.assets.reserve(assets_.size() + 1);
    result.instances.reserve(state_.objects.size());
    std::unordered_map<AssetId, int> asset_indices;
    std::unordered_map<AssetId, std::vector<int>> texture_remaps;
    bool has_procedural_spheres = false;

    for (const SceneObject& object : state_.objects) {
        if (object.type != SceneObjectType::Mesh) {
            continue;
        }
        const auto asset = find_asset(object.asset_id);
        if (!asset ||
            asset_indices.contains(asset->id)) {
            continue;
        }
        asset_indices.emplace(asset->id, -1);
        has_procedural_spheres = has_procedural_spheres ||
            !asset->procedural_spheres.empty();
        texture_remaps.emplace(
            asset->id,
            append_unique_textures(
                result.textures,
                asset->local_scene.textures));
        const std::shared_ptr<const Scene> render_geometry =
            asset->render_geometry
            ? asset->render_geometry
            : std::shared_ptr<const Scene>(asset, &asset->local_scene);
        if (render_geometry->spheres.empty() &&
            render_geometry->triangles.empty()) {
            continue;
        }
        const int asset_index =
            static_cast<int>(result.assets.size());
        asset_indices[asset->id] = asset_index;
        RenderSceneAssetSnapshot asset_view;
        asset_view.asset_id = asset->id;
        asset_view.geometry_revision = asset->geometry_revision;
        asset_view.local_scene = render_geometry;
        asset_view.sphere_material_slots.reserve(
            render_geometry->spheres.size());
        for (const Sphere& sphere : render_geometry->spheres) {
            asset_view.local_bounds.expand(sphere.bounds());
            asset_view.sphere_material_slots.push_back(
                validated_material_slot(
                    sphere.material_id(),
                    asset->local_scene.materials.size()));
        }
        asset_view.triangle_material_slots.reserve(
            render_geometry->triangles.size());
        for (const Triangle& triangle : render_geometry->triangles) {
            asset_view.local_bounds.expand(triangle.bounds());
            asset_view.triangle_material_slots.push_back(
                validated_material_slot(
                    triangle.material_id(),
                    asset->local_scene.materials.size()));
        }
        result.assets.push_back(std::move(asset_view));
    }

    int unit_sphere_asset_index = -1;
    if (has_procedural_spheres) {
        const auto& unit_geometry = canonical_unit_sphere_geometry();
        RenderSceneAssetSnapshot sphere_asset;
        sphere_asset.asset_id = std::numeric_limits<AssetId>::max();
        sphere_asset.geometry_revision = 1;
        sphere_asset.local_scene = unit_geometry;
        sphere_asset.local_bounds = Bounds3(
            -Vec3::Ones(),
            Vec3::Ones());
        sphere_asset.triangle_material_slots.assign(
            unit_geometry->triangles.size(),
            MaterialSlot::bound(0));
        unit_sphere_asset_index =
            static_cast<int>(result.assets.size());
        result.assets.push_back(std::move(sphere_asset));
    }

    for (const SceneObject& object : state_.objects) {
        if (!is_effectively_visible(object.id)) {
            continue;
        }
        const Mat4 world = world_matrix(object.id);
        if (object.type == SceneObjectType::PointLight) {
            result.point_lights.push_back(
                PointLight{
                    transform_point(world, Vec3::Zero()),
                    object.light_color,
                    object.light_range});
            continue;
        }
        if (object.type == SceneObjectType::DirectionalLight) {
            const Vec3 direction =
                (world.topLeftCorner<3, 3>() *
                 Vec3(0.0f, 0.0f, -1.0f)).normalized();
            result.directional_lights.push_back(
                DirectionalLight{direction, object.light_color});
            continue;
        }
        if (object.type == SceneObjectType::SpotLight) {
            const Vec3 direction =
                (world.topLeftCorner<3, 3>() * Vec3(0.0f, 0.0f, -1.0f)).normalized();
            result.spot_lights.push_back(SpotLight{
                transform_point(world, Vec3::Zero()),
                direction,
                object.light_color,
                object.light_range,
                object.spot_inner_cone_radians,
                object.spot_outer_cone_radians});
            continue;
        }
        if (object.type != SceneObjectType::Mesh) {
            continue;
        }
        const auto asset = find_asset(object.asset_id);
        if (!asset) {
            continue;
        }
        const int asset_index = asset_indices.at(asset->id);
        if (asset_index >= 0) {
            RenderSceneInstanceSnapshot instance;
            instance.object_id = object.id;
            instance.asset_index = asset_index;
            instance.object_to_world = world;
            instance.world_to_object = world.inverse();
            instance.normal_to_world =
                world.topLeftCorner<3, 3>().inverse().transpose();
            instance.world_bounds = transform_bounds(
                result.assets[static_cast<std::size_t>(asset_index)]
                    .local_bounds,
                world);
            instance.materials = asset->local_scene.materials;
            for (Material& material : instance.materials) {
                remap_material_textures(
                    material,
                    texture_remaps.at(asset->id));
            }
            for (const SceneMaterialOverride& material_override :
                 object.material_overrides) {
                if (material_override.material_slot >=
                    instance.materials.size()) {
                    continue;
                }
                apply_material_override(
                    instance.materials[material_override.material_slot],
                    material_override);
            }
            result.instances.push_back(std::move(instance));
        }

        for (const Sphere& sphere : asset->procedural_spheres) {
            if (unit_sphere_asset_index < 0) {
                throw std::logic_error(
                    "procedural sphere asset is missing canonical geometry");
            }
            const MaterialSlot material_slot = validated_material_slot(
                sphere.material_id(),
                asset->local_scene.materials.size());
            Material material = diagnostic_material();
            if (material_slot.has_value()) {
                material = asset->local_scene.materials[material_slot.value()];
                remap_material_textures(
                    material,
                    texture_remaps.at(asset->id));
                const auto material_override = std::find_if(
                    object.material_overrides.begin(),
                    object.material_overrides.end(),
                    [&material_slot](const SceneMaterialOverride& candidate) {
                        return candidate.material_slot == material_slot.value();
                    });
                if (material_override != object.material_overrides.end()) {
                    apply_material_override(material, *material_override);
                }
            }

            Mat4 local_sphere = Mat4::Identity();
            local_sphere.topLeftCorner<3, 3>() *= sphere.radius();
            local_sphere.topRightCorner<3, 1>() = sphere.center();
            RenderSceneInstanceSnapshot instance;
            instance.object_id = object.id;
            instance.asset_index = unit_sphere_asset_index;
            instance.object_to_world = world * local_sphere;
            instance.world_to_object = instance.object_to_world.inverse();
            instance.normal_to_world = instance.object_to_world
                .topLeftCorner<3, 3>()
                .inverse()
                .transpose();
            instance.world_bounds = transform_bounds(
                result.assets[
                    static_cast<std::size_t>(unit_sphere_asset_index)]
                    .local_bounds,
                instance.object_to_world);
            instance.materials.push_back(std::move(material));
            result.instances.push_back(std::move(instance));
        }
    }

    render_scene_snapshot_ = std::move(result);
    snapshot_dirty_ = false;
}

const RenderSceneSnapshot& SceneDocument::render_scene_snapshot() const {
    ensure_render_scene_snapshot();
    return render_scene_snapshot_;
}

const SceneRevisions& SceneDocument::revisions() const {
    return revisions_;
}

const Color& SceneDocument::environment() const {
    return state_.environment;
}

void SceneDocument::set_environment(Color color) {
    if (!color.allFinite()) {
        return;
    }
    color = color.cwiseMax(Color::Zero());
    if (state_.environment.isApprox(color, 0.0f)) {
        return;
    }
    state_.environment = color;
    mark_changed(SceneRevisionDomain::Environment);
}

void SceneDocument::set_environment_map(const std::filesystem::path& path) {
    const auto loaded = EnvironmentMap::load(path);
    if (!state_.environment_map) {
        state_.environment = Color::Ones();
    }
    state_.environment_map = loaded;
    state_.environment_path = loaded->source_path();
    mark_changed(
        SceneRevisionDomain::Environment | SceneRevisionDomain::Textures);
}

void SceneDocument::clear_environment_map() {
    state_.environment_map.reset();
    state_.environment_path.clear();
    mark_changed(
        SceneRevisionDomain::Environment | SceneRevisionDomain::Textures);
}

const std::shared_ptr<const EnvironmentMap>& SceneDocument::environment_map() const {
    return state_.environment_map;
}

const std::filesystem::path& SceneDocument::environment_path() const {
    return state_.environment_path;
}

float SceneDocument::environment_intensity() const {
    return state_.environment_intensity;
}

void SceneDocument::set_environment_intensity(float intensity) {
    if (!std::isfinite(intensity)) {
        return;
    }
    intensity = std::max(0.0f, intensity);
    if (state_.environment_intensity == intensity) {
        return;
    }
    state_.environment_intensity = intensity;
    mark_changed(SceneRevisionDomain::Environment);
}

float SceneDocument::environment_rotation_degrees() const {
    return state_.environment_rotation_degrees;
}

void SceneDocument::set_environment_rotation_degrees(float rotation_degrees) {
    if (!std::isfinite(rotation_degrees) ||
        state_.environment_rotation_degrees == rotation_degrees) {
        return;
    }
    state_.environment_rotation_degrees = rotation_degrees;
    mark_changed(SceneRevisionDomain::Environment);
}

bool SceneDocument::environment_background_visible() const {
    return state_.environment_background_visible;
}

void SceneDocument::set_environment_background_visible(bool visible) {
    if (state_.environment_background_visible == visible) {
        return;
    }
    state_.environment_background_visible = visible;
    mark_changed(SceneRevisionDomain::Environment);
}

struct SceneEditTransaction::Backup {
    SceneDocument::State state;
    std::vector<std::shared_ptr<SceneMeshAsset>> assets;
    ObjectId next_object_id = 1;
    AssetId next_asset_id = 1;
    std::vector<SceneDocument::State> history;
    std::size_t history_cursor = 0;
    std::optional<std::size_t> saved_cursor;
    bool uncheckpointed_changes = false;
    std::filesystem::path file_path;
    std::vector<std::string> warnings;
    std::string last_checkpoint_merge_key;
};

SceneEditTransaction::SceneEditTransaction(
    SceneDocument& document,
    std::string merge_key)
    : document_(&document),
      backup_(std::make_unique<Backup>()),
      merge_key_(std::move(merge_key)) {
    if (document.edit_transaction_active_) {
        throw std::logic_error("nested scene edit transactions are not supported");
    }
    backup_->state = document.state_;
    backup_->assets = document.assets_;
    backup_->next_object_id = document.next_object_id_;
    backup_->next_asset_id = document.next_asset_id_;
    backup_->history = document.history_;
    backup_->history_cursor = document.history_cursor_;
    backup_->saved_cursor = document.saved_cursor_;
    backup_->uncheckpointed_changes = document.uncheckpointed_changes_;
    backup_->file_path = document.file_path_;
    backup_->warnings = document.warnings_;
    backup_->last_checkpoint_merge_key =
        document.last_checkpoint_merge_key_;
    document.edit_transaction_active_ = true;
}

SceneEditTransaction::SceneEditTransaction(
    SceneEditTransaction&& other) noexcept
    : document_(std::exchange(other.document_, nullptr)),
      backup_(std::move(other.backup_)),
      merge_key_(std::move(other.merge_key_)) {}

SceneEditTransaction& SceneEditTransaction::operator=(
    SceneEditTransaction&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    cancel();
    document_ = std::exchange(other.document_, nullptr);
    backup_ = std::move(other.backup_);
    merge_key_ = std::move(other.merge_key_);
    return *this;
}

SceneEditTransaction::~SceneEditTransaction() {
    cancel();
}

bool SceneEditTransaction::active() const {
    return document_ != nullptr;
}

void SceneEditTransaction::commit() {
    if (!document_) {
        return;
    }
    SceneDocument& document = *document_;
    document.edit_transaction_active_ = false;
    const bool changed = document.uncheckpointed_changes_;
    const bool can_merge =
        changed &&
        !merge_key_.empty() &&
        merge_key_ == document.last_checkpoint_merge_key_ &&
        document.history_cursor_ > 0 &&
        document.history_cursor_ + 1 == document.history_.size();
    if (can_merge) {
        if (document.saved_cursor_ &&
            *document.saved_cursor_ == document.history_cursor_) {
            document.saved_cursor_.reset();
        }
        document.history_[document.history_cursor_] = document.state_;
        document.uncheckpointed_changes_ = false;
    } else {
        document.checkpoint();
    }
    if (changed) {
        document.last_checkpoint_merge_key_ = merge_key_;
    }
    document_ = nullptr;
    backup_.reset();
}

void SceneEditTransaction::cancel() {
    if (!document_) {
        return;
    }
    SceneDocument& document = *document_;
    document.state_ = std::move(backup_->state);
    document.assets_ = std::move(backup_->assets);
    document.next_object_id_ = backup_->next_object_id;
    document.next_asset_id_ = backup_->next_asset_id;
    document.history_ = std::move(backup_->history);
    document.history_cursor_ = backup_->history_cursor;
    document.saved_cursor_ = backup_->saved_cursor;
    document.uncheckpointed_changes_ =
        backup_->uncheckpointed_changes;
    document.file_path_ = std::move(backup_->file_path);
    document.warnings_ = std::move(backup_->warnings);
    document.last_checkpoint_merge_key_ =
        std::move(backup_->last_checkpoint_merge_key);
    document.edit_transaction_active_ = false;
    document.object_index_dirty_ = true;
    document.spatial_cache_dirty_ = true;
    document.snapshot_dirty_ = true;
    advance_scene_revisions(
        document.revisions_,
        SceneRevisionDomain::All);
    document_ = nullptr;
    backup_.reset();
}

SceneEditTransaction SceneDocument::begin_edit(
    std::string merge_key) {
    return SceneEditTransaction(*this, std::move(merge_key));
}

void SceneDocument::checkpoint() {
    if (edit_transaction_active_) {
        return;
    }
    last_checkpoint_merge_key_.clear();
    if (history_.empty()) {
        history_.push_back(state_);
        history_cursor_ = 0;
        uncheckpointed_changes_ = false;
        return;
    }
    if (!uncheckpointed_changes_) {
        return;
    }
    if (history_cursor_ + 1 < history_.size()) {
        if (saved_cursor_ && *saved_cursor_ > history_cursor_) {
            saved_cursor_.reset();
        }
        history_.erase(history_.begin() + static_cast<std::ptrdiff_t>(history_cursor_ + 1), history_.end());
    }
    history_.push_back(state_);
    history_cursor_ = history_.size() - 1;
    uncheckpointed_changes_ = false;
    if (history_.size() > kMaximumHistory) {
        history_.erase(history_.begin());
        --history_cursor_;
        if (saved_cursor_) {
            if (*saved_cursor_ == 0) {
                saved_cursor_.reset();
            } else {
                --*saved_cursor_;
            }
        }
    }
}

bool SceneDocument::undo() {
    if (!can_undo()) {
        return false;
    }
    state_ = history_[--history_cursor_];
    object_index_dirty_ = true;
    spatial_cache_dirty_ = true;
    uncheckpointed_changes_ = false;
    advance_scene_revisions(revisions_, SceneRevisionDomain::All);
    snapshot_dirty_ = true;
    return true;
}

bool SceneDocument::redo() {
    if (!can_redo()) {
        return false;
    }
    state_ = history_[++history_cursor_];
    object_index_dirty_ = true;
    spatial_cache_dirty_ = true;
    uncheckpointed_changes_ = false;
    advance_scene_revisions(revisions_, SceneRevisionDomain::All);
    snapshot_dirty_ = true;
    return true;
}

bool SceneDocument::can_undo() const {
    return history_cursor_ > 0;
}

bool SceneDocument::can_redo() const {
    return history_cursor_ + 1 < history_.size();
}

bool SceneDocument::dirty() const {
    return uncheckpointed_changes_ || !saved_cursor_ ||
        history_cursor_ != *saved_cursor_;
}

void SceneDocument::mark_saved() {
    checkpoint();
    saved_cursor_ = history_cursor_;
}

const std::filesystem::path& SceneDocument::file_path() const {
    return file_path_;
}

const std::vector<std::string>& SceneDocument::warnings() const {
    return warnings_;
}

nlohmann::json SceneDocument::serialize_document(
    const std::filesystem::path& base,
    bool session_snapshot) const {
    std::unordered_set<AssetId> referenced_asset_ids;
    for (const SceneObject& object : state_.objects) {
        if (object.type != SceneObjectType::Mesh) {
            continue;
        }
        referenced_asset_ids.insert(object.asset_id);
        const auto asset = find_asset(object.asset_id);
        if (asset && asset->source_path.empty() &&
            (!session_snapshot || asset->builtin_id.empty())) {
            throw std::runtime_error(
                session_snapshot
                    ? "embedded geometry has no session builtin id"
                    : "embedded builtin geometry cannot be saved as an .rscene asset");
        }
    }
    nlohmann::json root;
    root["version"] = 4;
    nlohmann::json environment{
        {"color", vec3_json(state_.environment)},
        {"intensity", state_.environment_intensity},
        {"rotation_degrees", state_.environment_rotation_degrees},
        {"background_visible", state_.environment_background_visible},
    };
    if (!state_.environment_path.empty()) {
        std::filesystem::path stored = state_.environment_path;
        if (session_snapshot) {
            stored = normalized_absolute(stored);
        } else {
            std::error_code error;
            const std::filesystem::path relative =
                std::filesystem::relative(stored, base, error);
            if (!error) {
                stored = relative;
            }
        }
        environment["path"] = stored.generic_string();
    }
    root["environment"] = std::move(environment);
    root["assets"] = nlohmann::json::array();
    for (const auto& asset : assets_) {
        if (!referenced_asset_ids.contains(asset->id)) {
            continue;
        }
        if (session_snapshot) {
            nlohmann::json source;
            if (!asset->source_path.empty()) {
                source = {
                    {"kind", asset->source_mesh_index >= 0 ? "gltf" : "obj"},
                    {"path", normalized_absolute(asset->source_path).generic_string()},
                };
                if (asset->source_mesh_index >= 0) {
                    source["mesh_index"] = asset->source_mesh_index;
                }
            } else if (!asset->builtin_id.empty()) {
                source = {
                    {"kind", "builtin"},
                    {"id", asset->builtin_id},
                };
            } else {
                continue;
            }
            root["assets"].push_back({
                {"id", asset->id},
                {"source", std::move(source)},
            });
        } else {
            if (asset->source_path.empty()) {
                continue;
            }
            std::error_code error;
            std::filesystem::path stored =
                std::filesystem::relative(asset->source_path, base, error);
            if (error) {
                stored = asset->source_path;
            }
            root["assets"].push_back({
                {"id", asset->id},
                {"path", stored.generic_string()},
                {"kind", asset->source_mesh_index >= 0 ? "gltf" : "obj"},
                {"mesh_index", asset->source_mesh_index},
            });
        }
    }
    root["objects"] = nlohmann::json::array();
    for (const SceneObject& object : state_.objects) {
        nlohmann::json object_entry{
            {"id", object.id},
            {"parent", object.parent_id},
            {"name", object.name},
            {"type", object_type_name(object.type)},
            {"local_matrix", matrix_json(object.transform.matrix())},
            {"visible", object.visible},
            {"locked", object.locked},
            {"asset", object.asset_id},
            {"light_color", vec3_json(object.light_color)},
            {"light_range", object.light_range},
            {"spot_inner_cone_radians", object.spot_inner_cone_radians},
            {"spot_outer_cone_radians", object.spot_outer_cone_radians},
            {"camera_projection", camera_projection_name(object.camera_projection)},
            {"camera_vertical_fov_degrees", object.camera_vertical_fov_degrees},
            {"camera_aspect_ratio", object.camera_aspect_ratio},
            {"camera_x_magnification", object.camera_x_magnification},
            {"camera_y_magnification", object.camera_y_magnification},
            {"camera_near_plane", object.camera_near_plane},
            {"camera_far_plane", object.camera_far_plane},
        };
        if (!object.material_overrides.empty()) {
            object_entry["material_overrides"] = nlohmann::json::array();
            const auto asset = find_asset(object.asset_id);
            for (const SceneMaterialOverride& material_override_value :
                 object.material_overrides) {
                const std::size_t slot = material_override_value.material_slot;
                const std::string material_name =
                    asset && slot < asset->material_names.size()
                    ? asset->material_names[slot]
                    : std::string();
                object_entry["material_overrides"].push_back({
                    {"slot", material_override_value.material_slot},
                    {"name", material_name},
                    {"type", material_type_name(material_override_value.type)},
                    {"base_color", vec3_json(material_override_value.base_color)},
                    {"emission", vec3_json(material_override_value.emission)},
                    {"roughness", material_override_value.roughness},
                    {"metallic", material_override_value.metallic},
                    {"ior", material_override_value.ior},
                    {"opacity", material_override_value.opacity},
                    {"alpha_cutoff", material_override_value.alpha_cutoff},
                    {"bump_scale", material_override_value.bump_scale},
                    {"normal_scale", material_override_value.normal_scale},
                    {"occlusion_strength", material_override_value.occlusion_strength},
                    {"alpha_mode", alpha_mode_name(material_override_value.alpha_mode)},
                    {"two_sided", material_override_value.two_sided},
                    {"use_diffuse_texture", material_override_value.use_diffuse_texture},
                    {"use_opacity_texture", material_override_value.use_opacity_texture},
                    {"use_bump_texture", material_override_value.use_bump_texture},
                    {"use_base_color_texture", material_override_value.use_base_color_texture},
                    {"use_metallic_roughness_texture", material_override_value.use_metallic_roughness_texture},
                    {"use_normal_texture", material_override_value.use_normal_texture},
                    {"use_occlusion_texture", material_override_value.use_occlusion_texture},
                    {"use_emissive_texture", material_override_value.use_emissive_texture},
                });
            }
        }
        root["objects"].push_back(std::move(object_entry));
    }
    return root;
}

void SceneDocument::save(const std::filesystem::path& path) {
    const std::filesystem::path absolute_path = normalized_absolute(path);
    const nlohmann::json root =
        serialize_document(absolute_path.parent_path(), false);
    write_file_atomically(absolute_path, root.dump(2) + '\n');
    file_path_ = absolute_path;
    mark_saved();
}

nlohmann::json SceneDocument::session_snapshot() const {
    return serialize_document({}, true);
}

void SceneDocument::restore_file_state(
    const std::filesystem::path& path,
    bool dirty_value) {
    file_path_ = path.empty()
        ? std::filesystem::path()
        : normalized_absolute(path);
    if (dirty_value) {
        saved_cursor_.reset();
    } else {
        checkpoint();
        saved_cursor_ = history_cursor_;
    }
}

SceneDocument SceneDocument::deserialize_document(
    const nlohmann::json& root,
    const std::filesystem::path& document_path,
    int width,
    int height,
    bool session_snapshot) {
    const int version = root.value("version", 0);
    if (version < 1 || version > 4) {
        throw std::runtime_error("unsupported scene file version");
    }

    SceneDocument document;
    document.file_path_ = session_snapshot
        ? std::filesystem::path()
        : document_path;
    document.state_.objects.clear();
    document.assets_.clear();
    if (version >= 3) {
        const auto& environment = root.at("environment");
        document.state_.environment = parse_vec3(environment.at("color"), "environment.color");
        document.state_.environment_intensity = environment.value("intensity", 1.0f);
        document.state_.environment_rotation_degrees = environment.value("rotation_degrees", 0.0f);
        document.state_.environment_background_visible =
            environment.value("background_visible", true);
        if (!std::isfinite(document.state_.environment_intensity) ||
            document.state_.environment_intensity < 0.0f ||
            !std::isfinite(document.state_.environment_rotation_degrees)) {
            throw std::runtime_error("scene environment settings are invalid");
        }
        if (environment.contains("path")) {
            std::filesystem::path environment_path =
                environment.at("path").get<std::string>();
            if (!session_snapshot && environment_path.is_relative()) {
                environment_path = document_path.parent_path() / environment_path;
            }
            document.state_.environment_path = normalized_absolute(environment_path);
            if (std::filesystem::exists(document.state_.environment_path)) {
                try {
                    document.state_.environment_map =
                        EnvironmentMap::load(document.state_.environment_path);
                } catch (const std::exception& error) {
                    document.warnings_.push_back(
                        "failed to restore environment '" +
                        document.state_.environment_path.string() + "': " + error.what());
                }
            } else {
                document.warnings_.push_back(
                    "missing environment: " + document.state_.environment_path.string());
            }
        }
    } else {
        document.state_.environment = parse_vec3(root.at("environment"), "environment");
    }
    std::unordered_set<AssetId> referenced_asset_ids;
    for (const auto& object_json : root.at("objects")) {
        if (object_json.at("type").get<std::string>() != "mesh") {
            continue;
        }
        const AssetId stored_asset =
            object_json.value("asset", kInvalidAssetId);
        if (stored_asset != kInvalidAssetId) {
            referenced_asset_ids.insert(stored_asset);
        }
    }
    std::unordered_map<AssetId, AssetId> asset_ids;
    std::unordered_set<AssetId> stored_asset_ids;
    std::unordered_map<std::wstring, LoadedGltfScene> restored_gltf_sources;
    const auto load_restored_asset =
        [&document, &restored_gltf_sources, width, height](
            const std::filesystem::path& source_path,
            int source_mesh_index) -> std::shared_ptr<SceneMeshAsset> {
            const std::filesystem::path normalized = normalized_absolute(source_path);
            const std::string extension = lowercase_extension(normalized);
            if (extension != ".gltf" && extension != ".glb") {
                return document.load_asset(normalized, width, height, -1);
            }
            const int mesh_index = source_mesh_index >= 0 ? source_mesh_index : 0;
            for (const auto& existing : document.assets_) {
                if (!existing->source_path.empty() &&
                    normalized_absolute(existing->source_path) == normalized &&
                    existing->source_mesh_index == mesh_index) {
                    return existing;
                }
            }
            const std::wstring cache_key = normalized.generic_wstring();
            auto found = restored_gltf_sources.find(cache_key);
            if (found == restored_gltf_sources.end()) {
                LoadedGltfScene loaded = load_gltf_scene(normalized, width, height);
                for (const std::string& warning : loaded.warnings) {
                    document.warnings_.push_back(normalized.string() + ": " + warning);
                }
                found = restored_gltf_sources.emplace(
                    cache_key,
                    std::move(loaded)).first;
            }
            if (mesh_index < 0 ||
                static_cast<std::size_t>(mesh_index) >= found->second.meshes.size()) {
                throw std::runtime_error("glTF source mesh index is out of range");
            }
            return document.store_loaded_asset(
                std::move(found->second.meshes[static_cast<std::size_t>(mesh_index)]),
                normalized,
                mesh_index);
        };
    for (const auto& asset_json : root.at("assets")) {
        const AssetId stored_id = asset_json.at("id").get<AssetId>();
        if (stored_id == kInvalidAssetId ||
            !stored_asset_ids.insert(stored_id).second) {
            throw std::runtime_error("scene asset ids must be unique and nonzero");
        }
        if (!referenced_asset_ids.contains(stored_id)) {
            continue;
        }
        if (session_snapshot) {
            const auto& source = asset_json.at("source");
            const std::string kind = source.at("kind").get<std::string>();
            if (kind == "builtin") {
                const std::string builtin_id = source.at("id").get<std::string>();
                if (builtin_id != "cornell_box") {
                    document.warnings_.push_back(
                        "unknown session builtin asset: " + builtin_id);
                    continue;
                }
                Scene builtin_scene = make_cornell_box_scene();
                auto asset = std::make_shared<SceneMeshAsset>();
                asset->id = document.next_asset_id_++;
                asset->builtin_id = builtin_id;
                asset->local_scene = std::move(builtin_scene);
                asset->material_names.reserve(asset->local_scene.materials.size());
                for (std::size_t index = 0;
                     index < asset->local_scene.materials.size();
                     ++index) {
                    asset->material_names.push_back(
                        "Material " + std::to_string(index + 1));
                }
                for (const Triangle& triangle :
                     asset->local_scene.triangles) {
                    asset->local_bounds.expand(triangle.bounds());
                }
                for (const Sphere& sphere : asset->local_scene.spheres) {
                    asset->local_bounds.expand(sphere.bounds());
                }
                build_picking_acceleration(asset);
                document.assets_.push_back(asset);
                asset_ids[stored_id] = asset->id;
                continue;
            }
            if (kind != "obj" && kind != "gltf") {
                document.warnings_.push_back(
                    "unknown session asset source kind: " + kind);
                continue;
            }
            const std::filesystem::path asset_path =
                source.at("path").get<std::string>();
            if (!std::filesystem::exists(asset_path)) {
                document.warnings_.push_back(
                    "missing session asset: " + asset_path.string());
                continue;
            }
            try {
                const int source_mesh_index = kind == "gltf"
                    ? source.value("mesh_index", 0)
                    : -1;
                const auto asset = load_restored_asset(
                    asset_path,
                    source_mesh_index);
                asset_ids[stored_id] = asset->id;
            } catch (const std::exception& error) {
                document.warnings_.push_back(
                    "failed to restore session asset '" + asset_path.string() +
                    "': " + error.what());
            }
            continue;
        }

        std::filesystem::path asset_path = asset_json.at("path").get<std::string>();
        if (asset_path.is_relative()) {
            asset_path = document_path.parent_path() / asset_path;
        }
        if (!std::filesystem::exists(asset_path)) {
            document.warnings_.push_back("missing asset: " + asset_path.string());
            auto missing_asset = std::make_shared<SceneMeshAsset>();
            missing_asset->id = document.next_asset_id_++;
            missing_asset->source_path = normalized_absolute(asset_path);
            document.assets_.push_back(missing_asset);
            asset_ids[stored_id] = missing_asset->id;
            continue;
        }
        const int source_mesh_index = version >= 3
            ? asset_json.value("mesh_index", -1)
            : -1;
        const auto asset = load_restored_asset(
            asset_path,
            source_mesh_index);
        asset_ids[stored_id] = asset->id;
    }
    std::unordered_set<ObjectId> object_ids;
    for (const auto& object_json : root.at("objects")) {
        SceneObject object;
        object.id = object_json.at("id").get<ObjectId>();
        if (object.id == kInvalidObjectId || !object_ids.insert(object.id).second) {
            throw std::runtime_error("scene object ids must be unique and nonzero");
        }
        object.parent_id = object_json.value("parent", kInvalidObjectId);
        object.name = object_json.at("name").get<std::string>();
        object.type = parse_object_type(object_json.at("type").get<std::string>());
        if (version >= 4) {
            object.transform.local_matrix = parse_matrix(
                object_json.at("local_matrix"),
                "local_matrix");
        } else {
            SceneTrs legacy_transform;
            legacy_transform.translation =
                parse_vec3(object_json.at("translation"), "translation");
            legacy_transform.rotation_degrees =
                parse_vec3(object_json.at("rotation_degrees"), "rotation_degrees");
            legacy_transform.scale =
                parse_vec3(object_json.at("scale"), "scale");
            object.transform = SceneTransform::from_trs(legacy_transform);
        }
        if (!object.transform.valid()) {
            throw std::runtime_error("scene object has a singular or invalid transform");
        }
        object.visible = object_json.value("visible", true);
        object.locked = object_json.value("locked", false);
        const AssetId stored_asset = object_json.value("asset", kInvalidAssetId);
        const auto mapped = asset_ids.find(stored_asset);
        if (session_snapshot &&
            object.type == SceneObjectType::Mesh &&
            mapped == asset_ids.end()) {
            document.warnings_.push_back(
                "skipped object '" + object.name +
                "' because its session asset is unavailable");
            continue;
        }
        object.asset_id = mapped == asset_ids.end() ? kInvalidAssetId : mapped->second;
        object.light_color = parse_vec3(object_json.at("light_color"), "light_color");
        if (version >= 3) {
            object.light_range = object_json.value("light_range", 0.0f);
            object.spot_inner_cone_radians =
                object_json.value("spot_inner_cone_radians", 0.0f);
            object.spot_outer_cone_radians =
                object_json.value("spot_outer_cone_radians", kPi * 0.25f);
            object.camera_projection = parse_camera_projection(
                object_json.value("camera_projection", std::string("perspective")));
            object.camera_vertical_fov_degrees =
                object_json.value("camera_vertical_fov_degrees", 45.0f);
            object.camera_aspect_ratio = object_json.value("camera_aspect_ratio", 0.0f);
            object.camera_x_magnification = object_json.value("camera_x_magnification", 1.0f);
            object.camera_y_magnification = object_json.value("camera_y_magnification", 1.0f);
            object.camera_near_plane = object_json.value("camera_near_plane", 0.01f);
            object.camera_far_plane = object_json.value("camera_far_plane", 1000.0f);
        }
        if (version >= 2 && object_json.contains("material_overrides")) {
            std::unordered_set<std::uint32_t> material_slots;
            for (const auto& override_json :
                 object_json.at("material_overrides")) {
                SceneMaterialOverride material_override_value;
                material_override_value.material_slot =
                    override_json.at("slot").get<std::uint32_t>();
                if (!material_slots.insert(
                        material_override_value.material_slot).second) {
                    throw std::runtime_error(
                        "scene object has duplicate material override slots");
                }
                material_override_value.type = parse_material_type(
                    override_json.at("type").get<std::string>());
                material_override_value.base_color =
                    parse_vec3(override_json.at("base_color"), "base_color");
                material_override_value.emission =
                    parse_vec3(override_json.at("emission"), "emission");
                material_override_value.roughness =
                    override_json.at("roughness").get<float>();
                material_override_value.metallic =
                    override_json.value("metallic", 0.0f);
                material_override_value.ior =
                    override_json.at("ior").get<float>();
                material_override_value.opacity =
                    override_json.at("opacity").get<float>();
                material_override_value.alpha_cutoff =
                    override_json.at("alpha_cutoff").get<float>();
                material_override_value.bump_scale =
                    override_json.at("bump_scale").get<float>();
                material_override_value.normal_scale =
                    override_json.value("normal_scale", 1.0f);
                material_override_value.occlusion_strength =
                    override_json.value("occlusion_strength", 1.0f);
                material_override_value.alpha_mode = parse_alpha_mode(
                    override_json.value("alpha_mode", std::string("opaque")));
                material_override_value.two_sided =
                    override_json.value("two_sided", true);
                material_override_value.use_diffuse_texture =
                    override_json.value("use_diffuse_texture", true);
                material_override_value.use_opacity_texture =
                    override_json.value("use_opacity_texture", true);
                material_override_value.use_bump_texture =
                    override_json.value("use_bump_texture", true);
                material_override_value.use_base_color_texture =
                    override_json.value("use_base_color_texture", true);
                material_override_value.use_metallic_roughness_texture =
                    override_json.value("use_metallic_roughness_texture", true);
                material_override_value.use_normal_texture =
                    override_json.value("use_normal_texture", true);
                material_override_value.use_occlusion_texture =
                    override_json.value("use_occlusion_texture", true);
                material_override_value.use_emissive_texture =
                    override_json.value("use_emissive_texture", true);
                if (!valid_material_override(material_override_value)) {
                    throw std::runtime_error(
                        "scene object has an invalid material override");
                }
                object.material_overrides.push_back(material_override_value);
            }
            std::sort(
                object.material_overrides.begin(),
                object.material_overrides.end(),
                [](const SceneMaterialOverride& lhs, const SceneMaterialOverride& rhs) {
                    return lhs.material_slot < rhs.material_slot;
                });
        }
        document.next_object_id_ = std::max(document.next_object_id_, object.id + 1);
        document.state_.objects.push_back(std::move(object));
    }
    if (session_snapshot) {
        bool removed_object = true;
        while (removed_object) {
            removed_object = false;
            std::unordered_set<ObjectId> surviving_ids;
            for (const SceneObject& object : document.state_.objects) {
                surviving_ids.insert(object.id);
            }
            const auto removed_begin = std::remove_if(
                document.state_.objects.begin(),
                document.state_.objects.end(),
                [&](const SceneObject& object) {
                    if (object.parent_id == kInvalidObjectId ||
                        surviving_ids.contains(object.parent_id)) {
                        return false;
                    }
                    document.warnings_.push_back(
                        "skipped object '" + object.name +
                        "' because its session parent is unavailable");
                    removed_object = true;
                    return true;
                });
            document.state_.objects.erase(
                removed_begin,
                document.state_.objects.end());
        }
    }
    for (const SceneObject& object : document.state_.objects) {
        if (object.parent_id != kInvalidObjectId && !document.find(object.parent_id)) {
            throw std::runtime_error("scene object references a missing parent");
        }
        if (document.is_descendant(object.parent_id, object.id)) {
            throw std::runtime_error("scene hierarchy contains a cycle");
        }
        const auto asset = document.find_asset(object.asset_id);
        if (asset && !asset->local_scene.materials.empty()) {
            for (const SceneMaterialOverride& material_override_value :
                 object.material_overrides) {
                if (material_override_value.material_slot >=
                    asset->local_scene.materials.size()) {
                    document.warnings_.push_back(
                        "object '" + object.name +
                        "' has an out-of-range material override slot " +
                        std::to_string(material_override_value.material_slot));
                }
            }
        }
    }
    document.history_.assign(1, document.state_);
    document.history_cursor_ = 0;
    document.saved_cursor_ = 0;
    document.uncheckpointed_changes_ = false;
    document.object_index_dirty_ = true;
    document.spatial_cache_dirty_ = true;
    advance_scene_revisions(
        document.revisions_,
        SceneRevisionDomain::All);
    document.snapshot_dirty_ = true;
    return document;
}

SceneDocument SceneDocument::load(
    const std::filesystem::path& path,
    int width,
    int height) {
    const std::filesystem::path absolute_path = normalized_absolute(path);
    std::ifstream input(absolute_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "failed to open scene file: " + absolute_path.string());
    }
    nlohmann::json root;
    input >> root;
    return deserialize_document(
        root,
        absolute_path,
        width,
        height,
        false);
}

SceneDocument SceneDocument::from_session_snapshot(
    const nlohmann::json& snapshot,
    int width,
    int height) {
    return deserialize_document(
        snapshot,
        {},
        width,
        height,
        true);
}

}  // namespace renderer
