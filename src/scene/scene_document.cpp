#include "scene/scene_document.h"

#include "scene/scene_asset_loader.h"

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace renderer {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kMinimumScale = 1.0e-6f;
constexpr int kMaximumHierarchyDepth = 1024;
constexpr std::size_t kMaximumHistory = 256;

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
    throw std::runtime_error("unknown material type: " + value);
}

bool valid_material_type(MaterialType type) {
    switch (type) {
        case MaterialType::Diffuse:
        case MaterialType::Metal:
        case MaterialType::Dielectric:
        case MaterialType::Emissive:
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
        std::isfinite(material_override.ior) &&
        material_override.ior > 0.0f &&
        std::isfinite(material_override.opacity) &&
        material_override.opacity >= 0.0f &&
        material_override.opacity <= 1.0f &&
        std::isfinite(material_override.alpha_cutoff) &&
        material_override.alpha_cutoff >= 0.0f &&
        material_override.alpha_cutoff <= 1.0f &&
        std::isfinite(material_override.bump_scale);
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
    result.ior = material.ior;
    result.opacity = material.opacity;
    result.alpha_cutoff = material.alpha_cutoff;
    result.bump_scale = material.bump_scale;
    result.two_sided = material.two_sided;
    result.use_diffuse_texture = material.diffuse_texture_id >= 0;
    result.use_opacity_texture = material.opacity_texture_id >= 0;
    result.use_bump_texture = material.bump_texture_id >= 0;
    return result;
}

void apply_material_override(
    Material& material,
    const SceneMaterialOverride& material_override) {
    material.type = material_override.type;
    material.base_color = material_override.base_color;
    material.emission = material_override.emission;
    material.roughness = material_override.roughness;
    material.ior = material_override.ior;
    material.opacity = material_override.opacity;
    material.alpha_cutoff = material_override.alpha_cutoff;
    material.bump_scale = material_override.bump_scale;
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

bool obj_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".obj";
}

}  // namespace

Mat4 SceneTransform::matrix() const {
    const Vec3 radians = rotation_degrees * (kPi / 180.0f);
    const Eigen::Affine3f transform =
        Eigen::Translation3f(translation) *
        Eigen::AngleAxisf(radians.z(), Vec3::UnitZ()) *
        Eigen::AngleAxisf(radians.y(), Vec3::UnitY()) *
        Eigen::AngleAxisf(radians.x(), Vec3::UnitX()) *
        Eigen::Scaling(scale);
    return transform.matrix();
}

bool SceneTransform::valid() const {
    return translation.allFinite() && rotation_degrees.allFinite() && scale.allFinite() &&
        std::abs(scale.x()) >= kMinimumScale &&
        std::abs(scale.y()) >= kMinimumScale &&
        std::abs(scale.z()) >= kMinimumScale;
}

SceneDocument::SceneDocument() {
    history_.push_back(state_);
}

SceneDocument SceneDocument::from_scene(
    Scene scene,
    std::string name,
    std::string builtin_id) {
    SceneDocument document;
    document.state_.environment = scene.environment;
    std::vector<PointLight> point_lights =
        std::move(scene.point_lights);
    std::vector<DirectionalLight> directional_lights =
        std::move(scene.directional_lights);
    scene.point_lights.clear();
    scene.directional_lights.clear();
    auto asset = std::make_shared<SceneMeshAsset>();
    asset->id = document.next_asset_id_++;
    asset->source_path.clear();
    asset->builtin_id = std::move(builtin_id);
    asset->local_scene = std::move(scene);
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
        document.create_point_light(
            "Point Light " + std::to_string(index + 1),
            point_lights[index].position,
            point_lights[index].intensity);
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
    document.render_dirty_ = true;
    document.instanced_dirty_ = true;
    return document;
}

const std::vector<SceneObject>& SceneDocument::objects() const {
    return state_.objects;
}

std::vector<SceneObject>& SceneDocument::objects() {
    render_dirty_ = true;
    instanced_dirty_ = true;
    return state_.objects;
}

const std::vector<std::shared_ptr<SceneMeshAsset>>& SceneDocument::assets() const {
    return assets_;
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
    const auto found = std::find_if(
        state_.objects.begin(),
        state_.objects.end(),
        [id](const SceneObject& object) { return object.id == id; });
    return found == state_.objects.end() ? nullptr : &*found;
}

SceneObject* SceneDocument::find(ObjectId id) {
    const auto found = std::find_if(
        state_.objects.begin(),
        state_.objects.end(),
        [id](const SceneObject& object) { return object.id == id; });
    if (found == state_.objects.end()) {
        return nullptr;
    }
    render_dirty_ = true;
    instanced_dirty_ = true;
    return &*found;
}

std::vector<ObjectId> SceneDocument::children(ObjectId parent_id) const {
    std::vector<ObjectId> result;
    for (const SceneObject& object : state_.objects) {
        if (object.parent_id == parent_id) {
            result.push_back(object.id);
        }
    }
    return result;
}

ObjectId SceneDocument::create_group(std::string name, ObjectId parent_id) {
    SceneObject object;
    object.id = next_object_id_++;
    object.parent_id = parent_id;
    object.name = std::move(name);
    object.type = SceneObjectType::Group;
    state_.objects.push_back(std::move(object));
    render_dirty_ = true;
    instanced_dirty_ = true;
    return state_.objects.back().id;
}

ObjectId SceneDocument::create_point_light(
    std::string name,
    const Vec3& position,
    const Color& intensity,
    ObjectId parent_id) {
    const ObjectId id = create_group(std::move(name), parent_id);
    SceneObject* object = find(id);
    object->type = SceneObjectType::PointLight;
    object->transform.translation = position;
    object->light_color = intensity;
    return id;
}

ObjectId SceneDocument::create_directional_light(
    std::string name,
    const Vec3& direction,
    const Color& radiance,
    ObjectId parent_id) {
    const ObjectId id = create_group(std::move(name), parent_id);
    SceneObject* object = find(id);
    object->type = SceneObjectType::DirectionalLight;
    object->light_color = radiance;
    const Vec3 normalized = usable_direction(direction)
        ? direction.normalized()
        : Vec3(0.0f, -1.0f, 0.0f);
    const Eigen::Quaternionf rotation =
        Eigen::Quaternionf::FromTwoVectors(Vec3(0.0f, 0.0f, -1.0f), normalized);
    const Vec3 zyx = rotation.toRotationMatrix().eulerAngles(2, 1, 0);
    object->transform.rotation_degrees =
        Vec3(zyx.z(), zyx.y(), zyx.x()) * (180.0f / kPi);
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
    int height) {
    const std::filesystem::path normalized = normalized_absolute(path);
    for (const auto& asset : assets_) {
        if (!asset->source_path.empty() &&
            normalized_absolute(asset->source_path) == normalized) {
            return asset;
        }
    }

    LoadedScene loaded = load_scene_asset(normalized.string(), width, height);
    auto asset = std::make_shared<SceneMeshAsset>();
    asset->id = next_asset_id_++;
    asset->source_path = normalized;
    asset->local_scene = std::move(loaded.scene);
    asset->local_scene.directional_lights.clear();
    asset->local_scene.point_lights.clear();
    asset->warnings = std::move(loaded.warnings);
    asset->material_names = std::move(loaded.material_names);
    asset->local_bounds = loaded.bounds;
    assets_.push_back(asset);
    for (const std::string& warning : asset->warnings) {
        warnings_.push_back(normalized.string() + ": " + warning);
    }
    return asset;
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
    render_dirty_ = true;
    instanced_dirty_ = true;
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
    try {
    std::vector<ObjectId> imported;
    if (std::filesystem::is_regular_file(normalized)) {
        if (!obj_extension(normalized)) {
            throw std::runtime_error("asset file is not an OBJ: " + normalized.string());
        }
        imported.push_back(import_obj(normalized, kInvalidObjectId, width, height));
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
            } else if (iterator->is_regular_file(error) && obj_extension(iterator->path())) {
                files.push_back(iterator->path());
            }
        }
        std::sort(files.begin(), files.end(), [&normalized](const auto& lhs, const auto& rhs) {
            return lhs.lexically_relative(normalized).generic_string() <
                rhs.lexically_relative(normalized).generic_string();
        });
        if (files.empty()) {
            throw std::runtime_error("asset directory contains no OBJ files: " + normalized.string());
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
            imported.push_back(import_obj(file, parent, width, height));
        }
    } else {
        throw std::runtime_error("asset path is neither a file nor directory: " + normalized.string());
    }

    if (std::none_of(state_.objects.begin(), state_.objects.end(), [](const SceneObject& object) {
            return object.type == SceneObjectType::DirectionalLight;
        })) {
        create_directional_light(
            "Sun",
            Vec3(-0.5f, -1.0f, -0.25f),
            Color(25.0f, 25.0f, 25.0f));
    }
    checkpoint();
    render_dirty_ = true;
    instanced_dirty_ = true;
    return imported;
    } catch (...) {
        state_ = previous_state;
        assets_.resize(previous_asset_count);
        warnings_.resize(previous_warning_count);
        next_object_id_ = previous_next_object_id;
        next_asset_id_ = previous_next_asset_id;
        render_dirty_ = true;
        instanced_dirty_ = true;
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
        render_dirty_ = true;
        instanced_dirty_ = true;
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
    render_dirty_ = true;
    instanced_dirty_ = true;
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
    SceneObject* object = find(id);
    if (!object || id == new_parent_id ||
        (new_parent_id != kInvalidObjectId && !find(new_parent_id)) ||
        is_descendant(new_parent_id, id)) {
        return false;
    }
    const ObjectId old_parent_id = object->parent_id;
    const Mat4 old_world = world_matrix(id);
    object->parent_id = new_parent_id;
    const Mat4 parent_world = new_parent_id == kInvalidObjectId
        ? Mat4::Identity()
        : world_matrix(new_parent_id);
    if (!decompose_matrix(parent_world.inverse() * old_world, object->transform)) {
        object->parent_id = old_parent_id;
        return false;
    }
    render_dirty_ = true;
    instanced_dirty_ = true;
    checkpoint();
    return true;
}

bool SceneDocument::decompose_matrix(const Mat4& matrix, SceneTransform& transform) {
    if (!matrix.allFinite()) {
        return false;
    }
    transform.translation = matrix.topRightCorner<3, 1>();
    Mat3 linear = matrix.topLeftCorner<3, 3>();
    transform.scale = Vec3(
        linear.col(0).norm(),
        linear.col(1).norm(),
        linear.col(2).norm());
    if ((transform.scale.array() < kMinimumScale).any()) {
        return false;
    }
    if (linear.determinant() < 0.0f) {
        transform.scale.x() = -transform.scale.x();
    }
    Mat3 rotation = linear;
    for (int axis = 0; axis < 3; ++axis) {
        rotation.col(axis) /= transform.scale[axis];
    }
    const Vec3 zyx = rotation.eulerAngles(2, 1, 0);
    transform.rotation_degrees =
        Vec3(zyx.z(), zyx.y(), zyx.x()) * (180.0f / kPi);
    return transform.valid();
}

bool SceneDocument::set_world_matrix(ObjectId id, const Mat4& world) {
    SceneObject* object = find(id);
    if (!object || object->locked) {
        return false;
    }
    const Mat4 parent_world = object->parent_id == kInvalidObjectId
        ? Mat4::Identity()
        : world_matrix(object->parent_id);
    if (!decompose_matrix(parent_world.inverse() * world, object->transform)) {
        return false;
    }
    render_dirty_ = true;
    instanced_dirty_ = true;
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
    SceneObject* object = find(id);
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
    render_dirty_ = true;
    instanced_dirty_ = true;
    return true;
}

bool SceneDocument::clear_material_override(
    ObjectId id,
    std::size_t material_slot) {
    SceneObject* object = find(id);
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
    render_dirty_ = true;
    instanced_dirty_ = true;
    return true;
}

Mat4 SceneDocument::world_matrix_recursive(ObjectId id, int depth) const {
    if (depth > kMaximumHierarchyDepth) {
        throw std::runtime_error("scene hierarchy exceeds maximum depth");
    }
    const SceneObject* object = find(id);
    if (!object) {
        return Mat4::Identity();
    }
    const Mat4 local = object->transform.matrix();
    return object->parent_id == kInvalidObjectId
        ? local
        : world_matrix_recursive(object->parent_id, depth + 1) * local;
}

Mat4 SceneDocument::world_matrix(ObjectId id) const {
    return world_matrix_recursive(id, 0);
}

Bounds3 SceneDocument::world_bounds(ObjectId id) const {
    Bounds3 result;
    const SceneObject* object = find(id);
    if (!object) {
        return result;
    }
    if (object->type == SceneObjectType::Mesh) {
        const auto asset = find_asset(object->asset_id);
        if (asset && finite_bounds(asset->local_bounds)) {
            result.expand(transform_bounds(asset->local_bounds, world_matrix(id)));
        }
    }
    for (ObjectId child : children(id)) {
        const Bounds3 child_bounds = world_bounds(child);
        if (finite_bounds(child_bounds)) {
            result.expand(child_bounds);
        }
    }
    return result;
}

Bounds3 SceneDocument::scene_bounds() const {
    ensure_render_scene();
    return render_bounds_;
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
        for (const Triangle& triangle : asset->local_scene.triangles) {
            HitRecord hit;
            if (triangle.intersect(
                    local_ray,
                    0.0f,
                    best ? best->distance : std::numeric_limits<float>::infinity(),
                    hit)) {
                best = ScenePickResult{object.id, hit.t};
            }
        }
    }
    return best;
}

void SceneDocument::ensure_render_scene() const {
    if (render_dirty_) {
        const_cast<SceneDocument*>(this)->rebuild_render_scene();
    }
}

const Scene& SceneDocument::render_scene() const {
    ensure_render_scene();
    return render_scene_;
}

void SceneDocument::ensure_instanced_scene() const {
    if (!instanced_dirty_) {
        return;
    }

    InstancedSceneView result;
    result.environment = state_.environment;
    result.assets.reserve(assets_.size());
    result.instances.reserve(state_.objects.size());
    std::unordered_map<AssetId, int> asset_indices;

    for (const SceneObject& object : state_.objects) {
        if (object.type != SceneObjectType::Mesh) {
            continue;
        }
        const auto asset = find_asset(object.asset_id);
        if (!asset ||
            asset_indices.contains(asset->id)) {
            continue;
        }
        const int asset_index =
            static_cast<int>(result.assets.size());
        asset_indices.emplace(asset->id, asset_index);
        result.assets.push_back(
            InstancedSceneAssetView{
                asset->id,
                &asset->local_scene,
                asset->local_bounds});
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
                    object.light_color});
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
        if (object.type != SceneObjectType::Mesh) {
            continue;
        }
        const auto asset = find_asset(object.asset_id);
        if (!asset) {
            continue;
        }
        const int asset_index =
            asset_indices.at(asset->id);

        InstancedSceneInstanceView instance;
        instance.object_id = object.id;
        instance.asset_index = asset_index;
        instance.object_to_world = world;
        instance.world_to_object = world.inverse();
        instance.normal_to_world =
            world.topLeftCorner<3, 3>().inverse().transpose();
        instance.world_bounds =
            transform_bounds(asset->local_bounds, world);
        instance.materials = asset->local_scene.materials;
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

    instanced_scene_ = std::move(result);
    instanced_dirty_ = false;
}

const InstancedSceneView& SceneDocument::instanced_render_scene() const {
    ensure_instanced_scene();
    return instanced_scene_;
}

bool SceneDocument::rebuild_render_scene() {
    Scene result;
    result.environment = state_.environment;
    Bounds3 bounds;
    std::unordered_map<AssetId, int> texture_bases;
    std::unordered_map<AssetId, int> default_material_bases;

    const auto ensure_texture_base =
        [&result, &texture_bases](const SceneMeshAsset& asset) {
            const auto found = texture_bases.find(asset.id);
            if (found != texture_bases.end()) {
                return found->second;
            }
            const int texture_base = static_cast<int>(result.textures.size());
            result.textures.insert(
                result.textures.end(),
                asset.local_scene.textures.begin(),
                asset.local_scene.textures.end());
            texture_bases.emplace(asset.id, texture_base);
            return texture_base;
        };
    const auto append_materials =
        [&result](
            const SceneMeshAsset& asset,
            const SceneObject* object,
            int texture_base) {
            const int material_base = static_cast<int>(result.materials.size());
            for (std::size_t index = 0;
                 index < asset.local_scene.materials.size();
                 ++index) {
                Material material = asset.local_scene.materials[index];
                if (object) {
                    const auto found = std::find_if(
                        object->material_overrides.begin(),
                        object->material_overrides.end(),
                        [index](const SceneMaterialOverride& candidate) {
                            return candidate.material_slot == index;
                        });
                    if (found != object->material_overrides.end()) {
                        apply_material_override(material, *found);
                    }
                }
                if (material.diffuse_texture_id >= 0) {
                    material.diffuse_texture_id += texture_base;
                }
                if (material.opacity_texture_id >= 0) {
                    material.opacity_texture_id += texture_base;
                }
                if (material.bump_texture_id >= 0) {
                    material.bump_texture_id += texture_base;
                }
                result.materials.push_back(material);
            }
            return material_base;
        };

    for (const SceneObject& object : state_.objects) {
        if (!is_effectively_visible(object.id) ||
            object.type != SceneObjectType::Mesh) {
            continue;
        }
        const auto asset = find_asset(object.asset_id);
        if (!asset) {
            continue;
        }
        const int texture_base = ensure_texture_base(*asset);
        int material_base = 0;
        if (object.material_overrides.empty()) {
            const auto found = default_material_bases.find(asset->id);
            if (found == default_material_bases.end()) {
                material_base = append_materials(*asset, nullptr, texture_base);
                default_material_bases.emplace(asset->id, material_base);
            } else {
                material_base = found->second;
            }
        } else {
            material_base = append_materials(*asset, &object, texture_base);
        }

        const Mat4 world = world_matrix(object.id);
        const Mat3 normal_matrix = world.topLeftCorner<3, 3>().inverse().transpose();
        for (const Triangle& triangle : asset->local_scene.triangles) {
            TriangleVertex vertices[3];
            for (int index = 0; index < 3; ++index) {
                const TriangleVertex& source = triangle.vertex(index);
                vertices[index] = source;
                vertices[index].position = transform_point(world, source.position);
                if (source.has_normal) {
                    vertices[index].normal = (normal_matrix * source.normal).normalized();
                }
                bounds.expand(vertices[index].position);
            }
            result.triangles.emplace_back(
                vertices[0],
                vertices[1],
                vertices[2],
                material_base + triangle.material_id());
        }
    }

    for (const SceneObject& object : state_.objects) {
        if (!is_effectively_visible(object.id)) {
            continue;
        }
        const Mat4 world = world_matrix(object.id);
        if (object.type == SceneObjectType::PointLight) {
            result.point_lights.push_back(
                PointLight{transform_point(world, Vec3::Zero()), object.light_color});
        } else if (object.type == SceneObjectType::DirectionalLight) {
            const Vec3 direction =
                (world.topLeftCorner<3, 3>() * Vec3(0.0f, 0.0f, -1.0f)).normalized();
            result.directional_lights.push_back(
                DirectionalLight{direction, object.light_color});
        }
    }

    if (!finite_bounds(bounds)) {
        bounds = Bounds3(
            Vec3(-0.5f, -0.5f, -0.5f),
            Vec3(0.5f, 0.5f, 0.5f));
    }
    render_scene_ = std::move(result);
    render_bounds_ = bounds;
    render_dirty_ = false;
    return true;
}

Color& SceneDocument::environment() {
    render_dirty_ = true;
    instanced_dirty_ = true;
    return state_.environment;
}

const Color& SceneDocument::environment() const {
    return state_.environment;
}

void SceneDocument::checkpoint() {
    if (history_.empty()) {
        history_.push_back(state_);
        history_cursor_ = 0;
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
    render_dirty_ = true;
    instanced_dirty_ = true;
    return true;
}

bool SceneDocument::redo() {
    if (!can_redo()) {
        return false;
    }
    state_ = history_[++history_cursor_];
    render_dirty_ = true;
    instanced_dirty_ = true;
    return true;
}

bool SceneDocument::can_undo() const {
    return history_cursor_ > 0;
}

bool SceneDocument::can_redo() const {
    return history_cursor_ + 1 < history_.size();
}

bool SceneDocument::dirty() const {
    return !saved_cursor_ || history_cursor_ != *saved_cursor_;
}

void SceneDocument::mark_saved() {
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
    root["version"] = 2;
    root["environment"] = vec3_json(state_.environment);
    root["assets"] = nlohmann::json::array();
    for (const auto& asset : assets_) {
        if (!referenced_asset_ids.contains(asset->id)) {
            continue;
        }
        if (session_snapshot) {
            nlohmann::json source;
            if (!asset->source_path.empty()) {
                source = {
                    {"kind", "obj"},
                    {"path", normalized_absolute(asset->source_path).generic_string()},
                };
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
            {"translation", vec3_json(object.transform.translation)},
            {"rotation_degrees", vec3_json(object.transform.rotation_degrees)},
            {"scale", vec3_json(object.transform.scale)},
            {"visible", object.visible},
            {"locked", object.locked},
            {"asset", object.asset_id},
            {"light_color", vec3_json(object.light_color)},
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
                    {"ior", material_override_value.ior},
                    {"opacity", material_override_value.opacity},
                    {"alpha_cutoff", material_override_value.alpha_cutoff},
                    {"bump_scale", material_override_value.bump_scale},
                    {"two_sided", material_override_value.two_sided},
                    {"use_diffuse_texture", material_override_value.use_diffuse_texture},
                    {"use_opacity_texture", material_override_value.use_opacity_texture},
                    {"use_bump_texture", material_override_value.use_bump_texture},
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
    const std::filesystem::path temporary = absolute_path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open scene file for writing: " + temporary.string());
    }
    output << root.dump(2) << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error("failed to write scene file: " + temporary.string());
    }
    std::error_code error;
    std::filesystem::rename(temporary, absolute_path, error);
    if (error) {
        std::filesystem::remove(absolute_path, error);
        error.clear();
        std::filesystem::rename(temporary, absolute_path, error);
    }
    if (error) {
        throw std::runtime_error("failed to replace scene file: " + error.message());
    }
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
    if (version != 1 && version != 2) {
        throw std::runtime_error("unsupported scene file version");
    }

    SceneDocument document;
    document.file_path_ = session_snapshot
        ? std::filesystem::path()
        : document_path;
    document.state_.objects.clear();
    document.assets_.clear();
    document.state_.environment = parse_vec3(root.at("environment"), "environment");
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
                document.assets_.push_back(asset);
                asset_ids[stored_id] = asset->id;
                continue;
            }
            if (kind != "obj") {
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
                const auto asset = document.load_asset(asset_path, width, height);
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
        const auto asset = document.load_asset(asset_path, width, height);
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
        object.transform.translation = parse_vec3(object_json.at("translation"), "translation");
        object.transform.rotation_degrees =
            parse_vec3(object_json.at("rotation_degrees"), "rotation_degrees");
        object.transform.scale = parse_vec3(object_json.at("scale"), "scale");
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
                material_override_value.ior =
                    override_json.at("ior").get<float>();
                material_override_value.opacity =
                    override_json.at("opacity").get<float>();
                material_override_value.alpha_cutoff =
                    override_json.at("alpha_cutoff").get<float>();
                material_override_value.bump_scale =
                    override_json.at("bump_scale").get<float>();
                material_override_value.two_sided =
                    override_json.value("two_sided", true);
                material_override_value.use_diffuse_texture =
                    override_json.value("use_diffuse_texture", true);
                material_override_value.use_opacity_texture =
                    override_json.value("use_opacity_texture", true);
                material_override_value.use_bump_texture =
                    override_json.value("use_bump_texture", true);
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
    document.render_dirty_ = true;
    document.instanced_dirty_ = true;
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
