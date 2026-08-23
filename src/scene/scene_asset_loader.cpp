#include "scene/scene_asset_loader.h"

#include "scene/gltf_loader.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4127 4244 4245 4267 4456 4505 4702 4996)
#endif

#define TINYOBJLOADER_DISABLE_FAST_FLOAT
#include "tiny_obj_loader.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace renderer {

namespace {

std::string loader_message(const tinyobj::ObjReader& reader, const std::string& path) {
    std::ostringstream out;
    out << "Failed to load scene asset '" << path << "'";
    if (!reader.Warning().empty()) {
        out << "\nwarning: " << reader.Warning();
    }
    if (!reader.Error().empty()) {
        out << "\nerror: " << reader.Error();
    }
    return out.str();
}

Vec3 vertex_from_index(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index) {
    if (index.vertex_index < 0) {
        throw std::runtime_error("OBJ face is missing a vertex index");
    }

    const std::size_t base = static_cast<std::size_t>(index.vertex_index) * 3U;
    if (base + 2U >= attrib.vertices.size()) {
        throw std::runtime_error("OBJ vertex index is out of range");
    }

    return Vec3(
        static_cast<float>(attrib.vertices[base]),
        static_cast<float>(attrib.vertices[base + 1U]),
        static_cast<float>(attrib.vertices[base + 2U]));
}

Vec2 texcoord_from_index(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index) {
    if (index.texcoord_index < 0) {
        return Vec2::Zero();
    }

    const std::size_t base = static_cast<std::size_t>(index.texcoord_index) * 2U;
    if (base + 1U >= attrib.texcoords.size()) {
        throw std::runtime_error("OBJ texcoord index is out of range");
    }

    return Vec2(
        static_cast<float>(attrib.texcoords[base]),
        static_cast<float>(attrib.texcoords[base + 1U]));
}

Vec3 normal_from_index(
    const tinyobj::attrib_t& attrib,
    const tinyobj::index_t& index,
    bool& valid) {
    valid = false;
    if (index.normal_index < 0) {
        return Vec3::Zero();
    }

    const std::size_t base = static_cast<std::size_t>(index.normal_index) * 3U;
    if (base + 2U >= attrib.normals.size()) {
        return Vec3::Zero();
    }

    const Vec3 normal(
        static_cast<float>(attrib.normals[base]),
        static_cast<float>(attrib.normals[base + 1U]),
        static_cast<float>(attrib.normals[base + 2U]));
    valid = usable_direction(normal);
    return valid ? normal.normalized() : Vec3::Zero();
}

Color array_to_color(const tinyobj::real_t values[3]) {
    return Color(
        static_cast<float>(values[0]),
        static_cast<float>(values[1]),
        static_cast<float>(values[2]));
}

float color_energy(const Color& color) {
    return color.squaredNorm();
}

float roughness_from_shininess(float shininess) {
    if (!std::isfinite(shininess) || shininess <= 0.0f) {
        return 0.2f;
    }
    return std::clamp(1.0f / std::sqrt(shininess), 0.0f, 1.0f);
}

bool illum_model_has_transmission(int illum) {
    // Wavefront illum models 4, 6, 7, and 9 explicitly enable glass
    // transparency or refraction. Exporters commonly write a default
    // `Tf 1 1 1` even for ordinary illum 2 materials, so Tf alone must not
    // turn an opaque surface into a dielectric.
    return illum == 4 || illum == 6 || illum == 7 || illum == 9;
}

std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

struct TextureCacheKey {
    std::string normalized_path;
    TextureEncoding encoding = TextureEncoding::Srgb;

    bool operator==(const TextureCacheKey&) const = default;
};

struct TextureCacheKeyHash {
    std::size_t operator()(const TextureCacheKey& key) const {
        const std::size_t path_hash = std::hash<std::string>{}(key.normalized_path);
        const std::size_t encoding_hash = std::hash<int>{}(static_cast<int>(key.encoding));
        return path_hash ^
            (encoding_hash + 0x9e3779b9U + (path_hash << 6U) + (path_hash >> 2U));
    }
};

void add_warning(
    std::vector<std::string>& warnings,
    std::unordered_set<std::string>& warning_keys,
    const std::string& key,
    const std::string& message) {
    if (warning_keys.insert(key).second) {
        warnings.push_back(message);
    }
}

std::filesystem::path resolve_texture_path(
    const std::filesystem::path& material_directory,
    const std::string& texture_name) {
    std::filesystem::path texture_path(texture_name);
    if (texture_path.is_relative()) {
        texture_path = material_directory / texture_path;
    }
    if (std::filesystem::exists(texture_path)) {
        return texture_path;
    }

    const std::filesystem::path directory = texture_path.parent_path().empty()
        ? std::filesystem::path(".")
        : texture_path.parent_path();
    const std::string target_name = lowercase_ascii(texture_path.filename().string());
    if (!std::filesystem::exists(directory)) {
        return texture_path;
    }

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && lowercase_ascii(entry.path().filename().string()) == target_name) {
            return entry.path();
        }
    }

    return texture_path;
}

int texture_id_for(
    const std::filesystem::path& material_directory,
    const std::string& texture_name,
    TextureEncoding encoding,
    const std::string& semantic,
    Scene& scene,
    std::unordered_map<TextureCacheKey, int, TextureCacheKeyHash>& texture_cache,
    std::vector<std::string>& warnings,
    std::unordered_set<std::string>& warning_keys) {
    if (texture_name.empty()) {
        return -1;
    }

    const std::filesystem::path resolved_path = resolve_texture_path(material_directory, texture_name);
    const std::string normalized_path = lowercase_ascii(resolved_path.lexically_normal().string());
    if (!std::filesystem::exists(resolved_path)) {
        add_warning(
            warnings,
            warning_keys,
            "missing:" + semantic + ":" + normalized_path,
            "Missing " + semantic + " texture: " + resolved_path.string());
        return -1;
    }

    const TextureCacheKey cache_key{normalized_path, encoding};
    const auto found = texture_cache.find(cache_key);
    if (found != texture_cache.end()) {
        return found->second;
    }

    const int texture_id = static_cast<int>(scene.textures.size());
    try {
        scene.textures.push_back(ImageTexture::load(resolved_path.string(), encoding));
    } catch (const std::exception& error) {
        add_warning(
            warnings,
            warning_keys,
            "decode:" + semantic + ":" + normalized_path,
            "Failed to decode " + semantic + " texture '" + resolved_path.string() +
                "': " + error.what());
        return -1;
    }
    texture_cache.emplace(cache_key, texture_id);
    return texture_id;
}

Material convert_material(
    const tinyobj::material_t& source,
    const std::filesystem::path& material_directory,
    Scene& scene,
    std::unordered_map<TextureCacheKey, int, TextureCacheKeyHash>& texture_cache,
    std::vector<std::string>& warnings,
    std::unordered_set<std::string>& warning_keys) {
    Material material;
    const Color diffuse = array_to_color(source.diffuse);
    const Color specular = array_to_color(source.specular);
    const Color emission = array_to_color(source.emission);

    material.base_color = diffuse;
    material.emission = emission;
    const float source_ior = static_cast<float>(source.ior);
    material.ior = source_ior > 0.0f ? source_ior : material.ior;
    material.roughness = roughness_from_shininess(static_cast<float>(source.shininess));
    material.opacity = std::clamp(static_cast<float>(source.dissolve), 0.0f, 1.0f);
    material.bump_scale = static_cast<float>(source.bump_texopt.bump_multiplier);
    material.two_sided = true;
    material.diffuse_texture_id = texture_id_for(
        material_directory,
        source.diffuse_texname,
        TextureEncoding::Srgb,
        "diffuse",
        scene,
        texture_cache,
        warnings,
        warning_keys);
    material.opacity_texture_id = texture_id_for(
        material_directory,
        source.alpha_texname,
        TextureEncoding::Linear,
        "opacity",
        scene,
        texture_cache,
        warnings,
        warning_keys);
    material.bump_texture_id = texture_id_for(
        material_directory,
        source.bump_texname,
        TextureEncoding::Linear,
        "bump",
        scene,
        texture_cache,
        warnings,
        warning_keys);

    if (color_energy(emission) > 0.0f) {
        material.type = MaterialType::Emissive;
        return material;
    }

    const bool has_transmission = illum_model_has_transmission(source.illum);
    if (has_transmission) {
        material.type = MaterialType::Dielectric;
        material.base_color = color_energy(diffuse) > 0.0f ? diffuse : Color(1.0f, 1.0f, 1.0f);
        return material;
    }

    const bool strong_specular = color_energy(specular) > 0.5f;
    if (source.illum == 5 || strong_specular) {
        material.type = MaterialType::Metal;
        material.base_color = color_energy(specular) > 0.0f ? specular : diffuse;
        return material;
    }

    material.type = MaterialType::Diffuse;
    return material;
}

Material fallback_material() {
    Material material;
    material.type = MaterialType::Diffuse;
    material.base_color = Color(1.0f, 0.0f, 1.0f);
    return material;
}

bool recognize_rect_area_light(
    const std::vector<std::array<Vec3, 3>>& triangles,
    const Material& material,
    RectAreaLight& result) {
    if (triangles.size() != 2 ||
        material.type != MaterialType::Emissive ||
        !material.emission.allFinite() ||
        color_energy(material.emission) <= 0.0f) {
        return false;
    }

    float scale = 0.0f;
    for (const auto& triangle : triangles) {
        for (const Vec3& a : triangle) {
            for (const auto& other_triangle : triangles) {
                for (const Vec3& b : other_triangle) {
                    scale = std::max(scale, (a - b).norm());
                }
            }
        }
    }
    if (!std::isfinite(scale) || scale <= 1.0e-8f) {
        return false;
    }
    const float position_tolerance = std::max(1.0e-6f, scale * 1.0e-5f);

    std::vector<Vec3> corners;
    corners.reserve(4);
    for (const auto& triangle : triangles) {
        for (const Vec3& point : triangle) {
            const bool duplicate = std::any_of(
                corners.begin(),
                corners.end(),
                [&](const Vec3& corner) {
                    return (point - corner).norm() <= position_tolerance;
                });
            if (!duplicate) {
                corners.push_back(point);
            }
        }
    }
    if (corners.size() != 4) {
        return false;
    }

    const Vec3 first_cross =
        (triangles[0][1] - triangles[0][0])
            .cross(triangles[0][2] - triangles[0][0]);
    const Vec3 second_cross =
        (triangles[1][1] - triangles[1][0])
            .cross(triangles[1][2] - triangles[1][0]);
    if (!usable_direction(first_cross) || !usable_direction(second_cross)) {
        return false;
    }
    const Vec3 normal = first_cross.normalized();
    if (normal.dot(second_cross.normalized()) < 0.999f) {
        return false;
    }

    for (std::size_t diagonal = 1; diagonal < corners.size(); ++diagonal) {
        std::array<std::size_t, 2> adjacent{};
        std::size_t adjacent_count = 0;
        for (std::size_t index = 1; index < corners.size(); ++index) {
            if (index != diagonal) {
                adjacent[adjacent_count++] = index;
            }
        }

        const Vec3 parallelogram_error =
            corners[0] + corners[diagonal] -
            corners[adjacent[0]] - corners[adjacent[1]];
        if (parallelogram_error.norm() > position_tolerance * 4.0f) {
            continue;
        }

        const Vec3 edge_a = corners[adjacent[0]] - corners[0];
        const Vec3 edge_b = corners[adjacent[1]] - corners[0];
        const float length_a = edge_a.norm();
        const float length_b = edge_b.norm();
        if (!std::isfinite(length_a) || !std::isfinite(length_b) ||
            length_a <= position_tolerance || length_b <= position_tolerance) {
            continue;
        }
        if (std::abs(edge_a.dot(edge_b)) > length_a * length_b * 1.0e-2f) {
            continue;
        }

        const float rectangle_area = edge_a.cross(edge_b).norm();
        const float triangle_area = 0.5f * (first_cross.norm() + second_cross.norm());
        if (!std::isfinite(rectangle_area) || rectangle_area <= 1.0e-12f ||
            std::abs(triangle_area - rectangle_area) > rectangle_area * 1.0e-3f) {
            continue;
        }

        result.position = (corners[0] + corners[diagonal]) * 0.5f;
        result.axis_u = edge_b * 0.5f;
        result.axis_v = edge_a * 0.5f;
        if (result.axis_v.cross(result.axis_u).dot(normal) < 0.0f) {
            result.axis_u = -result.axis_u;
        }
        result.radiance = material.emission.cwiseMax(Color::Zero());
        result.two_sided = false;
        return true;
    }
    return false;
}

Camera make_default_camera(const Bounds3& bounds, int width, int height) {
    const Vec3 center = (bounds.min + bounds.max) * 0.5f;
    const float radius = std::max(0.5f, (bounds.max - bounds.min).norm() * 0.5f);
    const float aspect = static_cast<float>(std::max(1, width)) /
        static_cast<float>(std::max(1, height));
    return Camera(
        center + Vec3(
            0.0f,
            radius * 0.15f,
            radius * 2.4f),
        center,
        Vec3(0.0f, 1.0f, 0.0f),
        45.0f,
        aspect);
}

Vec3 transform_point(const Mat4& transform, const Vec3& point) {
    return (transform * Vec4(point.x(), point.y(), point.z(), 1.0f)).head<3>();
}

Vec3 transform_direction(const Mat4& transform, const Vec3& direction) {
    const Vec3 result = transform.block<3, 3>(0, 0) * direction;
    return usable_direction(result) ? result.normalized() : direction;
}

TriangleVertex transform_vertex(
    const TriangleVertex& source,
    const Mat4& transform) {
    TriangleVertex result = source;
    result.position = transform_point(transform, source.position);
    const Eigen::Matrix3f linear = transform.block<3, 3>(0, 0);
    const float determinant = linear.determinant();
    const bool invertible = linear.allFinite() && std::isfinite(determinant) &&
        std::abs(determinant) > 1.0e-12f;
    if (source.has_normal) {
        if (invertible) {
            const Vec3 normal = linear.inverse().transpose() * source.normal;
            result.has_normal = usable_direction(normal);
            result.normal = result.has_normal ? normal.normalized() : Vec3::Zero();
        } else {
            result.has_normal = false;
            result.normal = Vec3::Zero();
        }
    }
    if (source.has_tangent) {
        const Vec3 tangent = linear * source.tangent.head<3>();
        result.has_tangent = invertible && usable_direction(tangent);
        if (result.has_tangent) {
            const Vec3 normalized_tangent = tangent.normalized();
            result.tangent = Vec4(
                normalized_tangent.x(),
                normalized_tangent.y(),
                normalized_tangent.z(),
                source.tangent.w() * (determinant < 0.0f ? -1.0f : 1.0f));
        } else {
            result.tangent = Vec4::Zero();
        }
    }
    return result;
}

LoadedScene load_flattened_gltf(
    const std::filesystem::path& path,
    int width,
    int height) {
    const LoadedGltfScene source = load_gltf_scene(path, width, height);
    LoadedScene loaded{
        Scene(),
        Camera(
            Vec3(0.0f, 0.0f, 1.0f),
            Vec3::Zero(),
            Vec3(0.0f, 1.0f, 0.0f),
            45.0f,
            1.0f),
        Bounds3()};
    loaded.warnings = source.warnings;
    if (!source.meshes.empty()) {
        loaded.scene.materials = source.meshes.front().scene.materials;
        loaded.scene.textures = source.meshes.front().scene.textures;
        loaded.material_names = source.meshes.front().material_names;
    }

    std::vector<Mat4> world_transforms(source.nodes.size(), Mat4::Identity());
    int first_camera_node = -1;
    for (std::size_t node_index = 0; node_index < source.nodes.size(); ++node_index) {
        const GltfNodeAsset& node = source.nodes[node_index];
        const Mat4 parent = node.parent_index >= 0
            ? world_transforms[static_cast<std::size_t>(node.parent_index)]
            : Mat4::Identity();
        const Mat4 world = parent * node.local_transform;
        world_transforms[node_index] = world;
        if (node.camera_index >= 0 && first_camera_node < 0) {
            first_camera_node = static_cast<int>(node_index);
        }
        if (node.mesh_index >= 0 &&
            static_cast<std::size_t>(node.mesh_index) < source.meshes.size()) {
            const Scene& mesh = source.meshes[static_cast<std::size_t>(node.mesh_index)].scene;
            for (const Triangle& triangle : mesh.triangles) {
                const TriangleVertex a = transform_vertex(triangle.vertex(0), world);
                const TriangleVertex b = transform_vertex(triangle.vertex(1), world);
                const TriangleVertex c = transform_vertex(triangle.vertex(2), world);
                loaded.scene.triangles.emplace_back(a, b, c, triangle.material_id());
                loaded.bounds.expand(a.position);
                loaded.bounds.expand(b.position);
                loaded.bounds.expand(c.position);
            }
        }

        const Vec3 position = transform_point(world, Vec3::Zero());
        const Vec3 direction = transform_direction(world, Vec3(0.0f, 0.0f, -1.0f));
        const Color intensity = node.light_color * std::max(0.0f, node.light_intensity);
        switch (node.light_type) {
            case GltfNodeAsset::LightType::Directional:
                loaded.scene.directional_lights.push_back(
                    DirectionalLight{direction, intensity});
                break;
            case GltfNodeAsset::LightType::Point:
                loaded.scene.point_lights.push_back(
                    PointLight{position, intensity, node.light_range});
                break;
            case GltfNodeAsset::LightType::Spot:
                loaded.scene.spot_lights.push_back(SpotLight{
                    position,
                    direction,
                    intensity,
                    node.light_range,
                    node.spot_inner_cone_radians,
                    node.spot_outer_cone_radians});
                break;
            case GltfNodeAsset::LightType::None:
                break;
        }
    }

    if (loaded.scene.triangles.empty()) {
        loaded.bounds = Bounds3(
            Vec3(-0.5f, -0.5f, -0.5f),
            Vec3(0.5f, 0.5f, 0.5f));
    }
    loaded.camera = make_default_camera(loaded.bounds, width, height);
    if (first_camera_node >= 0) {
        const GltfNodeAsset& node = source.nodes[static_cast<std::size_t>(first_camera_node)];
        if (static_cast<std::size_t>(node.camera_index) < source.cameras.size()) {
            const GltfCameraAsset& camera =
                source.cameras[static_cast<std::size_t>(node.camera_index)];
            const Mat4& world = world_transforms[static_cast<std::size_t>(first_camera_node)];
            const Vec3 eye = transform_point(world, Vec3::Zero());
            const Vec3 forward = transform_direction(world, Vec3(0.0f, 0.0f, -1.0f));
            const Vec3 up = transform_direction(world, Vec3(0.0f, 1.0f, 0.0f));
            const float fov_degrees = camera.orthographic
                ? 45.0f
                : camera.vertical_fov_radians * 180.0f / 3.14159265358979323846f;
            const float aspect = camera.aspect_ratio > 0.0f
                ? camera.aspect_ratio
                : static_cast<float>(std::max(1, width)) /
                    static_cast<float>(std::max(1, height));
            loaded.camera = Camera(eye, eye + forward, up, fov_degrees, aspect);
            if (camera.orthographic) {
                loaded.warnings.push_back(
                    "The offline renderer uses a perspective approximation for the glTF orthographic camera.");
            }
        }
    }
    loaded.scene.environment = Color(0.02f, 0.025f, 0.03f);
    return loaded;
}

}  // namespace

LoadedScene load_scene_asset(const std::string& path, int width, int height) {
    const std::string extension = lowercase_ascii(
        std::filesystem::path(path).extension().string());
    if (extension == ".gltf" || extension == ".glb") {
        return load_flattened_gltf(path, width, height);
    }
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    const std::filesystem::path obj_path(path);
    const std::filesystem::path parent_path = obj_path.parent_path();
    if (!parent_path.empty()) {
        config.mtl_search_path = parent_path.string();
    }

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, config)) {
        throw std::runtime_error(loader_message(reader, path));
    }

    LoadedScene loaded{
        Scene(),
        Camera(Vec3(0.0f, 0.0f, 1.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), 45.0f, 1.0f),
        Bounds3()};

    std::unordered_set<std::string> warning_keys;
    if (!reader.Warning().empty()) {
        add_warning(loaded.warnings, warning_keys, "tinyobj", reader.Warning());
    }
    std::unordered_map<TextureCacheKey, int, TextureCacheKeyHash> texture_cache;
    std::vector<tinyobj::material_t> source_materials = reader.GetMaterials();
    std::unordered_map<std::string, int> material_ids_by_name;
    for (std::size_t index = 0; index < source_materials.size(); ++index) {
        material_ids_by_name.emplace(
            lowercase_ascii(source_materials[index].name),
            static_cast<int>(index));
    }

    // Some archives ship an OBJ whose mtllib line points at a related, but
    // incomplete, material library. If tinyobj reports a missing material,
    // use a same-stem MTL as a supplemental source without overriding any
    // material that the OBJ explicitly resolved.
    if (reader.Warning().find("not found in .mtl") != std::string::npos) {
        std::filesystem::path supplemental_mtl = obj_path;
        supplemental_mtl.replace_extension(".mtl");
        if (std::filesystem::exists(supplemental_mtl)) {
            const std::size_t material_count_before_supplement =
                source_materials.size();
            std::ifstream input(supplemental_mtl);
            std::map<std::string, int> supplemental_map;
            std::vector<tinyobj::material_t> supplemental_materials;
            std::string supplemental_warning;
            std::string supplemental_error;
            tinyobj::LoadMtl(
                &supplemental_map,
                &supplemental_materials,
                &input,
                &supplemental_warning,
                &supplemental_error);
            for (const tinyobj::material_t& source : supplemental_materials) {
                const std::string key = lowercase_ascii(source.name);
                if (material_ids_by_name.contains(key)) {
                    continue;
                }
                material_ids_by_name.emplace(
                    key,
                    static_cast<int>(source_materials.size()));
                source_materials.push_back(source);
            }
            if (source_materials.size() > material_count_before_supplement) {
                add_warning(
                    loaded.warnings,
                    warning_keys,
                    "supplemental-mtl-recovered:" + supplemental_mtl.string(),
                    "Recovered missing OBJ materials from same-stem library '" +
                        supplemental_mtl.string() + "'.");
            }
            if (!supplemental_error.empty()) {
                add_warning(
                    loaded.warnings,
                    warning_keys,
                    "supplemental-mtl:" + supplemental_mtl.string(),
                    "Failed to read supplemental material library '" +
                        supplemental_mtl.string() + "': " + supplemental_error);
            }
        }
    }

    for (const tinyobj::material_t& source : source_materials) {
        loaded.scene.materials.push_back(convert_material(
            source,
            parent_path,
            loaded.scene,
            texture_cache,
            loaded.warnings,
            warning_keys));
        loaded.material_names.push_back(
            source.name.empty()
                ? "Material " + std::to_string(loaded.material_names.size() + 1)
                : source.name);
    }
    const int fallback_material_id = static_cast<int>(loaded.scene.materials.size());
    loaded.scene.materials.push_back(fallback_material());
    loaded.material_names.push_back("<default>");
    int recovered_light_material_id = -1;

    const auto recover_named_light_material = [&]() {
        if (recovered_light_material_id >= 0) {
            return recovered_light_material_id;
        }
        Material light;
        light.type = MaterialType::Emissive;
        light.base_color = Color(0.78f, 0.78f, 0.78f);
        light.emission = Color(10.0f, 10.0f, 10.0f);
        light.two_sided = true;
        recovered_light_material_id =
            static_cast<int>(loaded.scene.materials.size());
        loaded.scene.materials.push_back(light);
        loaded.material_names.push_back("light (recovered)");
        add_warning(
            loaded.warnings,
            warning_keys,
            "recovered-named-light",
            "OBJ group 'light' has no resolved material; treating its rectangular "
            "geometry as a white emissive light with radiance 10.");
        return recovered_light_material_id;
    };

    const tinyobj::attrib_t& attrib = reader.GetAttrib();
    std::unordered_set<std::size_t> promoted_light_triangle_indices;
    for (const tinyobj::shape_t& shape : reader.GetShapes()) {
        std::unordered_map<int, std::vector<std::array<Vec3, 3>>>
            emissive_triangles_by_material;
        std::unordered_map<int, std::vector<std::size_t>>
            emissive_triangle_indices_by_material;
        std::size_t index_offset = 0;
        for (std::size_t face = 0; face < shape.mesh.num_face_vertices.size(); ++face) {
            const int vertex_count = shape.mesh.num_face_vertices[face];
            if (vertex_count != 3) {
                throw std::runtime_error("Scene asset loader expected triangulated faces");
            }

            const tinyobj::index_t& i0 = shape.mesh.indices[index_offset + 0U];
            const tinyobj::index_t& i1 = shape.mesh.indices[index_offset + 1U];
            const tinyobj::index_t& i2 = shape.mesh.indices[index_offset + 2U];
            const int source_material_id = face < shape.mesh.material_ids.size()
                ? shape.mesh.material_ids[face]
                : -1;
            int material_id =
                source_material_id >= 0 &&
                    static_cast<std::size_t>(source_material_id) < reader.GetMaterials().size()
                ? source_material_id
                : fallback_material_id;
            if (source_material_id < 0) {
                const auto supplemental = material_ids_by_name.find(
                    lowercase_ascii(shape.name));
                if (supplemental != material_ids_by_name.end()) {
                    material_id = supplemental->second;
                } else if (lowercase_ascii(shape.name) == "light") {
                    material_id = recover_named_light_material();
                }
            }

            const Vec3 a = vertex_from_index(attrib, i0);
            const Vec3 b = vertex_from_index(attrib, i1);
            const Vec3 c = vertex_from_index(attrib, i2);
            const Vec2 uv0 = texcoord_from_index(attrib, i0);
            const Vec2 uv1 = texcoord_from_index(attrib, i1);
            const Vec2 uv2 = texcoord_from_index(attrib, i2);
            bool has_normal0 = false;
            bool has_normal1 = false;
            bool has_normal2 = false;
            const Vec3 normal0 = normal_from_index(attrib, i0, has_normal0);
            const Vec3 normal1 = normal_from_index(attrib, i1, has_normal1);
            const Vec3 normal2 = normal_from_index(attrib, i2, has_normal2);
            if (!has_normal0 || !has_normal1 || !has_normal2) {
                add_warning(
                    loaded.warnings,
                    warning_keys,
                    "normal-fallback",
                    "OBJ contains faces without valid vertex normals; using geometric normals.");
            }
            loaded.scene.triangles.emplace_back(
                TriangleVertex{a, uv0, normal0, has_normal0},
                TriangleVertex{b, uv1, normal1, has_normal1},
                TriangleVertex{c, uv2, normal2, has_normal2},
                material_id);
            if (material_id >= 0 &&
                static_cast<std::size_t>(material_id) < loaded.scene.materials.size() &&
                loaded.scene.materials[static_cast<std::size_t>(material_id)].type ==
                    MaterialType::Emissive) {
                emissive_triangles_by_material[material_id].push_back({a, b, c});
                emissive_triangle_indices_by_material[material_id].push_back(
                    loaded.scene.triangles.size() - 1U);
            }
            if (material_id >= 0 &&
                static_cast<std::size_t>(material_id) < loaded.scene.materials.size() &&
                loaded.scene.materials[static_cast<std::size_t>(material_id)].bump_texture_id >= 0 &&
                !loaded.scene.triangles.back().has_valid_uv_basis()) {
                add_warning(
                    loaded.warnings,
                    warning_keys,
                    "degenerate-bump-uv:" + std::to_string(material_id),
                    "Material " + std::to_string(material_id) +
                        " has a bump map on a triangle with degenerate UVs; using the unperturbed normal.");
            }
            loaded.bounds.expand(a);
            loaded.bounds.expand(b);
            loaded.bounds.expand(c);

            index_offset += static_cast<std::size_t>(vertex_count);
        }

        for (const auto& [material_id, triangles] : emissive_triangles_by_material) {
            RectAreaLight light;
            if (recognize_rect_area_light(
                    triangles,
                    loaded.scene.materials[static_cast<std::size_t>(material_id)],
                    light)) {
                loaded.scene.rect_area_lights.push_back(light);
                const auto indices =
                    emissive_triangle_indices_by_material.find(material_id);
                if (indices != emissive_triangle_indices_by_material.end()) {
                    promoted_light_triangle_indices.insert(
                        indices->second.begin(),
                        indices->second.end());
                }
            }
        }
    }

    if (loaded.scene.triangles.empty()) {
        throw std::runtime_error("Scene asset contains no triangles");
    }

    if (!promoted_light_triangle_indices.empty()) {
        std::vector<Triangle> retained_triangles;
        retained_triangles.reserve(
            loaded.scene.triangles.size() - promoted_light_triangle_indices.size());
        for (std::size_t index = 0; index < loaded.scene.triangles.size(); ++index) {
            if (!promoted_light_triangle_indices.contains(index)) {
                retained_triangles.push_back(std::move(loaded.scene.triangles[index]));
            }
        }
        loaded.scene.triangles = std::move(retained_triangles);
    }

    loaded.scene.environment = Color(0.02f, 0.025f, 0.03f);
    if (loaded.scene.rect_area_lights.empty()) {
        loaded.scene.directional_lights.push_back(
            DirectionalLight{
                Vec3(-0.5f, -1.0f, -0.25f).normalized(),
                Color(25.0f, 25.0f, 25.0f)});
    }
    loaded.camera = make_default_camera(loaded.bounds, width, height);
    return loaded;
}

}  // namespace renderer
