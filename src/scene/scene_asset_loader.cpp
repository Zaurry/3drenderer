#include "scene/scene_asset_loader.h"

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
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <functional>
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
    const Color transmittance = array_to_color(source.transmittance);

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

    const bool has_transmission = source.illum == 7 || color_energy(transmittance) > 0.0f;
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

}  // namespace

LoadedScene load_scene_asset(const std::string& path, int width, int height) {
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
        Camera(Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 45.0, 1.0),
        Bounds3()};

    std::unordered_set<std::string> warning_keys;
    if (!reader.Warning().empty()) {
        add_warning(loaded.warnings, warning_keys, "tinyobj", reader.Warning());
    }
    std::unordered_map<TextureCacheKey, int, TextureCacheKeyHash> texture_cache;
    for (const tinyobj::material_t& source : reader.GetMaterials()) {
        loaded.scene.materials.push_back(convert_material(
            source,
            parent_path,
            loaded.scene,
            texture_cache,
            loaded.warnings,
            warning_keys));
    }
    const int fallback_material_id = static_cast<int>(loaded.scene.materials.size());
    loaded.scene.materials.push_back(fallback_material());

    const tinyobj::attrib_t& attrib = reader.GetAttrib();
    for (const tinyobj::shape_t& shape : reader.GetShapes()) {
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
            const int material_id =
                source_material_id >= 0 &&
                    static_cast<std::size_t>(source_material_id) < reader.GetMaterials().size()
                ? source_material_id
                : fallback_material_id;

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
    }

    if (loaded.scene.triangles.empty()) {
        throw std::runtime_error("Scene asset contains no triangles");
    }

    loaded.scene.environment = Color(0.02f, 0.025f, 0.03f);
    loaded.scene.directional_lights.push_back(
        DirectionalLight{
            Vec3(-0.5f, -1.0f, -0.25f).normalized(),
            Color(0.25f, 0.25f, 0.25f)});
    loaded.camera = make_default_camera(loaded.bounds, width, height);
    return loaded;
}

}  // namespace renderer
