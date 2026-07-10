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
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
        static_cast<double>(attrib.vertices[base]),
        static_cast<double>(attrib.vertices[base + 1U]),
        static_cast<double>(attrib.vertices[base + 2U]));
}

Vec2 texcoord_from_index(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index) {
    if (index.texcoord_index < 0) {
        return Vec2();
    }

    const std::size_t base = static_cast<std::size_t>(index.texcoord_index) * 2U;
    if (base + 1U >= attrib.texcoords.size()) {
        throw std::runtime_error("OBJ texcoord index is out of range");
    }

    return Vec2(
        static_cast<double>(attrib.texcoords[base]),
        static_cast<double>(attrib.texcoords[base + 1U]));
}

Color array_to_color(const tinyobj::real_t values[3]) {
    return Color(
        static_cast<double>(values[0]),
        static_cast<double>(values[1]),
        static_cast<double>(values[2]));
}

double color_energy(const Color& color) {
    return color.x * color.x + color.y * color.y + color.z * color.z;
}

double roughness_from_shininess(double shininess) {
    if (!std::isfinite(shininess) || shininess <= 0.0) {
        return 0.2;
    }
    return std::clamp(1.0 / std::sqrt(shininess), 0.0, 1.0);
}

std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
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
    Scene& scene,
    std::unordered_map<std::string, int>& texture_cache) {
    if (texture_name.empty()) {
        return -1;
    }

    const std::filesystem::path resolved_path = resolve_texture_path(material_directory, texture_name);
    if (!std::filesystem::exists(resolved_path)) {
        return -1;
    }

    const std::string cache_key = lowercase_ascii(resolved_path.lexically_normal().string());
    const auto found = texture_cache.find(cache_key);
    if (found != texture_cache.end()) {
        return found->second;
    }

    const int texture_id = static_cast<int>(scene.textures.size());
    try {
        scene.textures.push_back(ImageTexture::load(resolved_path.string()));
    } catch (const std::exception&) {
        return -1;
    }
    texture_cache.emplace(cache_key, texture_id);
    return texture_id;
}

Material convert_material(
    const tinyobj::material_t& source,
    const std::filesystem::path& material_directory,
    Scene& scene,
    std::unordered_map<std::string, int>& texture_cache) {
    Material material;
    const Color diffuse = array_to_color(source.diffuse);
    const Color specular = array_to_color(source.specular);
    const Color emission = array_to_color(source.emission);
    const Color transmittance = array_to_color(source.transmittance);

    material.base_color = diffuse;
    material.emission = emission;
    material.ior = source.ior > 0.0 ? static_cast<double>(source.ior) : material.ior;
    material.roughness = roughness_from_shininess(static_cast<double>(source.shininess));
    material.diffuse_texture_id = texture_id_for(
        material_directory,
        source.diffuse_texname,
        scene,
        texture_cache);

    if (color_energy(emission) > 0.0) {
        material.type = MaterialType::Emissive;
        return material;
    }

    const bool has_transmission = source.illum == 7 || color_energy(transmittance) > 0.0;
    if (has_transmission) {
        material.type = MaterialType::Dielectric;
        material.base_color = color_energy(diffuse) > 0.0 ? diffuse : Color(1.0, 1.0, 1.0);
        return material;
    }

    const bool strong_specular = color_energy(specular) > 0.5;
    if (source.illum == 5 || strong_specular) {
        material.type = MaterialType::Metal;
        material.base_color = color_energy(specular) > 0.0 ? specular : diffuse;
        return material;
    }

    material.type = MaterialType::Diffuse;
    return material;
}

Material fallback_material() {
    Material material;
    material.type = MaterialType::Diffuse;
    material.base_color = Color(1.0, 0.0, 1.0);
    return material;
}

Camera make_default_camera(const Bounds3& bounds, int width, int height) {
    const Vec3 center = (bounds.min + bounds.max) * 0.5;
    const double radius = std::max(0.5, length(bounds.max - bounds.min) * 0.5);
    const double aspect = static_cast<double>(std::max(1, width)) / static_cast<double>(std::max(1, height));
    return Camera(
        center + Vec3(0.0, radius * 0.15, radius * 2.4),
        center,
        Vec3(0.0, 1.0, 0.0),
        45.0,
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

    std::unordered_map<std::string, int> texture_cache;
    for (const tinyobj::material_t& source : reader.GetMaterials()) {
        loaded.scene.materials.push_back(convert_material(source, parent_path, loaded.scene, texture_cache));
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
            loaded.scene.triangles.emplace_back(a, b, c, material_id, uv0, uv1, uv2);
            loaded.bounds.expand(a);
            loaded.bounds.expand(b);
            loaded.bounds.expand(c);

            index_offset += static_cast<std::size_t>(vertex_count);
        }
    }

    if (loaded.scene.triangles.empty()) {
        throw std::runtime_error("Scene asset contains no triangles");
    }

    loaded.scene.environment = Color(0.02, 0.025, 0.03);
    loaded.scene.directional_lights.push_back(
        DirectionalLight{normalize(Vec3(-0.5, -1.0, -0.25)), Color(0.25, 0.25, 0.25)});
    loaded.camera = make_default_camera(loaded.bounds, width, height);
    return loaded;
}

}  // namespace renderer
