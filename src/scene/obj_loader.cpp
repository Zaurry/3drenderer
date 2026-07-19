#include "scene/obj_loader.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4127 4244 4245 4267 4456 4505 4702 4996)
#endif

#define TINYOBJLOADER_DISABLE_FAST_FLOAT
// The San Miguel OBJ files are 599 MiB and 1.06 GiB, respectively.  Keep a
// finite parser limit, but make it large enough for those assets and map files
// instead of copying the complete text into a second in-memory buffer.
#define TINYOBJLOADER_STREAM_READER_MAX_BYTES \
    (size_t(2) * size_t(1024) * size_t(1024) * size_t(1024))
#define TINYOBJLOADER_USE_MMAP
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace renderer {

namespace {

std::string loader_message(const tinyobj::ObjReader& reader, const std::string& path) {
    std::ostringstream out;
    out << "Failed to load OBJ '" << path << "'";
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

    // OBJ stores positions as a flat float array: x0, y0, z0, x1, y1, z1...
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
        return Vec2::Zero();
    }
    return Vec2(
        static_cast<float>(attrib.texcoords[base]),
        static_cast<float>(attrib.texcoords[base + 1U]));
}

Vec3 normal_from_index(const tinyobj::attrib_t& attrib, const tinyobj::index_t& index, bool& valid) {
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

}  // namespace

Mesh load_obj_mesh(const std::string& path, int material_id) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, config)) {
        throw std::runtime_error(loader_message(reader, path));
    }

    Mesh mesh;
    const tinyobj::attrib_t& attrib = reader.GetAttrib();
    for (const tinyobj::shape_t& shape : reader.GetShapes()) {
        std::size_t index_offset = 0;
        for (std::size_t face = 0; face < shape.mesh.num_face_vertices.size(); ++face) {
            const int vertex_count = shape.mesh.num_face_vertices[face];
            if (vertex_count != 3) {
                throw std::runtime_error("OBJ loader expected triangulated faces");
            }

            const tinyobj::index_t& i0 = shape.mesh.indices[index_offset + 0U];
            const tinyobj::index_t& i1 = shape.mesh.indices[index_offset + 1U];
            const tinyobj::index_t& i2 = shape.mesh.indices[index_offset + 2U];
            bool has_normal0 = false;
            bool has_normal1 = false;
            bool has_normal2 = false;
            mesh.triangles.emplace_back(
                TriangleVertex{
                    vertex_from_index(attrib, i0),
                    texcoord_from_index(attrib, i0),
                    normal_from_index(attrib, i0, has_normal0),
                    has_normal0},
                TriangleVertex{
                    vertex_from_index(attrib, i1),
                    texcoord_from_index(attrib, i1),
                    normal_from_index(attrib, i1, has_normal1),
                    has_normal1},
                TriangleVertex{
                    vertex_from_index(attrib, i2),
                    texcoord_from_index(attrib, i2),
                    normal_from_index(attrib, i2, has_normal2),
                    has_normal2},
                material_id);

            index_offset += static_cast<std::size_t>(vertex_count);
        }
    }

    return mesh;
}

}  // namespace renderer
