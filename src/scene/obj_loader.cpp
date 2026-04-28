#include "scene/obj_loader.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4127 4244 4245 4267 4456 4505 4702 4996)
#endif

#define TINYOBJLOADER_DISABLE_FAST_FLOAT
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
    // We convert each referenced vertex into the renderer's double-precision Vec3.
    return Vec3(
        static_cast<double>(attrib.vertices[base]),
        static_cast<double>(attrib.vertices[base + 1U]),
        static_cast<double>(attrib.vertices[base + 2U]));
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
            mesh.triangles.emplace_back(
                vertex_from_index(attrib, i0),
                vertex_from_index(attrib, i1),
                vertex_from_index(attrib, i2),
                material_id);

            index_offset += static_cast<std::size_t>(vertex_count);
        }
    }

    return mesh;
}

}  // namespace renderer
