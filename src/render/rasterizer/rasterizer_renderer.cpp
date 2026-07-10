#include "render/rasterizer/rasterizer_renderer.h"

#include "core/timer.h"
#include "render/rasterizer/raster_geometry.h"
#include "scene/material_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace renderer {

namespace {

constexpr double near_plane = 1e-4;

struct ProjectedVertex {
    Vec3 screen;
    RasterVertex attributes;
};

Color multiply(const Color& a, const Color& b) {
    return Color(a.x * b.x, a.y * b.y, a.z * b.z);
}

bool material_exists(const Scene& scene, int material_id) {
    return material_id >= 0 && static_cast<std::size_t>(material_id) < scene.materials.size();
}

Vec3 view_position(const Vec3& world, const Camera& camera) {
    const Vec3 relative = world - camera.eye();
    return Vec3(
        dot(relative, camera.right()),
        dot(relative, camera.up()),
        dot(relative, camera.forward()));
}

ProjectedVertex project_to_screen(
    const RasterVertex& vertex,
    const Camera& camera,
    int width,
    int height) {
    const double half_width = camera.viewport_width() * 0.5;
    const double half_height = camera.viewport_height() * 0.5;
    const double ndc_x = (vertex.view.x / vertex.view.z) / half_width;
    const double ndc_y = (vertex.view.y / vertex.view.z) / half_height;

    ProjectedVertex projected;
    projected.screen = Vec3(
        (ndc_x * 0.5 + 0.5) * static_cast<double>(width - 1),
        (1.0 - (ndc_y * 0.5 + 0.5)) * static_cast<double>(height - 1),
        vertex.view.z);
    projected.attributes = vertex;
    return projected;
}

Vec3 vertex_normal(const TriangleVertex& vertex, const Vec3& geometric_normal) {
    if (!vertex.has_normal || !usable_direction(vertex.normal)) {
        return geometric_normal;
    }
    return dot(vertex.normal, geometric_normal) < 0.0 ? -vertex.normal : vertex.normal;
}

RasterVertex make_raster_vertex(
    const TriangleVertex& vertex,
    const Vec3& geometric_normal,
    const Camera& camera) {
    RasterVertex raster;
    raster.view = view_position(vertex.position, camera);
    raster.world = vertex.position;
    raster.uv = vertex.uv;
    raster.normal = vertex_normal(vertex, geometric_normal);
    return raster;
}

Color shade_surface(
    const Scene& scene,
    const Camera& camera,
    const Material& material,
    const Color& base_color,
    const Vec3& position,
    const Vec3& normal) {
    if (material.type == MaterialType::Emissive) {
        return material.emission;
    }

    const Vec3 unit_normal = normalize(normal);
    const Vec3 view_dir = normalize(camera.eye() - position);
    Color shaded = multiply(base_color, scene.environment) * 0.15;

    for (const DirectionalLight& light : scene.directional_lights) {
        const Vec3 light_dir = normalize(-light.direction);
        const double n_dot_l = std::max(0.0, dot(unit_normal, light_dir));
        if (n_dot_l <= 0.0) {
            continue;
        }

        const Vec3 half_vector = normalize(light_dir + view_dir);
        const double specular = std::pow(std::max(0.0, dot(unit_normal, half_vector)), 32.0) * 0.2;
        shaded += multiply(base_color, light.radiance) * n_dot_l + light.radiance * specular;
    }

    for (const PointLight& light : scene.point_lights) {
        const Vec3 to_light = light.position - position;
        const double distance_squared = std::max(length_squared(to_light), 1e-12);
        const Vec3 light_dir = to_light / std::sqrt(distance_squared);
        const double n_dot_l = std::max(0.0, dot(unit_normal, light_dir));
        if (n_dot_l <= 0.0) {
            continue;
        }

        const Vec3 radiance = light.intensity / distance_squared;
        const Vec3 half_vector = normalize(light_dir + view_dir);
        const double specular = std::pow(std::max(0.0, dot(unit_normal, half_vector)), 32.0) * 0.2;
        shaded += multiply(base_color, radiance) * n_dot_l + radiance * specular;
    }

    return shaded;
}

}  // namespace

RenderResult RasterizerRenderer::render(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings) {
    Timer timer;
    Image image(settings.width, settings.height);
    std::vector<double> depth_buffer(
        static_cast<std::size_t>(settings.width * settings.height),
        std::numeric_limits<double>::infinity());

    for (const Triangle& triangle : scene.triangles) {
        const Material* material = material_exists(scene, triangle.material_id())
            ? &scene.materials[static_cast<std::size_t>(triangle.material_id())]
            : nullptr;
        const Vec3 geometric_normal = triangle.geometric_normal();
        if (!usable_direction(geometric_normal)) {
            continue;
        }

        const bool front_facing = dot(geometric_normal, camera.eye() - triangle.centroid()) > 0.0;
        if (material && !material->two_sided && !front_facing) {
            continue;
        }

        const std::array<RasterVertex, 3> original{
            make_raster_vertex(triangle.vertex(0), geometric_normal, camera),
            make_raster_vertex(triangle.vertex(1), geometric_normal, camera),
            make_raster_vertex(triangle.vertex(2), geometric_normal, camera)};
        const std::vector<RasterVertex> clipped =
            clip_triangle_to_near_plane(original, near_plane);
        if (clipped.size() < 3) {
            continue;
        }

        for (std::size_t fan = 1; fan + 1 < clipped.size(); ++fan) {
            const ProjectedVertex v0 = project_to_screen(
                clipped[0], camera, settings.width, settings.height);
            const ProjectedVertex v1 = project_to_screen(
                clipped[fan], camera, settings.width, settings.height);
            const ProjectedVertex v2 = project_to_screen(
                clipped[fan + 1], camera, settings.width, settings.height);

            const double area = edge_function(v0.screen, v1.screen, v2.screen);
            if (std::abs(area) <= 1e-12) {
                continue;
            }

            const double min_x = std::min({v0.screen.x, v1.screen.x, v2.screen.x});
            const double max_x = std::max({v0.screen.x, v1.screen.x, v2.screen.x});
            const double min_y = std::min({v0.screen.y, v1.screen.y, v2.screen.y});
            const double max_y = std::max({v0.screen.y, v1.screen.y, v2.screen.y});
            const int start_x = std::max(0, static_cast<int>(std::floor(min_x)));
            const int end_x = std::min(settings.width - 1, static_cast<int>(std::ceil(max_x)));
            const int start_y = std::max(0, static_cast<int>(std::floor(min_y)));
            const int end_y = std::min(settings.height - 1, static_cast<int>(std::ceil(max_y)));

            for (int y = start_y; y <= end_y; ++y) {
                for (int x = start_x; x <= end_x; ++x) {
                    const Vec3 pixel(
                        static_cast<double>(x) + 0.5,
                        static_cast<double>(y) + 0.5,
                        0.0);
                    const double w0 = edge_function(v1.screen, v2.screen, pixel);
                    const double w1 = edge_function(v2.screen, v0.screen, pixel);
                    const double w2 = edge_function(v0.screen, v1.screen, pixel);
                    const bool inside = area > 0.0
                        ? (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)
                        : (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
                    if (!inside) {
                        continue;
                    }

                    const Vec3 screen_weights(w0 / area, w1 / area, w2 / area);
                    const Vec3 view_depths(
                        v0.attributes.view.z,
                        v1.attributes.view.z,
                        v2.attributes.view.z);
                    const Vec3 weights = perspective_correct_weights(screen_weights, view_depths);
                    if (weights.x + weights.y + weights.z <= 0.0) {
                        continue;
                    }
                    const double inverse_depth =
                        screen_weights.x / view_depths.x +
                        screen_weights.y / view_depths.y +
                        screen_weights.z / view_depths.z;
                    const double depth = 1.0 / inverse_depth;
                    const int buffer_index = y * settings.width + x;
                    if (depth >= depth_buffer[static_cast<std::size_t>(buffer_index)]) {
                        continue;
                    }

                    const Vec3 world_position =
                        v0.attributes.world * weights.x +
                        v1.attributes.world * weights.y +
                        v2.attributes.world * weights.z;
                    const Vec2 uv(
                        v0.attributes.uv.x * weights.x +
                            v1.attributes.uv.x * weights.y +
                            v2.attributes.uv.x * weights.z,
                        v0.attributes.uv.y * weights.x +
                            v1.attributes.uv.y * weights.y +
                            v2.attributes.uv.y * weights.z);
                    Vec3 interpolated_normal = normalize(
                        v0.attributes.normal * weights.x +
                        v1.attributes.normal * weights.y +
                        v2.attributes.normal * weights.z);
                    if (!usable_direction(interpolated_normal)) {
                        interpolated_normal = geometric_normal;
                    }

                    HitRecord hit;
                    hit.position = world_position;
                    hit.uv = uv;
                    hit.material_id = triangle.material_id();
                    hit.set_normals(
                        Ray(camera.eye(), world_position - camera.eye()),
                        geometric_normal,
                        interpolated_normal);
                    triangle.tangent_basis(hit.shading_normal, hit.tangent, hit.bitangent);

                    Color color(1.0, 0.0, 1.0);
                    if (material) {
                        const SurfaceMaterialSample surface =
                            evaluate_surface_material(scene, *material, hit);
                        if (surface.opacity < material->alpha_cutoff) {
                            continue;
                        }
                        color = shade_surface(
                            scene,
                            camera,
                            *material,
                            surface.base_color,
                            world_position,
                            surface.shading_normal);
                    }

                    depth_buffer[static_cast<std::size_t>(buffer_index)] = depth;
                    image.set_pixel(x, y, color);
                }
            }
        }
    }

    return RenderResult{image, timer.elapsed_seconds()};
}

double RasterizerRenderer::edge_function(
    const Vec3& a,
    const Vec3& b,
    const Vec3& c) const {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

}  // namespace renderer
