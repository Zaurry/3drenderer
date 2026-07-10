#include "render/rasterizer/rasterizer_renderer.h"

#include "core/timer.h"
#include "scene/texture.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace renderer {

namespace {

struct ProjectedVertex {
    Vec3 screen;
    Vec3 world;
    double view_depth = 0.0;
    bool valid = false;
};

Color multiply(const Color& a, const Color& b) {
    return Color(a.x * b.x, a.y * b.y, a.z * b.z);
}

bool material_exists(const Scene& scene, int material_id) {
    return material_id >= 0 && static_cast<std::size_t>(material_id) < scene.materials.size();
}

ProjectedVertex project_to_screen(const Vec3& world, const Camera& camera, int width, int height) {
    ProjectedVertex out;
    const Vec3 relative = world - camera.eye();
    const double view_x = dot(relative, camera.right());
    const double view_y = dot(relative, camera.up());
    const double view_z = dot(relative, camera.forward());
    constexpr double near_epsilon = 1e-4;

    // v0.1 keeps clipping deliberately simple: demo geometry must be in front
    // of the near plane. Later versions can split triangles against the plane.
    if (view_z <= near_epsilon) {
        return out;
    }

    const double half_width = camera.viewport_width() * 0.5;
    const double half_height = camera.viewport_height() * 0.5;
    const double ndc_x = (view_x / view_z) / half_width;
    const double ndc_y = (view_y / view_z) / half_height;

    out.screen = Vec3(
        (ndc_x * 0.5 + 0.5) * static_cast<double>(width - 1),
        (1.0 - (ndc_y * 0.5 + 0.5)) * static_cast<double>(height - 1),
        view_z);
    out.world = world;
    out.view_depth = view_z;
    out.valid = true;
    return out;
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

RenderResult RasterizerRenderer::render(const Scene& scene, const Camera& camera, const RenderSettings& settings) {
    Timer timer;
    Image image(settings.width, settings.height);
    std::vector<double> depth_buffer(
        static_cast<std::size_t>(settings.width * settings.height),
        std::numeric_limits<double>::infinity());

    for (const Triangle& triangle : scene.triangles) {
        const ProjectedVertex v0 = project_to_screen(triangle.a(), camera, settings.width, settings.height);
        const ProjectedVertex v1 = project_to_screen(triangle.b(), camera, settings.width, settings.height);
        const ProjectedVertex v2 = project_to_screen(triangle.c(), camera, settings.width, settings.height);
        if (!v0.valid || !v1.valid || !v2.valid) {
            continue;
        }

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

        const Vec3 normal = normalize(cross(triangle.b() - triangle.a(), triangle.c() - triangle.a()));
        const Material* material = nullptr;
        if (material_exists(scene, triangle.material_id())) {
            material = &scene.materials[static_cast<std::size_t>(triangle.material_id())];
        }

        for (int y = start_y; y <= end_y; ++y) {
            for (int x = start_x; x <= end_x; ++x) {
                const Vec3 p(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5, 0.0);
                const double w0 = edge_function(v1.screen, v2.screen, p);
                const double w1 = edge_function(v2.screen, v0.screen, p);
                const double w2 = edge_function(v0.screen, v1.screen, p);
                const bool inside = area > 0.0
                    ? (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)
                    : (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
                if (!inside) {
                    continue;
                }

                const double b0 = w0 / area;
                const double b1 = w1 / area;
                const double b2 = w2 / area;
                const double inverse_depth =
                    b0 / v0.view_depth + b1 / v1.view_depth + b2 / v2.view_depth;
                if (inverse_depth <= 0.0) {
                    continue;
                }

                const double depth = 1.0 / inverse_depth;
                const int buffer_index = y * settings.width + x;
                if (depth >= depth_buffer[static_cast<std::size_t>(buffer_index)]) {
                    continue;
                }
                depth_buffer[static_cast<std::size_t>(buffer_index)] = depth;

                const Vec3 world_position =
                    (v0.world * (b0 / v0.view_depth) +
                     v1.world * (b1 / v1.view_depth) +
                     v2.world * (b2 / v2.view_depth)) /
                    inverse_depth;
                const Vec2 uv = triangle.interpolate_uv(b0, b1, b2);

                const Color color = material
                    ? shade_surface(
                        scene,
                        camera,
                        *material,
                        sample_material_base_color(scene, *material, uv),
                        world_position,
                        normal)
                    : Color(1.0, 0.0, 1.0);
                image.set_pixel(x, y, color);
            }
        }
    }

    return RenderResult{image, timer.elapsed_seconds()};
}

double RasterizerRenderer::edge_function(const Vec3& a, const Vec3& b, const Vec3& c) const {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

}  // namespace renderer
