#include "scene/scene.h"

#include <algorithm>
#include <cmath>

namespace renderer {

namespace {

constexpr float kPi = 3.14159265358979323846f;

Material diffuse(const Color& color) {
    Material material;
    material.type = MaterialType::Diffuse;
    material.base_color = color;
    return material;
}

Material metal(const Color& color, float roughness) {
    Material material;
    material.type = MaterialType::Metal;
    material.base_color = color;
    material.metallic = 1.0f;
    material.roughness = roughness;
    return material;
}

Material emissive(const Color& color, const Color& emission) {
    Material material;
    material.type = MaterialType::Emissive;
    material.base_color = color;
    material.emission = emission;
    return material;
}

void add_quad(
    Scene& scene,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const Vec3& d,
    int material_id) {
    scene.triangles.emplace_back(a, b, c, material_id);
    scene.triangles.emplace_back(a, c, d, material_id);
}

}  // namespace

void tessellate_spheres(
    Scene& scene,
    int longitude_segments,
    int latitude_segments) {
    longitude_segments = std::max(3, longitude_segments);
    latitude_segments = std::max(2, latitude_segments);
    if (scene.spheres.empty()) {
        return;
    }

    const auto make_vertex = [](
        const Sphere& sphere,
        int longitude,
        int latitude,
        int longitude_count,
        int latitude_count) {
        const float u = static_cast<float>(longitude) /
            static_cast<float>(longitude_count);
        const float v = static_cast<float>(latitude) /
            static_cast<float>(latitude_count);
        const float phi = u * 2.0f * kPi;
        const float theta = v * kPi;
        const float sin_theta = std::sin(theta);
        const Vec3 normal(
            sin_theta * std::cos(phi),
            std::cos(theta),
            sin_theta * std::sin(phi));
        TriangleVertex result;
        result.position = sphere.center() + normal * sphere.radius();
        result.uv = Vec2(u, v);
        result.uv1 = result.uv;
        result.has_uv1 = true;
        result.normal = normal;
        result.has_normal = true;
        result.tangent = Vec4(-std::sin(phi), 0.0f, std::cos(phi), 1.0f);
        result.has_tangent = true;
        return result;
    };

    const std::vector<Sphere> spheres = std::move(scene.spheres);
    scene.spheres.clear();
    scene.triangles.reserve(
        scene.triangles.size() +
        spheres.size() * static_cast<std::size_t>(
            2 * longitude_segments * (latitude_segments - 1)));
    for (const Sphere& sphere : spheres) {
        for (int latitude = 0; latitude < latitude_segments; ++latitude) {
            for (int longitude = 0; longitude < longitude_segments; ++longitude) {
                const TriangleVertex top_left = make_vertex(
                    sphere,
                    longitude,
                    latitude,
                    longitude_segments,
                    latitude_segments);
                const TriangleVertex top_right = make_vertex(
                    sphere,
                    longitude + 1,
                    latitude,
                    longitude_segments,
                    latitude_segments);
                const TriangleVertex bottom_left = make_vertex(
                    sphere,
                    longitude,
                    latitude + 1,
                    longitude_segments,
                    latitude_segments);
                const TriangleVertex bottom_right = make_vertex(
                    sphere,
                    longitude + 1,
                    latitude + 1,
                    longitude_segments,
                    latitude_segments);
                if (latitude > 0) {
                    scene.triangles.emplace_back(
                        top_left,
                        top_right,
                        bottom_right,
                        sphere.material_id());
                }
                if (latitude + 1 < latitude_segments) {
                    scene.triangles.emplace_back(
                        top_left,
                        bottom_right,
                        bottom_left,
                        sphere.material_id());
                }
            }
        }
    }
}

Scene make_gradient_sphere_scene() {
    Scene scene;
    scene.environment = Color(0.4f, 0.6f, 0.9f);
    scene.materials.push_back(diffuse(Color(0.8f, 0.25f, 0.15f)));
    scene.spheres.emplace_back(Vec3(0, 0, -1), 0.5f, 0);
    return scene;
}

Scene make_triangle_scene() {
    Scene scene;
    scene.environment = Color(0.05f, 0.06f, 0.08f);
    scene.materials.push_back(diffuse(Color(0.2f, 0.7f, 0.9f)));
    scene.triangles.emplace_back(
        Vec3(-0.8f, -0.5f, -1.5f),
        Vec3(0.8f, -0.5f, -1.5f),
        Vec3(0.0f, 0.7f, -1.5f),
        0);
    scene.directional_lights.push_back(
        DirectionalLight{Vec3(-1.0f, -1.0f, -1.0f).normalized(), Color(0.8f, 0.8f, 0.8f)});
    return scene;
}

Scene make_mirror_spheres_scene() {
    Scene scene;
    scene.environment = Color(0.02f, 0.03f, 0.06f);
    scene.materials.push_back(diffuse(Color(0.65f, 0.68f, 0.7f)));
    scene.materials.push_back(metal(Color(0.9f, 0.86f, 0.78f), 0.04f));
    scene.spheres.emplace_back(Vec3(0, -100.5f, -1), 100.0f, 0);
    scene.spheres.emplace_back(Vec3(0, 0, -1), 0.5f, 1);
    scene.point_lights.push_back(
        PointLight{Vec3(1.5f, 3.0f, 1.0f), Color(8.0f, 7.0f, 6.0f)});
    return scene;
}

Scene make_cornell_box_scene() {
    Scene scene;
    scene.environment = Color(0, 0, 0);

    const int red_id = 0;
    const int green_id = 1;
    const int white_id = 2;
    const int light_id = 3;
    scene.materials.push_back(diffuse(Color(0.75f, 0.12f, 0.08f)));
    scene.materials.push_back(diffuse(Color(0.12f, 0.45f, 0.15f)));
    scene.materials.push_back(diffuse(Color(0.73f, 0.73f, 0.68f)));
    scene.materials.push_back(
        emissive(Color(1.0f, 0.95f, 0.8f), Color(12.0f, 10.0f, 7.0f)));

    add_quad(scene, Vec3(-1, -1, -3), Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, -1, -3), white_id);
    add_quad(scene, Vec3(-1, 1, -1), Vec3(-1, 1, -3), Vec3(1, 1, -3), Vec3(1, 1, -1), white_id);
    add_quad(scene, Vec3(-1, -1, -3), Vec3(-1, 1, -3), Vec3(-1, 1, -1), Vec3(-1, -1, -1), red_id);
    add_quad(scene, Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(1, 1, -3), Vec3(1, -1, -3), green_id);
    add_quad(scene, Vec3(-1, -1, -3), Vec3(1, -1, -3), Vec3(1, 1, -3), Vec3(-1, 1, -3), white_id);
    add_quad(
        scene,
        Vec3(-0.35f, 0.99f, -2.35f),
        Vec3(0.35f, 0.99f, -2.35f),
        Vec3(0.35f, 0.99f, -1.65f),
        Vec3(-0.35f, 0.99f, -1.65f),
        light_id);

    return scene;
}

}  // namespace renderer
