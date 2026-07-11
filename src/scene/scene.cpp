#include "scene/scene.h"

namespace renderer {

namespace {

Material diffuse(const Color& color) {
    Material material;
    material.type = MaterialType::Diffuse;
    material.base_color = color;
    return material;
}

Material metal(const Color& color, double roughness) {
    Material material;
    material.type = MaterialType::Metal;
    material.base_color = color;
    material.metallic = 1.0;
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

Scene make_gradient_sphere_scene() {
    Scene scene;
    scene.environment = Color(0.4f, 0.6f, 0.9f);
    scene.materials.push_back(diffuse(Color(0.8f, 0.25f, 0.15f)));
    scene.spheres.emplace_back(Vec3(0, 0, -1), 0.5, 0);
    return scene;
}

Scene make_raster_triangle_scene() {
    Scene scene;
    scene.environment = Color(0.05f, 0.06f, 0.08f);
    scene.materials.push_back(diffuse(Color(0.2f, 0.7f, 0.9f)));
    scene.triangles.emplace_back(
        Vec3(-0.8f, -0.5f, -1.5f),
        Vec3(0.8f, -0.5f, -1.5f),
        Vec3(0.0f, 0.7f, -1.5f),
        0);
    scene.directional_lights.push_back(
        DirectionalLight{normalize(Vec3(-1, -1, -1)), Color(0.8f, 0.8f, 0.8f)});
    return scene;
}

Scene make_mirror_spheres_scene() {
    Scene scene;
    scene.environment = Color(0.02f, 0.03f, 0.06f);
    scene.materials.push_back(diffuse(Color(0.65f, 0.68f, 0.7f)));
    scene.materials.push_back(metal(Color(0.9f, 0.86f, 0.78f), 0.04));
    scene.spheres.emplace_back(Vec3(0, -100.5, -1), 100.0, 0);
    scene.spheres.emplace_back(Vec3(0, 0, -1), 0.5, 1);
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
