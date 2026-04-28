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
    scene.environment = Color(0.4, 0.6, 0.9);
    scene.materials.push_back(diffuse(Color(0.8, 0.25, 0.15)));
    scene.spheres.emplace_back(Vec3(0, 0, -1), 0.5, 0);
    return scene;
}

Scene make_raster_triangle_scene() {
    Scene scene;
    scene.environment = Color(0.05, 0.06, 0.08);
    scene.materials.push_back(diffuse(Color(0.2, 0.7, 0.9)));
    scene.triangles.emplace_back(
        Vec3(-0.8, -0.5, -1.5),
        Vec3(0.8, -0.5, -1.5),
        Vec3(0.0, 0.7, -1.5),
        0);
    scene.directional_lights.push_back(DirectionalLight{normalize(Vec3(-1, -1, -1)), Color(0.8, 0.8, 0.8)});
    return scene;
}

Scene make_mirror_spheres_scene() {
    Scene scene;
    scene.environment = Color(0.02, 0.03, 0.06);
    scene.materials.push_back(diffuse(Color(0.65, 0.68, 0.7)));
    scene.materials.push_back(metal(Color(0.9, 0.86, 0.78), 0.04));
    scene.spheres.emplace_back(Vec3(0, -100.5, -1), 100.0, 0);
    scene.spheres.emplace_back(Vec3(0, 0, -1), 0.5, 1);
    scene.point_lights.push_back(PointLight{Vec3(1.5, 3.0, 1.0), Color(8.0, 7.0, 6.0)});
    return scene;
}

Scene make_cornell_box_scene() {
    Scene scene;
    scene.environment = Color(0, 0, 0);

    const int red_id = 0;
    const int green_id = 1;
    const int white_id = 2;
    const int light_id = 3;
    scene.materials.push_back(diffuse(Color(0.75, 0.12, 0.08)));
    scene.materials.push_back(diffuse(Color(0.12, 0.45, 0.15)));
    scene.materials.push_back(diffuse(Color(0.73, 0.73, 0.68)));
    scene.materials.push_back(emissive(Color(1.0, 0.95, 0.8), Color(12.0, 10.0, 7.0)));

    add_quad(scene, Vec3(-1, -1, -3), Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, -1, -3), white_id);
    add_quad(scene, Vec3(-1, 1, -1), Vec3(-1, 1, -3), Vec3(1, 1, -3), Vec3(1, 1, -1), white_id);
    add_quad(scene, Vec3(-1, -1, -3), Vec3(-1, 1, -3), Vec3(-1, 1, -1), Vec3(-1, -1, -1), red_id);
    add_quad(scene, Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(1, 1, -3), Vec3(1, -1, -3), green_id);
    add_quad(scene, Vec3(-1, -1, -3), Vec3(1, -1, -3), Vec3(1, 1, -3), Vec3(-1, 1, -3), white_id);
    add_quad(scene, Vec3(-0.35, 0.99, -2.35), Vec3(0.35, 0.99, -2.35), Vec3(0.35, 0.99, -1.65), Vec3(-0.35, 0.99, -1.65), light_id);

    return scene;
}

}  // namespace renderer
