#include "render/raytracer/raytracer_renderer.h"

#include "core/timer.h"
#include "sampling/sampler.h"

#include <algorithm>
#include <cmath>

namespace renderer {

namespace {

Color multiply(const Color& a, const Color& b) {
    return Color(a.x * b.x, a.y * b.y, a.z * b.z);
}

Color black() {
    return Color(0.0, 0.0, 0.0);
}

bool material_exists(const Scene& scene, int material_id) {
    return material_id >= 0 && static_cast<std::size_t>(material_id) < scene.materials.size();
}

double reflectance(double cosine, double refraction_index) {
    double r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 *= r0;
    return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
}

Color background_color(const Scene& scene, const RenderSettings& settings) {
    if (length_squared(scene.environment) > 0.0) {
        return scene.environment;
    }
    return settings.background;
}

}  // namespace

RenderResult RayTracerRenderer::render(const Scene& scene, const Camera& camera, const RenderSettings& settings) {
    Timer timer;
    Image image(settings.width, settings.height);
    Bvh bvh;
    bvh.build(scene.triangles);

    for (int y = 0; y < settings.height; ++y) {
        for (int x = 0; x < settings.width; ++x) {
            const double u = (static_cast<double>(x) + 0.5) / static_cast<double>(settings.width);
            const double v = 1.0 - (static_cast<double>(y) + 0.5) / static_cast<double>(settings.height);
            const Ray ray = camera.generate_ray(u, v);
            image.set_pixel(x, y, trace_ray(ray, scene, bvh, settings.max_depth, settings));
        }
    }

    return RenderResult{image, timer.elapsed_seconds()};
}

Color RayTracerRenderer::trace_ray(
    const Ray& ray,
    const Scene& scene,
    const Bvh& bvh,
    int depth,
    const RenderSettings& settings) const {
    if (depth <= 0) {
        return black();
    }

    HitRecord hit;
    if (!hit_scene(ray, scene, bvh, 0.001, 1.0e30, hit)) {
        return background_color(scene, settings);
    }

    if (!material_exists(scene, hit.material_id)) {
        return Color(1.0, 0.0, 1.0);
    }

    const Material& material = scene.materials[hit.material_id];
    Color result = material.emission;

    if (material.type == MaterialType::Emissive) {
        return result + material.base_color * 0.02;
    }

    // 一点环境项让没有显式灯光的教学场景仍然可见；真实路径追踪会由间接光积分处理这部分。
    result += multiply(material.base_color, background_color(scene, settings)) * 0.25;

    for (const PointLight& light : scene.point_lights) {
        const Vec3 to_light = light.position - hit.position;
        const double distance_squared = length_squared(to_light);
        if (distance_squared <= 1e-12) {
            continue;
        }
        const double distance = std::sqrt(distance_squared);
        const Vec3 light_dir = to_light / distance;

        HitRecord shadow_hit;
        const Ray shadow_ray(hit.position, light_dir);
        if (hit_scene(shadow_ray, scene, bvh, 0.001, distance - 0.001, shadow_hit)) {
            continue;
        }

        const double n_dot_l = std::max(0.0, dot(hit.normal, light_dir));
        result += multiply(material.base_color, light.intensity) * (n_dot_l / distance_squared);
    }

    for (const DirectionalLight& light : scene.directional_lights) {
        const Vec3 light_dir = normalize(-light.direction);
        HitRecord shadow_hit;
        const Ray shadow_ray(hit.position, light_dir);
        if (hit_scene(shadow_ray, scene, bvh, 0.001, 1.0e30, shadow_hit)) {
            continue;
        }

        const double n_dot_l = std::max(0.0, dot(hit.normal, light_dir));
        result += multiply(material.base_color, light.radiance) * n_dot_l;
    }

    if (material.type == MaterialType::Metal) {
        const Vec3 reflected = reflect(normalize(ray.direction), hit.normal);
        const Color reflected_color = trace_ray(Ray(hit.position, reflected), scene, bvh, depth - 1, settings);
        result += multiply(material.base_color, reflected_color) * 0.8;
    } else if (material.type == MaterialType::Dielectric) {
        const double eta_ratio = hit.front_face ? (1.0 / material.ior) : material.ior;
        const Vec3 unit_direction = normalize(ray.direction);
        const double cos_theta = std::min(dot(-unit_direction, hit.normal), 1.0);

        Vec3 refracted;
        const Vec3 reflected = reflect(unit_direction, hit.normal);
        const Color reflected_color = trace_ray(Ray(hit.position, reflected), scene, bvh, depth - 1, settings);
        if (refract(unit_direction, hit.normal, eta_ratio, refracted)) {
            const Color refracted_color = trace_ray(Ray(hit.position, refracted), scene, bvh, depth - 1, settings);
            const double reflect_weight = reflectance(cos_theta, eta_ratio);
            result += reflected_color * reflect_weight + refracted_color * (1.0 - reflect_weight);
        } else {
            result += reflected_color;
        }
    }

    return result;
}

bool RayTracerRenderer::hit_scene(
    const Ray& ray,
    const Scene& scene,
    const Bvh& bvh,
    double t_min,
    double t_max,
    HitRecord& hit) const {
    bool hit_anything = false;
    double closest_t = t_max;

    for (const Sphere& sphere : scene.spheres) {
        HitRecord candidate;
        if (sphere.intersect(ray, t_min, closest_t, candidate)) {
            hit_anything = true;
            closest_t = candidate.t;
            hit = candidate;
        }
    }

    HitRecord triangle_hit;
    if (bvh.intersect(ray, t_min, closest_t, triangle_hit)) {
        hit_anything = true;
        hit = triangle_hit;
    }

    return hit_anything;
}

}  // namespace renderer
