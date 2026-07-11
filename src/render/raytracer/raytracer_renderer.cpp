#include "render/raytracer/raytracer_renderer.h"

#include "core/timer.h"
#include "render/scene_intersector.h"
#include "sampling/sampler.h"
#include "scene/material_evaluator.h"

#include <algorithm>
#include <cmath>

namespace renderer {

namespace {

Color multiply(const Color& a, const Color& b) {
    return Color(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
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
    const SceneIntersector intersector(scene);

    for (int y = 0; y < settings.height; ++y) {
        for (int x = 0; x < settings.width; ++x) {
            const double u = (static_cast<double>(x) + 0.5) / static_cast<double>(settings.width);
            const double v = 1.0 - (static_cast<double>(y) + 0.5) / static_cast<double>(settings.height);
            // Primary ray：每个像素从相机出发打一条主光线，找到屏幕上能看到的第一个表面。
            const Ray ray = camera.generate_ray(u, v);
            image.set_pixel(x, y, trace_ray(ray, scene, intersector, settings.max_depth, settings));
        }
    }

    return RenderResult{image, timer.elapsed_seconds()};
}

Color RayTracerRenderer::trace_ray(
    const Ray& ray,
    const Scene& scene,
    const SceneIntersector& intersector,
    int depth,
    const RenderSettings& settings) const {
    // 递归深度限制防止镜面互相反射或玻璃反复折射时无限追踪。
    if (depth <= 0) {
        return black();
    }

    HitRecord hit;
    if (!intersector.intersect(ray, 0.0, 1.0e30, hit)) {
        return background_color(scene, settings);
    }

    if (!material_exists(scene, hit.material_id)) {
        return Color(1.0, 0.0, 1.0);
    }

    const Material& material = scene.materials[hit.material_id];
    const SurfaceMaterialSample surface = evaluate_surface_material(scene, material, hit);
    const Color base_color = surface.base_color;
    const Vec3 shading_normal = surface.shading_normal;
    Color result = material.emission;

    if (material.type == MaterialType::Emissive) {
        return result + base_color * 0.02f;
    }

    // 一点环境项让没有显式灯光的教学场景仍然可见；真实路径追踪会由间接光积分处理这部分。
    result += multiply(base_color, background_color(scene, settings)) * 0.25f;

    for (const PointLight& light : scene.point_lights) {
        const Vec3 to_light = light.position - hit.position;
        const float distance_squared = length_squared(to_light);
        if (distance_squared <= 1e-12) {
            continue;
        }
        const float distance = std::sqrt(distance_squared);
        const Vec3 light_dir = to_light / distance;

        // Shadow ray：从命中点朝灯光打一条检测光线，若中途碰到物体，这个灯就被遮挡。
        const Ray shadow_ray(
            offset_ray_origin(hit.position, hit.geometric_normal, light_dir),
            light_dir);
        if (intersector.occluded(shadow_ray, 0.0, distance - 1e-7)) {
            continue;
        }

        const float n_dot_l = std::max(0.0f, dot(shading_normal, light_dir));
        result += multiply(base_color, light.intensity) * (n_dot_l / distance_squared);
    }

    for (const DirectionalLight& light : scene.directional_lights) {
        const Vec3 light_dir = normalize(-light.direction);
        // 方向光没有距离衰减，shadow ray 只需要确认沿光源方向是否有任意遮挡物。
        const Ray shadow_ray(
            offset_ray_origin(hit.position, hit.geometric_normal, light_dir),
            light_dir);
        if (intersector.occluded(shadow_ray, 0.0, 1.0e30)) {
            continue;
        }

        const float n_dot_l = std::max(0.0f, dot(shading_normal, light_dir));
        result += multiply(base_color, light.radiance) * n_dot_l;
    }

    if (material.type == MaterialType::Metal) {
        // Reflection ray：镜面材质沿法线反射入射方向，再递归查询反射方向看到的颜色。
        const Vec3 reflected = reflect(normalize(ray.direction), shading_normal);
        const Color reflected_color = trace_ray(
            Ray(offset_ray_origin(hit.position, hit.geometric_normal, reflected), reflected),
            scene,
            intersector,
            depth - 1,
            settings);
        result += multiply(base_color, reflected_color) * 0.8f;
    } else if (material.type == MaterialType::Dielectric) {
        // Refraction ray：玻璃材质根据相对折射率弯折光线，并用 Schlick 近似混合反射/折射。
        const double eta_ratio = hit.front_face ? (1.0 / material.ior) : material.ior;
        const Vec3 unit_direction = normalize(ray.direction);
        const double cos_theta = std::min(
            static_cast<double>(dot(-unit_direction, shading_normal)), 1.0);

        Vec3 refracted;
        const Vec3 reflected = reflect(unit_direction, shading_normal);
        const Color reflected_color = trace_ray(
            Ray(offset_ray_origin(hit.position, hit.geometric_normal, reflected), reflected),
            scene,
            intersector,
            depth - 1,
            settings);
        if (refract(unit_direction, shading_normal, eta_ratio, refracted)) {
            const Color refracted_color = trace_ray(
                Ray(offset_ray_origin(hit.position, hit.geometric_normal, refracted), refracted),
                scene,
                intersector,
                depth - 1,
                settings);
            const double reflect_weight = reflectance(cos_theta, eta_ratio);
            result += reflected_color * static_cast<float>(reflect_weight) +
                refracted_color * static_cast<float>(1.0 - reflect_weight);
        } else {
            result += reflected_color;
        }
    }

    return result;
}
}  // namespace renderer
