#include "render/pathtracer/pathtracer_renderer.h"

#include "core/timer.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/pathtracer/path_backend.h"
#include "render/pbr.h"
#include "render/scene_intersector.h"
#include "sampling/sampler.h"
#include "scene/material_evaluator.h"
#include "scene/environment.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace renderer {

namespace {

constexpr int kRussianRouletteStartBounce = 3;
constexpr int kMaxPathBounces = 64;
constexpr float kMinContinuationProbability = 0.05f;
constexpr float kMaxContinuationProbability = 0.95f;

Color black() {
    return Color::Zero();
}

bool material_exists(const Scene& scene, int material_id) {
    return material_id >= 0 && static_cast<std::size_t>(material_id) < scene.materials.size();
}

float reflectance(float cosine, float refraction_index) {
    float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 *= r0;
    return r0 + (1.0f - r0) * std::pow(1.0f - cosine, 5.0f);
}

float punctual_range_attenuation(float distance, float range) {
    if (!(range > 0.0f)) {
        return 1.0f;
    }
    const float ratio = distance / range;
    const float cutoff = std::max(0.0f, 1.0f - ratio * ratio * ratio * ratio);
    return cutoff * cutoff;
}

std::uint64_t pixel_seed(int x, int y, int width, std::uint64_t sample_seed_offset) {
    // The seed is deterministic per pixel so test renders are reproducible even
    // when tiles are processed by different worker threads.
    std::uint64_t seed = 1469598103934665603ULL;
    seed ^= static_cast<std::uint64_t>(x + 1);
    seed *= 1099511628211ULL;
    seed ^= static_cast<std::uint64_t>((y + 1) * 65537);
    seed *= 1099511628211ULL;
    seed ^= static_cast<std::uint64_t>(width + 1);
    if (sample_seed_offset != 0) {
        seed *= 1099511628211ULL;
        seed ^= sample_seed_offset;
    }
    return seed;
}

int worker_count_for(const RenderSettings& settings, int total_tiles) {
    if (total_tiles <= 0) {
        return 1;
    }

    int worker_count = settings.path.thread_count;
    if (worker_count <= 0) {
        worker_count = static_cast<int>(std::thread::hardware_concurrency());
    }
    worker_count = std::max(1, worker_count);
    return std::min(worker_count, total_tiles);
}

}  // namespace

RenderResult PathTracerRenderer::render(const Scene& scene, const Camera& camera, const RenderSettings& settings) {
    if (resolve_path_backend(settings.path.backend) == ExecutionBackend::Cuda) {
        return render_cuda_path(scene, camera, settings);
    }
    return render_cpu(scene, camera, settings);
}

RenderResult PathTracerRenderer::render_cpu(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings) {
    Timer timer;
    Image image(settings.width, settings.height);
    const SceneIntersector intersector(scene);

    const int samples_per_pixel = std::max(1, settings.path.samples_per_pixel);
    const int tile_size = std::max(1, settings.path.tile_size);
    const int tiles_x = (settings.width + tile_size - 1) / tile_size;
    const int tiles_y = (settings.height + tile_size - 1) / tile_size;
    const int total_tiles = tiles_x * tiles_y;
    const int worker_count = worker_count_for(settings, total_tiles);

    std::atomic<int> next_tile{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count));

    auto render_worker = [&]() {
        while (true) {
            const int tile_index = next_tile.fetch_add(1);
            if (tile_index >= total_tiles) {
                break;
            }

            const int tile_x = tile_index % tiles_x;
            const int tile_y = tile_index / tiles_x;
            const int start_x = tile_x * tile_size;
            const int start_y = tile_y * tile_size;
            const int end_x = std::min(start_x + tile_size, settings.width);
            const int end_y = std::min(start_y + tile_size, settings.height);

            for (int y = start_y; y < end_y; ++y) {
                for (int x = start_x; x < end_x; ++x) {
                    PcgRandom rng(pixel_seed(
                        x,
                        y,
                        settings.width,
                        settings.path.sample_seed_offset));
                    Color accumulated = black();

                    for (int sample = 0; sample < samples_per_pixel; ++sample) {
                        const float u =
                            (static_cast<float>(x) + rng.next_float()) / static_cast<float>(settings.width);
                        const float v =
                            1.0f -
                            (static_cast<float>(y) + rng.next_float()) / static_cast<float>(settings.height);
                        accumulated += trace_path(
                            camera.generate_ray(u, v),
                            scene,
                            intersector,
                            rng);
                    }

                    image.set_pixel(x, y, accumulated / static_cast<float>(samples_per_pixel));
                }
            }
        }
    };

    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back(render_worker);
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    return RenderResult{image, timer.elapsed_seconds(), ExecutionBackend::Cpu};
}

Color PathTracerRenderer::trace_path(
    const Ray& ray,
    const Scene& scene,
    const SceneIntersector& intersector,
    PcgRandom& rng) const {
    Color radiance = black();
    Color throughput = Color::Ones();
    Ray current_ray = ray;
    float previous_bsdf_pdf = 0.0f;
    bool previous_delta = true;

    for (int bounce = 0; bounce < kMaxPathBounces; ++bounce) {
        HitRecord hit;
        if (!intersector.intersect(current_ray, 0.0f, 1.0e30f, hit)) {
            if (bounce > 0 || scene.environment_background_visible) {
                const Color incoming = environment_radiance(
                    scene.environment,
                    scene.environment_map,
                    scene.environment_intensity,
                    scene.environment_rotation_degrees,
                    current_ray.direction);
                const float weight = bounce == 0 || previous_delta
                    ? 1.0f
                    : power_heuristic(
                          previous_bsdf_pdf,
                          environment_pdf(
                              scene.environment_map,
                              scene.environment_rotation_degrees,
                              current_ray.direction));
                radiance += throughput.cwiseProduct(incoming) * weight;
            }
            break;
        }

        if (!material_exists(scene, hit.material_id)) {
            radiance += throughput.cwiseProduct(Color(1.0f, 0.0f, 1.0f));
            break;
        }

        const Material& material = scene.materials[hit.material_id];
        const SurfaceMaterialSample surface = evaluate_surface_material(scene, material, hit);
        if (material.type == MaterialType::Pbr &&
            material.alpha_mode == AlphaMode::Blend &&
            rng.next_float() >= surface.opacity) {
            const Vec3 direction = current_ray.direction.normalized();
            current_ray = Ray(
                offset_ray_origin(hit.position, hit.geometric_normal, direction),
                direction);
            previous_bsdf_pdf = 0.0f;
            previous_delta = true;
            continue;
        }
        const Color emitted = surface.emission;
        if (material.type == MaterialType::Emissive) {
            radiance += throughput.cwiseProduct(emitted);
            break;
        }

        Color attenuation;
        Ray scattered(hit.position, surface.shading_normal);
        float bsdf_pdf = 0.0f;
        bool delta = false;
        const bool scattered_valid = scatter(
                current_ray,
                hit,
                material,
                surface,
                rng,
                attenuation,
                scattered,
                bsdf_pdf,
                delta);

        Color direct = black();
        if (material.type != MaterialType::Dielectric) {
            direct += estimate_direct_lighting(
                scene,
                intersector,
                hit,
                surface,
                -current_ray.direction.normalized());
            direct += estimate_environment_lighting(
                scene,
                intersector,
                hit,
                surface,
                -current_ray.direction.normalized(),
                rng);
        }
        radiance += throughput.cwiseProduct(emitted + direct);
        if (!scattered_valid) {
            break;
        }

        throughput = throughput.cwiseProduct(attenuation);
        previous_bsdf_pdf = bsdf_pdf;
        previous_delta = delta;

        if (!throughput.allFinite() || throughput.maxCoeff() <= 0.0f) {
            break;
        }
        if (bounce + 1 >= kMaxPathBounces) {
            break;
        }

        // Short paths are always traced. Afterwards, low-throughput paths are
        // probabilistically terminated. Dividing surviving paths by their
        // continuation probability keeps the Monte Carlo estimator unbiased.
        if (bounce + 1 >= kRussianRouletteStartBounce) {
            const float continuation_probability = std::clamp(
                throughput.maxCoeff(),
                kMinContinuationProbability,
                kMaxContinuationProbability);
            if (rng.next_float() >= continuation_probability) {
                break;
            }
            throughput /= continuation_probability;
        }

        current_ray = scattered;
    }

    return radiance;
}

bool PathTracerRenderer::scatter(
    const Ray& ray,
    const HitRecord& hit,
    const Material& material,
    const SurfaceMaterialSample& surface,
    PcgRandom& rng,
    Color& attenuation,
    Ray& scattered,
    float& pdf,
    bool& delta) const {
    const Color base_color = surface.base_color;
    const Vec3 shading_normal = surface.shading_normal;
    if (material.type == MaterialType::Dielectric) {
        const float refraction_ratio = hit.front_face ? (1.0f / material.ior) : material.ior;
        const Vec3 unit_direction = ray.direction.normalized();
        const float cos_theta = std::min((-unit_direction).dot(shading_normal), 1.0f);

        Vec3 refracted;
        const bool can_refract = refract(
            unit_direction,
            shading_normal,
            refraction_ratio,
            refracted);
        const bool choose_reflection =
            !can_refract || reflectance(cos_theta, refraction_ratio) > rng.next_float();
        const Vec3 scatter_direction = choose_reflection
            ? reflect(unit_direction, shading_normal)
            : refracted;

        attenuation = Color::Ones();
        const Vec3 unit_scatter = scatter_direction.normalized();
        scattered = Ray(
            offset_ray_origin(hit.position, hit.geometric_normal, unit_scatter),
            unit_scatter);
        pdf = 1.0f;
        delta = true;
        return true;
    }

    if (material.type == MaterialType::Diffuse ||
        material.type == MaterialType::Metal ||
        material.type == MaterialType::Pbr) {
        const PbrSample sampled = sample_pbr(
            PbrSurface{
                surface.diffuse_color,
                surface.specular_f0,
                surface.specular_f90,
                surface.diffuse_fresnel_f0,
                surface.diffuse_fresnel_f90,
                surface.roughness,
                surface.diffuse_fresnel_uses_max},
            shading_normal,
            -ray.direction.normalized(),
            rng.next_float(),
            rng.next_float(),
            rng.next_float());
        if (!sampled.valid) {
            return false;
        }
        attenuation = sampled.weight;
        pdf = sampled.pdf;
        delta = false;
        scattered = Ray(
            offset_ray_origin(hit.position, hit.geometric_normal, sampled.direction),
            sampled.direction);
        return true;
    }

    return false;
}

Color PathTracerRenderer::estimate_direct_lighting(
    const Scene& scene,
    const SceneIntersector& intersector,
    const HitRecord& hit,
    const SurfaceMaterialSample& surface,
    const Vec3& outgoing) const {
    Color direct = black();

    const auto evaluate_light = [&](const Vec3& light_dir, const Color& incoming) {
        const PbrEvaluation evaluated = evaluate_pbr(
            PbrSurface{
                surface.diffuse_color,
                surface.specular_f0,
                surface.specular_f90,
                surface.diffuse_fresnel_f0,
                surface.diffuse_fresnel_f90,
                surface.roughness,
                surface.diffuse_fresnel_uses_max},
            surface.shading_normal,
            outgoing,
            light_dir);
        return evaluated.brdf.cwiseProduct(incoming) *
            std::max(0.0f, surface.shading_normal.dot(light_dir));
    };

    for (const DirectionalLight& light : scene.directional_lights) {
        if (!usable_direction(light.direction)) {
            continue;
        }
        const Vec3 light_dir = (-light.direction).normalized();
        if (!usable_direction(light_dir)) {
            continue;
        }
        const float n_dot_l = std::max(0.0f, surface.shading_normal.dot(light_dir));
        if (n_dot_l <= 0.0f) {
            continue;
        }
        const Ray shadow_ray(
            offset_ray_origin(hit.position, hit.geometric_normal, light_dir),
            light_dir);
        if (intersector.occluded(shadow_ray, 0.0f, 1.0e30f)) {
            continue;
        }
        direct += evaluate_light(light_dir, light.radiance);
    }

    for (const PointLight& light : scene.point_lights) {
        const Vec3 to_light = light.position - hit.position;
        const float distance_squared = to_light.squaredNorm();
        if (distance_squared <= 1e-12f) {
            continue;
        }
        const float distance = std::sqrt(distance_squared);
        const float range_attenuation = punctual_range_attenuation(distance, light.range);
        if (range_attenuation <= 0.0f) {
            continue;
        }
        const Vec3 light_dir = to_light / distance;
        const float n_dot_l = std::max(0.0f, surface.shading_normal.dot(light_dir));
        if (n_dot_l <= 0.0f) {
            continue;
        }
        const Ray shadow_ray(
            offset_ray_origin(hit.position, hit.geometric_normal, light_dir),
            light_dir);
        if (intersector.occluded(shadow_ray, 0.0f, distance - 1e-7f)) {
            continue;
        }
        const Color incoming = light.intensity * (range_attenuation / distance_squared);
        direct += evaluate_light(light_dir, incoming);
    }

    for (const SpotLight& light : scene.spot_lights) {
        const Vec3 to_light = light.position - hit.position;
        const float distance_squared = to_light.squaredNorm();
        if (distance_squared <= 1.0e-12f) {
            continue;
        }
        const float distance = std::sqrt(distance_squared);
        const float range_attenuation = punctual_range_attenuation(distance, light.range);
        if (range_attenuation <= 0.0f) {
            continue;
        }
        const Vec3 light_dir = to_light / distance;
        const float cone_cosine = (-light_dir).dot(light.direction.normalized());
        const float inner = std::cos(light.inner_cone_radians);
        const float outer = std::cos(light.outer_cone_radians);
        const float cone = inner <= outer
            ? (cone_cosine >= outer ? 1.0f : 0.0f)
            : std::clamp((cone_cosine - outer) / (inner - outer), 0.0f, 1.0f);
        if (cone <= 0.0f || surface.shading_normal.dot(light_dir) <= 0.0f) {
            continue;
        }
        const Ray shadow_ray(
            offset_ray_origin(hit.position, hit.geometric_normal, light_dir),
            light_dir);
        if (intersector.occluded(shadow_ray, 0.0f, distance - 1.0e-7f)) {
            continue;
        }
        direct += evaluate_light(
            light_dir,
            light.intensity * (cone * range_attenuation / distance_squared));
    }

    return direct;
}

Color PathTracerRenderer::estimate_environment_lighting(
    const Scene& scene,
    const SceneIntersector& intersector,
    const HitRecord& hit,
    const SurfaceMaterialSample& surface,
    const Vec3& outgoing,
    PcgRandom& rng) const {
    if (scene.environment_intensity <= 0.0f ||
        (!scene.environment_map && scene.environment.maxCoeff() <= 0.0f)) {
        return Color::Zero();
    }
    const float select = rng.next_float();
    const float jitter_u = rng.next_float();
    const float jitter_v = scene.environment_map
        ? rng.next_float()
        : 0.0f;
    const EnvironmentMapSample light = sample_environment(
        scene.environment,
        scene.environment_map,
        scene.environment_intensity,
        scene.environment_rotation_degrees,
        select,
        jitter_u,
        jitter_v);
    const float cosine = surface.shading_normal.dot(light.direction);
    if (!(light.pdf > 0.0f) || cosine <= 0.0f || light.radiance.maxCoeff() <= 0.0f) {
        return Color::Zero();
    }
    const Ray shadow_ray(
        offset_ray_origin(hit.position, hit.geometric_normal, light.direction),
        light.direction);
    if (intersector.occluded(shadow_ray, 0.0f, 1.0e30f)) {
        return Color::Zero();
    }
    const PbrEvaluation evaluated = evaluate_pbr(
        PbrSurface{
            surface.diffuse_color,
            surface.specular_f0,
            surface.specular_f90,
            surface.diffuse_fresnel_f0,
            surface.diffuse_fresnel_f90,
            surface.roughness,
            surface.diffuse_fresnel_uses_max},
        surface.shading_normal,
        outgoing,
        light.direction);
    const float weight = power_heuristic(light.pdf, evaluated.pdf);
    return evaluated.brdf.cwiseProduct(light.radiance) *
        (cosine * weight * surface.occlusion / light.pdf);
}
}  // namespace renderer
