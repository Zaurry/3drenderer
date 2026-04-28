#include "render/pathtracer/pathtracer_renderer.h"

#include "core/timer.h"
#include "sampling/sampler.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace renderer {

namespace {

Color black() {
    return Color(0.0, 0.0, 0.0);
}

Color multiply(const Color& a, const Color& b) {
    return Color(a.x * b.x, a.y * b.y, a.z * b.z);
}

bool material_exists(const Scene& scene, int material_id) {
    return material_id >= 0 && static_cast<std::size_t>(material_id) < scene.materials.size();
}

double reflectance(double cosine, double refraction_index) {
    double r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 *= r0;
    return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
}

Vec3 tangent_to_world(const Vec3& local_direction, const Vec3& normal) {
    const Vec3 w = normalize(normal);
    const Vec3 helper = std::abs(w.x) > 0.9 ? Vec3(0.0, 1.0, 0.0) : Vec3(1.0, 0.0, 0.0);
    const Vec3 v = normalize(cross(w, helper));
    const Vec3 u = cross(v, w);
    return normalize(local_direction.x * u + local_direction.y * v + local_direction.z * w);
}

std::uint64_t pixel_seed(int x, int y, int width) {
    // The seed is deterministic per pixel so test renders are reproducible even
    // when tiles are processed by different worker threads.
    std::uint64_t seed = 1469598103934665603ULL;
    seed ^= static_cast<std::uint64_t>(x + 1);
    seed *= 1099511628211ULL;
    seed ^= static_cast<std::uint64_t>((y + 1) * 65537);
    seed *= 1099511628211ULL;
    seed ^= static_cast<std::uint64_t>(width + 1);
    return seed;
}

int worker_count_for(const RenderSettings& settings, int total_tiles) {
    if (total_tiles <= 0) {
        return 1;
    }

    int worker_count = settings.thread_count;
    if (worker_count <= 0) {
        worker_count = static_cast<int>(std::thread::hardware_concurrency());
    }
    worker_count = std::max(1, worker_count);
    return std::min(worker_count, total_tiles);
}

}  // namespace

RenderResult PathTracerRenderer::render(const Scene& scene, const Camera& camera, const RenderSettings& settings) {
    Timer timer;
    Image image(settings.width, settings.height);
    Bvh bvh;
    bvh.build(scene.triangles);

    const int samples_per_pixel = std::max(1, settings.samples_per_pixel);
    const int max_depth = std::max(0, settings.max_depth);
    const int tile_size = std::max(1, settings.tile_size);
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
                    PcgRandom rng(pixel_seed(x, y, settings.width));
                    Color accumulated = black();

                    for (int sample = 0; sample < samples_per_pixel; ++sample) {
                        const double u =
                            (static_cast<double>(x) + rng.next_double()) / static_cast<double>(settings.width);
                        const double v =
                            1.0 -
                            (static_cast<double>(y) + rng.next_double()) / static_cast<double>(settings.height);
                        accumulated += trace_path(camera.generate_ray(u, v), scene, bvh, rng, max_depth);
                    }

                    image.set_pixel(x, y, accumulated / static_cast<double>(samples_per_pixel));
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

    return RenderResult{image, timer.elapsed_seconds()};
}

Color PathTracerRenderer::trace_path(const Ray& ray, const Scene& scene, const Bvh& bvh, PcgRandom& rng, int depth) const {
    if (depth <= 0) {
        return black();
    }

    HitRecord hit;
    if (!hit_scene(ray, scene, bvh, 0.001, 1.0e30, hit)) {
        return scene.environment;
    }

    if (!material_exists(scene, hit.material_id)) {
        return Color(1.0, 0.0, 1.0);
    }

    const Material& material = scene.materials[hit.material_id];
    const Color emitted = material.emission;
    if (material.type == MaterialType::Emissive) {
        return emitted;
    }

    Color attenuation;
    Ray scattered(hit.position, hit.normal);
    if (!scatter(ray, hit, material, rng, attenuation, scattered)) {
        return emitted;
    }

    // This is the recursive path-tracing estimator for the rendering equation:
    // emitted radiance at the hit point plus material throughput (attenuation)
    // multiplied by the incoming radiance sampled along one new bounce.
    return emitted + multiply(attenuation, trace_path(scattered, scene, bvh, rng, depth - 1));
}

bool PathTracerRenderer::scatter(
    const Ray& ray,
    const HitRecord& hit,
    const Material& material,
    PcgRandom& rng,
    Color& attenuation,
    Ray& scattered) const {
    if (material.type == MaterialType::Diffuse) {
        const Vec3 local_direction = cosine_weighted_hemisphere(rng);
        const Vec3 scatter_direction = tangent_to_world(local_direction, hit.normal);
        attenuation = material.base_color;
        scattered = Ray(hit.position, scatter_direction);
        return true;
    }

    if (material.type == MaterialType::Metal) {
        Vec3 scatter_direction = reflect(normalize(ray.direction), hit.normal);
        if (material.roughness > 0.0) {
            scatter_direction += std::max(0.0, material.roughness) * random_in_unit_sphere(rng);
        }
        scatter_direction = normalize(scatter_direction);
        if (dot(scatter_direction, hit.normal) <= 0.0) {
            return false;
        }

        attenuation = material.base_color;
        scattered = Ray(hit.position, scatter_direction);
        return true;
    }

    if (material.type == MaterialType::Dielectric) {
        const double refraction_ratio = hit.front_face ? (1.0 / material.ior) : material.ior;
        const Vec3 unit_direction = normalize(ray.direction);
        const double cos_theta = std::min(dot(-unit_direction, hit.normal), 1.0);

        Vec3 refracted;
        const bool can_refract = refract(unit_direction, hit.normal, refraction_ratio, refracted);
        const bool choose_reflection =
            !can_refract || reflectance(cos_theta, refraction_ratio) > rng.next_double();
        const Vec3 scatter_direction = choose_reflection
            ? reflect(unit_direction, hit.normal)
            : refracted;

        attenuation = Color(1.0, 1.0, 1.0);
        scattered = Ray(hit.position, normalize(scatter_direction));
        return true;
    }

    return false;
}

bool PathTracerRenderer::hit_scene(
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
