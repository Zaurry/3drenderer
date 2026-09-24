// cuda_scene establishes EIGEN_NO_CUDA before any public Eigen headers are read.
// clang-format off
#include "render/pathtracer/cuda_scene.cuh"
#include "render/ddgi/cuda_ddgi_volume.h"
// clang-format on
#include <deque>

namespace renderer {
namespace {
struct DdgiProbe {
    DVec3 offset{};
    int state = 0; // -1 inside geometry, 0 uninitialized, 1 valid
    float time = 0;
    unsigned int frame = 0, updates = 0;
    unsigned int geometry_epoch = 0;
    int relocated = 0;
};
struct DdgiRay {
    DVec3 direction, radiance;
    float distance;
    int back;
};
struct DdgiRect {
    DVec3 position, u, v, radiance;
    int two_sided, shadows;
};
template <class T> void zero_buffer(DeviceBuffer<T> &buffer, cudaStream_t stream) {
    check_cuda(cudaMemsetAsync(buffer.get(), 0, buffer.size() * sizeof(T), stream), "DDGI clear");
}
struct DdgiGpu {
    DScene scene;
    DVec3 origin, spacing;
    int nx, ny, nz, columns, rows, count;
    int ray_count, update_count;
    unsigned int frame, geometry_epoch;
    float time, hysteresis, normal_bias, view_bias, far_distance;
    int fast, relocation, classification, shadows;
    const int *indices;
    const DdgiRect *rectangles;
    int rectangle_count;
    const DdgiProbe *previous;
    DdgiProbe *next;
    const float4 *irradiance;
    const float4 *distance;
    DdgiRay *rays;
    int *error;
};

__device__ DVec3 probe_position(DdgiGpu f, int index, const DdgiProbe &p) {
    return add(add(f.origin,
                   v3(f.spacing.x * float(index % f.nx), f.spacing.y * float((index / f.nx) % f.ny),
                      f.spacing.z * float(index / (f.nx * f.ny)))),
               p.offset);
}
__device__ float sign_nonzero(float x) {
    return x >= 0 ? 1.0f : -1.0f;
}
__device__ DVec2 oct_encode(DVec3 d) {
    d = divv(d, fmaxf(fabsf(d.x) + fabsf(d.y) + fabsf(d.z), 1e-20f));
    if (d.z < 0)
        return {(1 - fabsf(d.y)) * sign_nonzero(d.x), (1 - fabsf(d.x)) * sign_nonzero(d.y)};
    return {d.x, d.y};
}
__device__ DVec3 oct_decode(float x, float y) {
    DVec3 d = v3(x, y, 1 - fabsf(x) - fabsf(y));
    if (d.z < 0) {
        const float old = d.x;
        d.x = (1 - fabsf(d.y)) * sign_nonzero(old);
        d.y = (1 - fabsf(old)) * sign_nonzero(d.y);
    }
    return normalize(d);
}
__device__ float4 atlas_sample(const float4 *data, DdgiGpu f, int probe, int n, DVec3 d) {
    const DVec2 oct = oct_encode(d);
    const float x = (oct.x * .5f + .5f) * n + .5f;
    const float y = (oct.y * .5f + .5f) * n + .5f;
    const int ix = int(floorf(x)), iy = int(floorf(y));
    const int pitch = f.columns * (n + 2);
    const int base = (probe / f.columns * (n + 2) + iy) * pitch + probe % f.columns * (n + 2) + ix;
    const float fx = x - ix, fy = y - iy;
    const float4 a = data[base], b = data[base + 1], c = data[base + pitch],
                 e = data[base + pitch + 1];
    return make_float4((a.x * (1 - fx) + b.x * fx) * (1 - fy) + (c.x * (1 - fx) + e.x * fx) * fy,
                       (a.y * (1 - fx) + b.y * fx) * (1 - fy) + (c.y * (1 - fx) + e.y * fx) * fy,
                       (a.z * (1 - fx) + b.z * fx) * (1 - fy) + (c.z * (1 - fx) + e.z * fx) * fy,
                       0);
}
__device__ DVec3 probe_irradiance(DdgiGpu f, DVec3 position, DVec3 normal, DVec3 shading_normal,
                                  DVec3 outgoing) {
    const float spacing = fminf(f.spacing.x, fminf(f.spacing.y, f.spacing.z));
    const DVec3 biased =
        add(position, mul(add(mul(normal, f.normal_bias), mul(outgoing, f.view_bias)), spacing));
    const DVec3 grid =
        v3((biased.x - f.origin.x) / f.spacing.x, (biased.y - f.origin.y) / f.spacing.y,
           (biased.z - f.origin.z) / f.spacing.z);
    if (grid.x < 0 || grid.y < 0 || grid.z < 0 || grid.x > f.nx - 1 || grid.y > f.ny - 1 ||
        grid.z > f.nz - 1)
        return v3(0, 0, 0);
    const int bx = min(int(grid.x), f.nx - 2), by = min(int(grid.y), f.ny - 2),
              bz = min(int(grid.z), f.nz - 2);
    const DVec3 fraction = sub(grid, v3(float(bx), float(by), float(bz)));
    DVec3 sum = v3(0, 0, 0);
    float total = 0;
    for (int corner = 0; corner < 8; ++corner) {
        const int x = corner & 1, y = (corner >> 1) & 1, z = (corner >> 2) & 1;
        const int index = bx + x + f.nx * (by + y + f.ny * (bz + z));
        const DdgiProbe p = f.previous[index];
        if (p.state != 1)
            continue;
        const DVec3 delta = sub(biased, probe_position(f, index, p));
        const float distance = sqrtf(fmaxf(length_squared(delta), 1e-20f));
        const DVec3 direction = divv(delta, distance);
        const float4 moments = atlas_sample(f.distance, f, index, 16, direction);
        const float variance = fmaxf(moments.y - moments.x * moments.x, spacing * spacing * 1e-6f);
        const float excess = fmaxf(distance - moments.x, 0.0f);
        float visibility = variance / (variance + excess * excess);
        visibility = visibility * visibility * visibility;
        // Geometric normals reject probes behind the receiving surface. Distance
        // moments alone are insufficient at grazing angles along thin walls.
        const float facing = distance < spacing * 1e-5f ? 1.f : fmaxf(0, -dot(normal, direction));
        float weight = (x ? fraction.x : 1 - fraction.x) * (y ? fraction.y : 1 - fraction.y) *
                       (z ? fraction.z : 1 - fraction.z);
        weight *= facing * facing * visibility;
        if (visibility < .2f)
            weight *= visibility * visibility / (.2f * .2f);
        const float4 e = atlas_sample(f.irradiance, f, index, 8, shading_normal);
        sum = add(sum, mul(v3(e.x, e.y, e.z), weight));
        total += weight;
    }
    return total > 1e-8f ? divv(sum, total) : v3(0, 0, 0);
}
__device__ DVec3 fibonacci_direction(int index, int count) {
    const float z = 1 - 2 * (float(index) + .5f) / float(count);
    const float r = sqrtf(fmaxf(0, 1 - z * z)), phi = float(index) * 2.399963229728653f;
    return v3(r * cosf(phi), r * sinf(phi), z);
}
__device__ DVec3 rotated_direction(int index, int count, int probe, unsigned int frame) {
    DPcgState rng;
    pcg_seed(rng, (static_cast<unsigned long long>(frame) << 32) ^ unsigned(probe + 1));
    const float u = random_float(rng), v = random_float(rng) * 2 * kPi,
                w = random_float(rng) * 2 * kPi;
    const DVec3 q = v3(sqrtf(1 - u) * sinf(v), sqrtf(1 - u) * cosf(v), sqrtf(u) * sinf(w));
    const float qw = sqrtf(u) * cosf(w);
    const DVec3 d = fibonacci_direction(index, count);
    return add(d, add(mul(cross(q, d), 2 * qw), mul(cross(q, cross(q, d)), 2)));
}
__device__ bool probe_hit(DdgiGpu f, DVec3 position, DVec3 dir, DHit &hit) {
    DCompactHit compact;
    DScene scene = f.scene;
    scene.probe_backfaces = 1;
    const DRay ray{position, dir};
    if (!intersect_scene_compact(scene, ray, 1e-5f, f.far_distance, compact, f.error))
        return false;
    reconstruct_hit(scene, ray, compact, true, hit);
    return true;
}
__device__ bool probe_visible(DdgiGpu f, const DHit &hit, DVec3 dir, float distance, int shadows) {
    if (!f.shadows || !shadows)
        return true;
    DScene scene = f.scene;
    scene.probe_backfaces = 0;
    return !occluded_scene(scene, {offset_origin(hit.position, hit.geometric_normal, dir), dir},
                           1e-5f, distance - 1e-4f, f.error);
}
__device__ DVec3 diffuse_brdf(const DSurface &s, DVec3 outgoing, DVec3 incoming) {
    const DVec3 halfdir = add(outgoing, incoming);
    if (!usable(halfdir))
        return v3(0, 0, 0);
    const DVec3 fresnel = fresnel_schlick(saturate(dot(outgoing, normalize(halfdir))),
                                          s.diffuse_fresnel_f0, s.diffuse_fresnel_f90);
    const float m = 1 - max_component(fresnel);
    return mul(product(s.diffuse_color,
                       s.diffuse_fresnel_uses_max ? v3(m, m, m) : sub(v3(1, 1, 1), fresnel)),
               1 / kPi);
}
__device__ DVec3 probe_direct(DdgiGpu f, const DHit &hit, const DSurface &s, DVec3 outgoing,
                              DPcgState &rng) {
    DVec3 sum = v3(0, 0, 0);
    for (int k = 0; k < f.scene.directional_light_count; ++k) {
        const auto light = f.scene.directional_lights[k];
        if (!usable(light.direction))
            continue;
        const DVec3 dir = normalize(mul(light.direction, -1));
        const float cosine = fmaxf(0, dot(s.shading_normal, dir));
        if (cosine > 0 && probe_visible(f, hit, dir, 1e30f, light.casts_shadows))
            sum = add(sum, mul(product(diffuse_brdf(s, outgoing, dir), light.radiance), cosine));
    }
    for (int kind = 0; kind < 2; ++kind)
        for (int k = 0; k < (kind ? f.scene.spot_light_count : f.scene.point_light_count); ++k) {
            DVec3 position, intensity;
            float range;
            int shadows;
            if (kind) {
                auto l = f.scene.spot_lights[k];
                position = l.position;
                intensity = l.intensity;
                range = l.range;
                shadows = l.casts_shadows;
            } else {
                auto l = f.scene.point_lights[k];
                position = l.position;
                intensity = l.intensity;
                range = l.range;
                shadows = l.casts_shadows;
            }
            const DVec3 delta = sub(position, hit.position);
            const float d2 = length_squared(delta);
            if (d2 < 1e-10f)
                continue;
            const float distance = sqrtf(d2);
            const DVec3 dir = divv(delta, distance);
            float factor = punctual_range_attenuation(distance, range) / d2;
            if (kind) {
                auto l = f.scene.spot_lights[k];
                const float c = dot(mul(dir, -1), normalize(l.direction));
                factor *= l.inner_cosine <= l.outer_cosine
                              ? (c >= l.outer_cosine ? 1.f : 0.f)
                              : saturate((c - l.outer_cosine) / (l.inner_cosine - l.outer_cosine));
            }
            const float cosine = fmaxf(0, dot(s.shading_normal, dir));
            if (factor > 0 && cosine > 0 && probe_visible(f, hit, dir, distance, shadows))
                sum = add(sum,
                          mul(product(diffuse_brdf(s, outgoing, dir), intensity), factor * cosine));
        }
    // One unbiased area sample per analytic rectangle. No path-tracer MIS:
    // these direct emitter connections are excluded from the recursive field.
    for (int k = 0; k < f.rectangle_count; ++k) {
        const auto l = f.rectangles[k];
        const DVec3 crossuv = cross(l.v, l.u);
        const float area = 4 * sqrtf(length_squared(crossuv));
        if (area < 1e-12f)
            continue;
        const DVec3 point = add(l.position, add(mul(l.u, 2 * random_float(rng) - 1),
                                                mul(l.v, 2 * random_float(rng) - 1)));
        const DVec3 delta = sub(point, hit.position);
        const float d2 = length_squared(delta);
        if (d2 < 1e-10f)
            continue;
        const float distance = sqrtf(d2);
        const DVec3 dir = divv(delta, distance);
        float lc = dot(normalize(crossuv), mul(dir, -1));
        lc = l.two_sided ? fabsf(lc) : fmaxf(0, lc);
        const float cosine = fmaxf(0, dot(s.shading_normal, dir));
        if (lc > 0 && cosine > 0 && probe_visible(f, hit, dir, distance, l.shadows))
            sum = add(sum, mul(product(diffuse_brdf(s, outgoing, dir), l.radiance),
                               area * lc * cosine / d2));
    }
    return sum;
}
__device__ DVec3 residual_emission(DdgiGpu f, DVec3 position, DVec3 emission) {
    for (int i = 0; i < f.rectangle_count; ++i) {
        const auto l = f.rectangles[i];
        const DVec3 d = sub(position, l.position), n = cross(l.u, l.v);
        const float uu = dot(l.u, l.u), vv = dot(l.v, l.v), uv = dot(l.u, l.v),
                    det = uu * vv - uv * uv;
        if (det < 1e-12f || fabsf(dot(d, normalize(n))) > 1e-4f * f.far_distance)
            continue;
        const float a = (dot(d, l.u) * vv - dot(d, l.v) * uv) / det,
                    b = (dot(d, l.v) * uu - dot(d, l.u) * uv) / det;
        if (fabsf(a) <= 1.0001f && fabsf(b) <= 1.0001f)
            emission = v3(fmaxf(0, emission.x - l.radiance.x), fmaxf(0, emission.y - l.radiance.y),
                          fmaxf(0, emission.z - l.radiance.z));
    }
    return emission;
}
__global__ void trace_fixed(DdgiGpu f) {
    const int job = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (job >= f.update_count * 32)
        return;
    const int local = job / 32, index = f.indices[local];
    const DVec3 dir = fibonacci_direction(job % 32, 32);
    DHit hit{};
    DdgiRay result{dir, v3(0, 0, 0), f.far_distance, 0};
    if (probe_hit(f, probe_position(f, index, f.previous[index]), dir, hit)) {
        result.distance = hit.t;
        result.back = !hit.front_face &&
                      (hit.material_id < 0 || !f.scene.materials[hit.material_id].two_sided);
    }
    f.rays[local * (f.ray_count + 32) + job % 32] = result;
}
__global__ void relocate_probes(DdgiGpu f) {
    const int job = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (job >= f.update_count)
        return;
    const int index = f.indices[job];
    DdgiProbe p = f.previous[index];
    p.relocated = 0;
    int backs = 0;
    float nearest_back = f.far_distance, nearest_front = f.far_distance;
    DVec3 back_dir = v3(0, 0, 0), front_dir = v3(0, 0, 0);
    for (int i = 0; i < 32; ++i) {
        const auto r = f.rays[job * (f.ray_count + 32) + i];
        if (r.back) {
            ++backs;
            if (r.distance < nearest_back) {
                nearest_back = r.distance;
                back_dir = r.direction;
            }
        } else if (r.distance < nearest_front) {
            nearest_front = r.distance;
            front_dir = r.direction;
        }
    }
    const float cell = fminf(f.spacing.x, fminf(f.spacing.y, f.spacing.z));
    const bool inside = backs > 8;
    DVec3 movement = v3(0, 0, 0);
    if (f.relocation) {
        if (inside)
            movement = mul(back_dir, nearest_back + cell * .12f);
        else if (nearest_front < cell * .10f)
            movement = mul(front_dir, nearest_front - cell * .10f);
        DVec3 candidate = add(p.offset, movement);
        if (fabsf(candidate.x) <= .45f * f.spacing.x && fabsf(candidate.y) <= .45f * f.spacing.y &&
            fabsf(candidate.z) <= .45f * f.spacing.z &&
            length_squared(movement) > cell * cell * 1e-6f) {
            p.offset = candidate;
            p.relocated = 1;
            p.state = 0;
            p.updates = 0;
        }
    }
    if (!p.relocated && inside && f.classification)
        p.state = -1;
    else if (p.state < 0)
        p.state = 0;
    f.next[index] = p;
}
__global__ void trace_illumination(DdgiGpu f) {
    const int job = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (job >= f.update_count * f.ray_count)
        return;
    const int local = job / f.ray_count, ray_index = job % f.ray_count, index = f.indices[local];
    const auto probe = f.next[index];
    // Advance the sampling sequence only when this probe receives new samples.
    // Frame-based seeds subsample a different sequence at every update budget.
    const DVec3 dir = rotated_direction(ray_index, f.ray_count, index, probe.updates);
    DdgiRay result{dir, v3(0, 0, 0), f.far_distance, 0};
    if (probe.state < 0) {
        f.rays[local * (f.ray_count + 32) + 32 + ray_index] = result;
        return;
    }
    DHit hit{};
    if (!probe_hit(f, probe_position(f, index, probe), dir, hit))
        result.radiance = environment_radiance(f.scene, dir);
    else {
        result.distance = hit.t;
        if (hit.material_id >= 0 && hit.material_id < f.scene.material_count) {
            const auto material = f.scene.materials[hit.material_id];
            result.back = !hit.front_face && !material.two_sided;
            if (!result.back) {
                const auto surface = evaluate_surface(f.scene, material, hit);
                result.radiance = residual_emission(f, hit.position, surface.emission);
                if (material.type != int(MaterialType::Dielectric) &&
                    material.type != int(MaterialType::Emissive)) {
                    DPcgState rng;
                    pcg_seed(rng, (static_cast<unsigned long long>(probe.updates) << 32) ^
                                      unsigned(index * f.ray_count + ray_index));
                    const DVec3 outgoing = mul(dir, -1);
                    result.radiance =
                        add(result.radiance, probe_direct(f, hit, surface, outgoing, rng));
                    const DVec3 irradiance = probe_irradiance(f, hit.position, hit.geometric_normal,
                                                              surface.shading_normal, outgoing);
                    const DVec3 fr =
                        fresnel_schlick(saturate(dot(surface.shading_normal, outgoing)),
                                        surface.diffuse_fresnel_f0, surface.diffuse_fresnel_f90);
                    const float m = 1 - max_component(fr);
                    const DVec3 albedo = product(
                        surface.diffuse_color,
                        surface.diffuse_fresnel_uses_max ? v3(m, m, m) : sub(v3(1, 1, 1), fr));
                    result.radiance = add(
                        result.radiance, mul(product(albedo, irradiance), surface.occlusion / kPi));
                }
            }
        }
    }
    if (!finite(result.radiance)) {
        atomicCAS(f.error, 0, 100);
        result.radiance = v3(0, 0, 0);
    }
    f.rays[local * (f.ray_count + 32) + 32 + ray_index] = result;
}
__global__ void blend_probes(DdgiGpu f, float4 *output, int n) {
    const int job = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (job >= f.update_count * n * n)
        return;
    const int local = job / (n * n), texel = job % (n * n), index = f.indices[local];
    const auto p = f.next[index];
    const DVec3 direction =
        oct_decode((float(texel % n) + .5f) * 2 / n - 1, (float(texel / n) + .5f) * 2 / n - 1);
    float4 value = make_float4(0, 0, 0, 0);
    float total = 0;
    if (p.state >= 0)
        for (int i = 0; i < f.ray_count; ++i) {
            const auto r = f.rays[local * (f.ray_count + 32) + 32 + i];
            float w = fmaxf(0, dot(direction, r.direction));
            if (n == 16)
                w = powf(w, 50.0f);
            total += w;
            if (n == 8) {
                value.x += r.radiance.x * w;
                value.y += r.radiance.y * w;
                value.z += r.radiance.z * w;
            } else {
                const float d = r.back ? r.distance * .2f : r.distance;
                value.x += d * w;
                value.y += d * d * w;
            }
        }
    if (total > 1e-12f) {
        const float scale = (n == 8 ? kPi : 1.f) / total;
        value.x *= scale;
        value.y *= scale;
        value.z *= scale;
    } else if (n == 16) {
        value.x = f.far_distance;
        value.y = f.far_distance * f.far_distance;
    }
    const int pitch = f.columns * (n + 2), x = index % f.columns * (n + 2) + texel % n + 1,
              y = index / f.columns * (n + 2) + texel / n + 1;
    const int address = y * pitch + x;
    // A delayed update still contains only one batch of rays. It must not
    // replace several missing batches worth of history with that noisy sample.
    // Keep the configured per-measurement noise floor, while correcting faster
    // than 60 Hz updates so their temporal response remains time based.
    float h = p.updates > 0 && p.state == 1 && !p.relocated
                  ? powf(f.hysteresis, fminf(1.f, fmaxf(0.f, f.time - p.time) * 60))
                  : 0;
    // Do not retain the previous measurement during the first eight sweeps after
    // a transport change. The recursive history remains immutable for this frame,
    // but stale multibounce energy must be allowed to decay when a light turns off.
    if (n == 8 && f.fast)
        h = 0;
    // Lighting changes do not change visibility. Refresh stale geometry (also
    // including alpha/material edits) once, then keep accumulating distances.
    if (n == 16 && p.geometry_epoch != f.geometry_epoch)
        h = 0;
    const float4 old = (n == 8 ? f.irradiance : f.distance)[address];
    output[address] = make_float4(value.x * (1 - h) + old.x * h, value.y * (1 - h) + old.y * h,
                                  value.z * (1 - h) + old.z * h, 1);
}
__global__ void update_borders(DdgiGpu f, float4 *data, int n) {
    const int tile = n + 2, job = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (job >= f.update_count * tile * tile)
        return;
    const int index = f.indices[job / (tile * tile)], x = job % tile, y = (job / tile) % tile;
    if (x > 0 && y > 0 && x <= n && y <= n)
        return;
    int sx = x, sy = y;
    if ((x == 0 || x == n + 1) && (y == 0 || y == n + 1)) {
        sx = x == 0 ? n : 1;
        sy = y == 0 ? n : 1;
    } else if (x == 0 || x == n + 1) {
        sx = x == 0 ? 1 : n;
        sy = n + 1 - y;
    } else {
        sx = n + 1 - x;
        sy = y == 0 ? 1 : n;
    }
    const int width = f.columns * tile,
              base = index / f.columns * tile * width + index % f.columns * tile;
    data[base + y * width + x] = data[base + sy * width + sx];
}
__global__ void finish_probes(DdgiGpu f) {
    const int job = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (job >= f.update_count)
        return;
    auto &p = f.next[f.indices[job]];
    if (p.state >= 0) {
        p.state = 1;
        ++p.updates;
    }
    p.frame = f.frame;
    p.time = f.time;
    p.geometry_epoch = f.geometry_epoch;
}
__global__ void probe_counters(const DdgiProbe *probes, int count, unsigned int frame,
                               unsigned int *counters) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    if (probes[index].state == 1)
        atomicAdd(counters, 1u);
    atomicMax(counters + 1, frame - probes[index].frame);
}
__global__ void export_texture(const float4 *data, int count, int width,
                               cudaSurfaceObject_t surface) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count)
        surf2Dwrite(data[i], surface, (i % width) * int(sizeof(float4)), i / width);
}
__global__ void export_metadata(const DdgiProbe *p, int count, unsigned int frame,
                                cudaSurfaceObject_t surface) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count)
        return;
    surf2Dwrite(make_float4(p[i].offset.x, p[i].offset.y, p[i].offset.z, float(p[i].state)),
                surface, i * int(sizeof(float4)), 0);
    surf2Dwrite(make_float4(float(p[i].updates), float(p[i].frame), float(frame - p[i].frame), 0),
                surface, i * int(sizeof(float4)), 1);
}
} // namespace
class CudaDdgiVolume::Impl {
  public:
    explicit Impl(CudaDeviceContext context) : context_(context) {
        context_.activate();
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "DDGI stream");
        error_.resize(1, storage_stats_);
        counters_.resize(2, storage_stats_);
        stats_.available = true;
    }
    ~Impl() {
        context_.activate();
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }
    void update(const RenderSceneSnapshot &snapshot, const OpenGlRenderSettings &settings,
                const InteractiveFrameState &state) {
        context_.activate();
        const auto config = normalized_ddgi_settings(settings.ddgi);
        const bool new_scene = !scene_ || source_id_ != snapshot.source_id;
        SceneChangeSet changes = scene_changes_for_snapshot(source_id_, revisions_, bool(scene_),
                                                            snapshot, state.scene_changes);
        const bool transport_changed =
            !has_settings_ || settings.ibl_enabled != cached_.ibl_enabled ||
            settings.ltc_area_lights_enabled != cached_.ltc_area_lights_enabled ||
            settings.dominant_light != cached_.dominant_light ||
            settings.shadow_map.enabled != cached_.shadow_map.enabled;
        const bool layout_changed =
            new_scene || !has_settings_ || config.probe_counts != cached_.ddgi.probe_counts ||
            config.auto_fit != cached_.ddgi.auto_fit ||
            (!config.auto_fit &&
             (config.origin != cached_.ddgi.origin || config.extent != cached_.ddgi.extent)) ||
            config.fit_generation != cached_.ddgi.fit_generation;
        const bool reset = layout_changed ||
                           config.reset_generation != cached_.ddgi.reset_generation ||
                           state.reset_requested;
        if (transport_changed)
            changes |= SceneChange::Lighting | SceneChange::Environment;
        if (changes != SceneChange::None || new_scene) {
            RenderSceneSnapshot effective = snapshot;
            if (settings.dominant_light.enabled) {
                const auto dominant = extract_dominant_environment_light(
                    snapshot.environment_map, settings.dominant_light.peak_threshold_ev,
                    settings.dominant_light.minimum_energy_fraction);
                if (dominant.valid) {
                    const float angle = snapshot.environment_rotation_degrees * kPi / 180;
                    const Vec3 d = dominant.direction;
                    const Vec3 toward(std::cos(angle) * d.x() + std::sin(angle) * d.z(), d.y(),
                                      -std::sin(angle) * d.x() + std::cos(angle) * d.z());
                    DirectionalLight light;
                    light.direction = -toward;
                    light.radiance =
                        dominant.integrated_radiance.cwiseProduct(snapshot.environment) *
                        (std::max(0.f, snapshot.environment_intensity) *
                         std::max(0.f, settings.dominant_light.intensity_scale));
                    light.angular_radius_radians = dominant.angular_radius_radians;
                    effective.directional_lights.push_back(light);
                    effective.environment_map = dominant.residual_map;
                }
            }
            if (!settings.ibl_enabled)
                effective.environment_intensity = 0;
            if (new_scene)
                scene_ = std::make_unique<CudaSceneStorage>(effective, stream_, storage_stats_);
            else
                scene_->sync(effective, changes);
            std::vector<DdgiRect> rectangles;
            if (settings.ltc_area_lights_enabled)
                for (const auto &light : snapshot.rect_area_lights)
                    rectangles.push_back({to_device(light.position), to_device(light.axis_u),
                                          to_device(light.axis_v), to_device(light.radiance),
                                          light.two_sided ? 1 : 0, light.casts_shadows ? 1 : 0});
            rectangles_.upload(rectangles, stream_, storage_stats_);
            // Keep temporary upload sources alive until the transfers complete.
            check_cuda(cudaStreamSynchronize(stream_), "DDGI scene sync");
        }
        if (layout_changed)
            layout_ = make_ddgi_layout(snapshot, config);
        if (reset)
            initialize();
        if (reset || has_scene_change(changes, SceneChange::Geometry) ||
            has_scene_change(changes, SceneChange::InstanceTransforms) ||
            has_scene_change(changes, SceneChange::MaterialBindings) ||
            has_scene_change(changes, SceneChange::Materials) ||
            has_scene_change(changes, SceneChange::Textures))
            ++geometry_epoch_;
        if (changes != SceneChange::None || transport_changed ||
            config.relocation != cached_.ddgi.relocation ||
            config.classification != cached_.ddgi.classification)
            fast_remaining_ = layout_.count() * 8;
        if (has_scene_change(changes, SceneChange::Geometry) ||
            has_scene_change(changes, SceneChange::InstanceTransforms)) {
            const Vec3 margin(layout_.spacing[0] * 1.5f, layout_.spacing[1] * 1.5f,
                              layout_.spacing[2] * 1.5f);
            const auto queue_bounds = [&](const Bounds3 &bounds) {
                for (int i = 0; i < layout_.count(); ++i) {
                    const Vec3 p = layout_.position(i);
                    if (!queued_[std::size_t(i)] &&
                        (p.array() >= (bounds.min - margin).array()).all() &&
                        (p.array() <= (bounds.max + margin).array()).all()) {
                        priority_.push_back(i);
                        queued_[std::size_t(i)] = true;
                    }
                }
            };
            std::unordered_map<std::uint64_t, Bounds3> current;
            for (const auto &instance : snapshot.instances) {
                const auto old = bounds_.find(instance.object_id);
                const bool changed = old == bounds_.end() ||
                                     !old->second.min.isApprox(instance.world_bounds.min) ||
                                     !old->second.max.isApprox(instance.world_bounds.max) ||
                                     has_scene_change(changes, SceneChange::Geometry);
                if (changed) {
                    if (old != bounds_.end())
                        queue_bounds(old->second);
                    queue_bounds(instance.world_bounds);
                }
                current.emplace(instance.object_id, instance.world_bounds);
            }
            for (const auto &[id, b] : bounds_)
                if (!current.contains(id))
                    queue_bounds(b);
            bounds_ = std::move(current);
        }
        source_id_ = snapshot.source_id;
        revisions_ = snapshot.revisions;
        cached_ = settings;
        cached_.ddgi = config;
        has_settings_ = true;
        time_ += std::isfinite(state.delta_seconds) && state.delta_seconds > 0
                     ? state.delta_seconds
                     : 1.f / 60;
        ++frame_;
        stats_.frame_index = frame_;
        stats_.updated_probes = 0;
        if (config.paused) {
            stats_.active = true;
            ++stats_.maximum_age;
            return;
        }
        std::vector<int> jobs;
        std::vector<bool> selected(std::size_t(layout_.count()), false);
        const int budget = config.probes_per_frame;
        while (!priority_.empty() && int(jobs.size()) < std::max(1, budget / 2)) {
            const int index = priority_.front();
            priority_.pop_front();
            queued_[std::size_t(index)] = false;
            if (!selected[std::size_t(index)]) {
                jobs.push_back(index);
                selected[std::size_t(index)] = true;
            }
        }
        while (int(jobs.size()) < budget) {
            const int index = cursor_;
            cursor_ = (cursor_ + 1) % layout_.count();
            if (!selected[std::size_t(index)]) {
                jobs.push_back(index);
                selected[std::size_t(index)] = true;
            }
        }
        indices_.upload(jobs, stream_, storage_stats_);
        rays_.resize(std::size_t(budget * (config.rays_per_probe + 32)), storage_stats_);
        const int next = 1 - current_;
        copy_buffer(irradiance_[next], irradiance_[current_]);
        copy_buffer(distance_[next], distance_[current_]);
        copy_buffer(probes_[next], probes_[current_]);
        zero_buffer(error_, stream_);
        zero_buffer(counters_, stream_);
        DdgiGpu f{};
        f.scene = scene_->view();
        f.origin = DVec3{layout_.origin[0], layout_.origin[1], layout_.origin[2]};
        f.spacing = DVec3{layout_.spacing[0], layout_.spacing[1], layout_.spacing[2]};
        f.nx = layout_.counts[0];
        f.ny = layout_.counts[1];
        f.nz = layout_.counts[2];
        f.count = layout_.count();
        f.columns = layout_.columns;
        f.rows = layout_.rows;
        f.ray_count = config.rays_per_probe;
        f.update_count = budget;
        f.frame = frame_;
        f.geometry_epoch = geometry_epoch_;
        f.time = time_;
        f.hysteresis = config.hysteresis;
        f.normal_bias = config.normal_bias;
        f.view_bias = config.view_bias;
        f.fast = fast_remaining_ > 0;
        f.relocation = config.relocation;
        f.classification = config.classification;
        f.shadows = settings.shadow_map.enabled;
        f.far_distance = 2 * std::sqrt(std::pow(layout_.spacing[0] * (f.nx - 1), 2) +
                                       std::pow(layout_.spacing[1] * (f.ny - 1), 2) +
                                       std::pow(layout_.spacing[2] * (f.nz - 1), 2));
        // A manually placed volume can be much smaller than the geometry around it.
        // Fixed classification rays must still reach that geometry from inside it.
        const Vec3 volume_min = layout_.position(0),
                   volume_max = layout_.position(layout_.count() - 1);
        for (const auto &instance : snapshot.instances) {
            const auto &b = instance.world_bounds;
            if (b.min.allFinite() && b.max.allFinite()) {
                const Vec3 span =
                    (b.max - volume_min).cwiseAbs().cwiseMax((volume_max - b.min).cwiseAbs());
                f.far_distance = std::max(f.far_distance, span.norm() * 1.1f);
            }
        }
        f.indices = indices_.get();
        f.rectangles = rectangles_.get();
        f.rectangle_count = int(rectangles_.size());
        f.previous = probes_[current_].get();
        f.next = probes_[next].get();
        f.irradiance = irradiance_[current_].get();
        f.distance = distance_[current_].get();
        f.rays = rays_.get();
        f.error = error_.get();
        trace_timer_.begin(stream_);
        trace_fixed<<<blocks(budget * 32), 128, 0, stream_>>>(f);
        relocate_probes<<<blocks(budget), 128, 0, stream_>>>(f);
        trace_illumination<<<blocks(budget * f.ray_count), 128, 0, stream_>>>(f);
        trace_timer_.end(stream_);
        blend_timer_.begin(stream_);
        blend_probes<<<blocks(budget * 64), 128, 0, stream_>>>(f, irradiance_[next].get(), 8);
        blend_probes<<<blocks(budget * 256), 128, 0, stream_>>>(f, distance_[next].get(), 16);
        update_borders<<<blocks(budget * 100), 128, 0, stream_>>>(f, irradiance_[next].get(), 8);
        update_borders<<<blocks(budget * 324), 128, 0, stream_>>>(f, distance_[next].get(), 16);
        finish_probes<<<blocks(budget), 128, 0, stream_>>>(f);
        probe_counters<<<blocks(f.count), 128, 0, stream_>>>(f.next, f.count, frame_,
                                                             counters_.get());
        blend_timer_.end(stream_);
        check_cuda(cudaGetLastError(), "DDGI kernels");
        int error = 0;
        unsigned int counts[2]{};
        check_cuda(
            cudaMemcpyAsync(&error, error_.get(), sizeof(error), cudaMemcpyDeviceToHost, stream_),
            "DDGI error read");
        check_cuda(cudaMemcpyAsync(counts, counters_.get(), sizeof(counts), cudaMemcpyDeviceToHost,
                                   stream_),
                   "DDGI counters read");
        check_cuda(cudaStreamSynchronize(stream_), "DDGI update");
        if (error)
            throw std::runtime_error("DDGI traversal/nonfinite error " + std::to_string(error));
        trace_timer_.update(stats_.trace_ms);
        blend_timer_.update(stats_.blend_ms);
        scene_->update_timing();
        current_ = next;
        fast_remaining_ = std::max(0, fast_remaining_ - budget);
        stats_.active = true;
        stats_.status = "ready";
        stats_.updated_probes = budget;
        stats_.active_probes = int(counts[0]);
        stats_.maximum_age = int(counts[1]);
        stats_.tlas_refits = storage_stats_.tlas_refit_count;
        stats_.blas_builds = storage_stats_.blas_build_count;
        stats_.memory_bytes = 2 * (irradiance_[0].size() + distance_[0].size()) * sizeof(float4) +
                              2 * probes_[0].size() * sizeof(DdgiProbe) +
                              rays_.capacity() * sizeof(DdgiRay);
    }
    void export_atlases(CudaSurfaceHandle irradiance, CudaSurfaceHandle distance,
                        CudaSurfaceHandle metadata) {
        context_.activate();
        export_texture<<<blocks(int(irradiance_[current_].size())), 128, 0, stream_>>>(
            irradiance_[current_].get(), int(irradiance_[current_].size()), layout_.width(8),
            irradiance);
        export_texture<<<blocks(int(distance_[current_].size())), 128, 0, stream_>>>(
            distance_[current_].get(), int(distance_[current_].size()), layout_.width(16),
            distance);
        export_metadata<<<blocks(layout_.count()), 128, 0, stream_>>>(
            probes_[current_].get(), layout_.count(), frame_, metadata);
        check_cuda(cudaGetLastError(), "DDGI atlas export");
    }
    DdgiAtlasReadback download() {
        context_.activate();
        DdgiAtlasReadback result;
        result.irradiance.resize(irradiance_[current_].size());
        result.distance.resize(distance_[current_].size());
        std::vector<DdgiProbe> probes(std::size_t(layout_.count()));
        result.metadata.resize(probes.size() * 2);
        check_cuda(cudaMemcpyAsync(result.irradiance.data(), irradiance_[current_].get(),
                                   result.irradiance.size() * sizeof(float4),
                                   cudaMemcpyDeviceToHost, stream_),
                   "DDGI irradiance download");
        check_cuda(cudaMemcpyAsync(result.distance.data(), distance_[current_].get(),
                                   result.distance.size() * sizeof(float4), cudaMemcpyDeviceToHost,
                                   stream_),
                   "DDGI distance download");
        check_cuda(cudaMemcpyAsync(probes.data(), probes_[current_].get(),
                                   probes.size() * sizeof(DdgiProbe), cudaMemcpyDeviceToHost,
                                   stream_),
                   "DDGI metadata download");
        check_cuda(cudaStreamSynchronize(stream_), "DDGI atlas download");
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const auto p = probes[i];
            result.metadata[i] = {p.offset.x, p.offset.y, p.offset.z, float(p.state)};
            result.metadata[i + probes.size()] = {float(p.updates), float(p.frame),
                                                  float(frame_ - p.frame), 0};
        }
        ++stats_.atlas_downloads;
        return result;
    }
    CudaDeviceContext context_;
    cudaStream_t stream_ = nullptr;
    DdgiStatistics stats_;

  private:
    static int blocks(int size) {
        return (size + 127) / 128;
    }
    template <class T> void copy_buffer(DeviceBuffer<T> &dst, const DeviceBuffer<T> &src) {
        check_cuda(cudaMemcpyAsync(dst.get(), src.get(), src.size() * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream_),
                   "DDGI history copy");
    }
    void initialize() {
        check_cuda(cudaStreamSynchronize(stream_), "DDGI reset");
        for (int i = 0; i < 2; ++i) {
            irradiance_[i].resize_exact(std::size_t(layout_.width(8) * layout_.height(8)),
                                        storage_stats_);
            distance_[i].resize_exact(std::size_t(layout_.width(16) * layout_.height(16)),
                                      storage_stats_);
            probes_[i].resize_exact(std::size_t(layout_.count()), storage_stats_);
            zero_buffer(irradiance_[i], stream_);
            zero_buffer(distance_[i], stream_);
            zero_buffer(probes_[i], stream_);
        }
        current_ = 0;
        cursor_ = 0;
        frame_ = 0;
        time_ = 0;
        priority_.clear();
        queued_.assign(std::size_t(layout_.count()), false);
        stats_.layout = layout_;
        stats_.probe_count = layout_.count();
        stats_.active_probes = 0;
        stats_.maximum_age = 0;
        ++stats_.reset_count;
        fast_remaining_ = layout_.count() * 8;
    }
    CudaPathStatistics storage_stats_;
    std::unique_ptr<CudaSceneStorage> scene_;
    DeviceBuffer<float4> irradiance_[2], distance_[2];
    DeviceBuffer<DdgiProbe> probes_[2];
    DeviceBuffer<DdgiRay> rays_;
    DeviceBuffer<DdgiRect> rectangles_;
    DeviceBuffer<int> indices_, error_;
    DeviceBuffer<unsigned int> counters_;
    CudaEventTimer trace_timer_, blend_timer_;
    DdgiLayout layout_;
    OpenGlRenderSettings cached_;
    bool has_settings_ = false;
    SceneRevisions revisions_;
    std::uint64_t source_id_ = 0;
    unsigned int frame_ = 0, geometry_epoch_ = 0;
    float time_ = 0;
    int current_ = 0, cursor_ = 0, fast_remaining_ = 0;
    std::deque<int> priority_;
    std::vector<bool> queued_;
    std::unordered_map<std::uint64_t, Bounds3> bounds_;
};

CudaDdgiVolume::CudaDdgiVolume(CudaDeviceContext context)
    : impl_(std::make_unique<Impl>(context)) {}
CudaDdgiVolume::~CudaDdgiVolume() = default;
void CudaDdgiVolume::update(const RenderSceneSnapshot &scene, const OpenGlRenderSettings &settings,
                            const InteractiveFrameState &state) {
    impl_->update(scene, settings, state);
}
void CudaDdgiVolume::export_atlases(CudaSurfaceHandle a, CudaSurfaceHandle b, CudaSurfaceHandle c) {
    impl_->export_atlases(a, b, c);
}
DdgiAtlasReadback CudaDdgiVolume::download_atlases() {
    return impl_->download();
}
CudaStreamHandle CudaDdgiVolume::stream_handle() const {
    return reinterpret_cast<CudaStreamHandle>(impl_->stream_);
}
const DdgiStatistics &CudaDdgiVolume::statistics() const {
    return impl_->stats_;
}
} // namespace renderer
