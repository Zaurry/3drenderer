#include "render/pathtracer/cuda_scene.cuh"

namespace renderer {
namespace {

__global__ void initialize_frame_kernel(
    DVec3* accumulation,
    DPcgState* random_states,
    int width,
    int height,
    unsigned long long seed_offset,
    int* error_code) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = width * height;
    if (index >= pixel_count) {
        return;
    }
    accumulation[index] = v3(0.0f, 0.0f, 0.0f);
    const int x = index % width;
    const int y = index / width;
    pcg_seed(random_states[index], pixel_seed(x, y, width, seed_offset));
    if (index == 0) {
        *error_code = 0;
    }
}

__global__ void prepare_sample_kernel(
    const DFrameParameters* parameters,
    DPathState* primary_paths,
    int* primary_count,
    int* secondary_count,
    int* bounce_index,
    cudaGraphConditionalHandle wavefront_handle,
    cudaGraphConditionalHandle emissive_handle) {
    const DFrameParameters& frame = *parameters;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int sample_index = first;
         sample_index < frame.pixel_count;
         sample_index += stride) {
        const int pixel_index = frame.pixel_offset + sample_index;
        const int x = pixel_index % frame.width;
        const int y = pixel_index / frame.width;
        DPcgState rng = frame.random_states[pixel_index];
        const float u =
            (static_cast<float>(x) + random_float(rng)) /
            static_cast<float>(frame.width);
        const float v = 1.0f -
            (static_cast<float>(y) + random_float(rng)) /
                static_cast<float>(frame.height);
        const DVec3 direction = normalize(add(
            add(
                frame.camera.forward,
                mul(
                    frame.camera.right,
                    (u - 0.5f) * frame.camera.viewport_width)),
            mul(
                frame.camera.up,
                (v - 0.5f) * frame.camera.viewport_height)));
        primary_paths[sample_index] = DPathState{
            DRay{frame.camera.eye, direction},
            v3(1.0f, 1.0f, 1.0f),
            0.0f,
            1,
            sample_index};
        frame.sample_radiance[sample_index] =
            v3(0.0f, 0.0f, 0.0f);
        frame.random_states[pixel_index] = rng;
    }
    if (first == 0) {
        *primary_count = frame.pixel_count;
        *secondary_count = 0;
        *bounce_index = 0;
        if (wavefront_handle != 0) {
            cudaGraphSetConditional(wavefront_handle, 1U);
        }
        if (emissive_handle != 0) {
            cudaGraphSetConditional(
                emissive_handle,
                (frame.scene.emissive_light_count > 0 ||
                 environment_direct_active(frame.scene))
                    ? 1U
                    : 0U);
        }
    }
}

__global__ void intersect_wavefront_kernel(
    const DFrameParameters* parameters,
    const DPathState* first_paths,
    const DPathState* second_paths,
    int* path_counts,
    const int* bounce_index,
    DWavefrontHit* hits) {
    const DFrameParameters& frame = *parameters;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int active_index = *bounce_index & 1;
    const DPathState* active_paths =
        active_index == 0 ? first_paths : second_paths;
    int count = path_counts[active_index];
    if (first == 0) {
        path_counts[1 - active_index] = 0;
    }
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    if (first == 0) {
        const int bounce = *bounce_index;
        atomicAdd(
            frame.diagnostic_counters +
                (bounce == 0
                    ? kDiagnosticPrimaryRays
                    : kDiagnosticContinuationRays),
            static_cast<unsigned long long>(count));
        atomicAdd(
            frame.diagnostic_counters + kDiagnosticRaysByBounce +
                max(0, min(bounce, kMaxPathBounces)),
            static_cast<unsigned long long>(count));
    }
#endif
    if (first >= count) {
        return;
    }
    DCompactHit hit{};
    const bool found = intersect_scene_compact(
        frame.scene,
        active_paths[first].ray,
        0.0f,
        1.0e30f,
        hit,
        frame.error_code);
    hits[first] = DWavefrontHit{hit, found ? 1 : 0};
}

__global__ void shade_wavefront_kernel(
    const DFrameParameters* parameters,
    DPathState* first_paths,
    DPathState* second_paths,
    int* path_counts,
    const int* bounce_index,
    const DWavefrontHit* hits,
    DShadingRecord* shading_records) {
    const DFrameParameters& frame = *parameters;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int bounce = *bounce_index;
    const int active_index = bounce & 1;
    const int next_index = 1 - active_index;
    const DPathState* active_paths =
        active_index == 0 ? first_paths : second_paths;
    DPathState* next_paths =
        active_index == 0
        ? second_paths
        : first_paths;
    int count = path_counts[active_index];
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
    if (index >= count) {
        return;
    }

    const DPathState path = active_paths[index];
    const DWavefrontHit wavefront_hit = hits[index];
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    const auto record_termination = [&frame](int surface_bounces) {
        atomicAdd(
            frame.diagnostic_counters + kDiagnosticTerminationByBounce +
                max(0, min(surface_bounces, kMaxPathBounces)),
            1ULL);
    };
    if (bounce == 0) {
        atomicAdd(
            frame.diagnostic_counters +
                (wavefront_hit.found
                    ? kDiagnosticPrimaryHits
                    : kDiagnosticPrimaryMisses),
            1ULL);
    }
#endif
    shading_records[index].valid = 0;
    if (!wavefront_hit.found) {
        if (bounce > 0 || frame.scene.environment_background_visible != 0) {
            const DVec3 incoming = environment_radiance(
                frame.scene,
                path.ray.direction);
            const float weight = bounce == 0 || path.previous_was_delta != 0
                ? 1.0f
                : power_heuristic(
                      path.previous_bsdf_pdf,
                      environment_strategy_probability(frame.scene) *
                          environment_pdf(frame.scene, path.ray.direction));
            frame.sample_radiance[path.pixel_index] = add(
                frame.sample_radiance[path.pixel_index],
                mul(product(path.throughput, incoming), weight));
        }
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce);
#endif
        return;
    }

    DHit hit{};
    reconstruct_hit(
        frame.scene,
        path.ray,
        wavefront_hit.hit,
        true,
        hit);
    if (hit.material_id < 0 || hit.material_id >= frame.scene.material_count) {
        frame.sample_radiance[path.pixel_index] = add(
            frame.sample_radiance[path.pixel_index],
            product(path.throughput, v3(1.0f, 0.0f, 1.0f)));
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
        return;
    }
    const DMaterial material = frame.scene.materials[hit.material_id];
    const DSurface surface = evaluate_surface(frame.scene, material, hit);
    const int random_index = frame.pixel_offset + path.pixel_index;
    DPcgState rng = frame.random_states[random_index];
    // evaluate_surface() already computed coverage. Only the effective mode is
    // needed here; sampling the opacity textures again is both redundant and
    // a measurable cost in foliage-heavy scenes.
    const int alpha_mode = effective_alpha_mode(material);
    const bool transparent_passthrough =
        alpha_mode == static_cast<int>(AlphaMode::Blend) &&
        random_float(rng) >= surface.opacity;

    if (!transparent_passthrough && max_component(surface.emission) > 0.0f) {
        float emission_weight = 1.0f;
        if (!path.previous_was_delta && path.previous_bsdf_pdf > 0.0f) {
            const float light_pdf = emissive_light_pdf_for_hit(
                frame.scene,
                path.ray.origin,
                hit);
            emission_weight = power_heuristic(
                path.previous_bsdf_pdf,
                light_pdf);
        }
        frame.sample_radiance[path.pixel_index] = add(
            frame.sample_radiance[path.pixel_index],
            mul(
                product(path.throughput, surface.emission),
                emission_weight));
    }
    if (!transparent_passthrough &&
        material.type == static_cast<int>(MaterialType::Emissive)) {
        frame.random_states[random_index] = rng;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
        return;
    }

    const DVec3 outgoing = normalize(mul(path.ray.direction, -1.0f));
    if (!transparent_passthrough &&
        (material.type == static_cast<int>(MaterialType::Diffuse) ||
         material.type == static_cast<int>(MaterialType::Metal) ||
         material.type == static_cast<int>(MaterialType::Pbr))) {
        shading_records[index] = DShadingRecord{
            hit.position,
            hit.geometric_normal,
            surface.shading_normal,
            surface.base_color,
            surface.diffuse_color,
            surface.specular_f0,
            surface.specular_f90,
            surface.diffuse_fresnel_f0,
            surface.diffuse_fresnel_f90,
            surface.diffuse_fresnel_uses_max,
            outgoing,
            surface.metallic,
            surface.roughness,
            surface.occlusion,
            path.throughput,
            path.pixel_index,
            1};
    }

    DVec3 attenuation{};
    DRay scattered{};
    float bsdf_pdf = 0.0f;
    int was_delta = 1;
    if (transparent_passthrough) {
        attenuation = v3(1.0f, 1.0f, 1.0f);
        scattered = DRay{
            offset_origin(hit.position, hit.geometric_normal, path.ray.direction),
            path.ray.direction};
        bsdf_pdf = 0.0f;
        was_delta = 1;
    } else if (!scatter(
                   path.ray,
                   hit,
                   material,
                   surface,
                   rng,
                   attenuation,
                   scattered,
                   bsdf_pdf,
                   was_delta)) {
        frame.random_states[random_index] = rng;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
        return;
    }

    DVec3 throughput = product(path.throughput, attenuation);
    if (!finite(throughput) || max_component(throughput) <= 0.0f ||
        bounce + 1 >= frame.max_bounces) {
        frame.random_states[random_index] = rng;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
        return;
    }
    if (bounce + 1 >= frame.russian_roulette_start_bounce) {
        const float probability = fminf(
            fmaxf(
                max_component(throughput),
                frame.russian_roulette_min_probability),
            frame.russian_roulette_max_probability);
        if (random_float(rng) >= probability) {
            frame.random_states[random_index] = rng;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
            record_termination(bounce + 1);
#endif
            return;
        }
        throughput = divv(throughput, probability);
    }

    const unsigned int active_mask = __activemask();
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int leader = __ffs(active_mask) - 1;
    int output_base = 0;
    if (lane == leader) {
        output_base = atomicAdd(
            path_counts + next_index,
            __popc(active_mask));
    }
    output_base = __shfl_sync(
        active_mask,
        output_base,
        leader);
    const unsigned int lower_lanes =
        lane == 0 ? 0U : ((1U << lane) - 1U);
    const int output_index =
        output_base + __popc(active_mask & lower_lanes);
    if (output_index < frame.path_capacity) {
        next_paths[output_index] = DPathState{
            scattered,
            throughput,
            bsdf_pdf,
            was_delta,
            path.pixel_index};
    } else {
        atomicCAS(frame.error_code, 0, 1);
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
    }
    frame.random_states[random_index] = rng;
}

__global__ void sample_emissive_lights_kernel(
    const DFrameParameters* parameters,
    const DShadingRecord* shading_records,
    DShadowTask* shadow_tasks,
    const int* path_counts,
    const int* bounce_index) {
    const DFrameParameters& frame = *parameters;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    int count = path_counts[*bounce_index & 1];
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
    if (first >= count) {
        return;
    }
    shadow_tasks[first].t_max = 0.0f;
    if (frame.scene.emissive_light_count <= 0 &&
        !environment_direct_active(frame.scene)) {
        return;
    }
    const DShadingRecord record = shading_records[first];
    if (!record.valid) {
        return;
    }
    DHit hit{};
    hit.position = record.position;
    hit.geometric_normal = record.geometric_normal;
    DSurface surface{};
    surface.base_color = record.base_color;
    surface.diffuse_color = record.diffuse_color;
    surface.specular_f0 = record.specular_f0;
    surface.specular_f90 = record.specular_f90;
    surface.diffuse_fresnel_f0 = record.diffuse_fresnel_f0;
    surface.diffuse_fresnel_f90 = record.diffuse_fresnel_f90;
    surface.diffuse_fresnel_uses_max = record.diffuse_fresnel_uses_max;
    surface.metallic = record.metallic;
    surface.roughness = record.roughness;
    surface.occlusion = record.occlusion;
    surface.shading_normal = record.shading_normal;
    const int random_index =
        frame.pixel_offset + record.pixel_index;
    DPcgState rng = frame.random_states[random_index];
    DShadowTask shadow_task{};
    const float environment_probability =
        environment_strategy_probability(frame.scene);
    const bool choose_environment = environment_probability > 0.0f &&
        (frame.scene.emissive_light_count <= 0 ||
         random_float(rng) < environment_probability);
    const bool sampled = choose_environment
        ? sample_environment_shadow_task(
              frame.scene,
              hit,
              surface,
              record.outgoing,
              record.throughput,
              record.pixel_index,
              rng,
              shadow_task)
        : sample_emissive_shadow_task(
              frame.scene,
              hit,
              surface,
              record.outgoing,
              record.throughput,
              record.pixel_index,
              rng,
              shadow_task);
    if (sampled) {
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        shadow_task.diagnostic_kind = choose_environment
            ? kDiagnosticShadowEnvironment
            : kDiagnosticShadowEmissive;
#endif
        shadow_tasks[first] = shadow_task;
    }
    frame.random_states[random_index] = rng;
}

__global__ void direct_visibility_kernel(
    const DFrameParameters* parameters,
    const DShadingRecord* shading_records,
    const int* path_counts,
    const int* bounce_index,
    const DShadowTask* shadow_tasks) {
    const DFrameParameters& frame = *parameters;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    int count = path_counts[*bounce_index & 1];
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
    if (first >= count) {
        return;
    }
    const DShadingRecord record = shading_records[first];
    if (!record.valid) {
        return;
    }
    DHit hit{};
    hit.position = record.position;
    hit.geometric_normal = record.geometric_normal;
    DSurface surface{};
    surface.base_color = record.base_color;
    surface.diffuse_color = record.diffuse_color;
    surface.specular_f0 = record.specular_f0;
    surface.specular_f90 = record.specular_f90;
    surface.diffuse_fresnel_f0 = record.diffuse_fresnel_f0;
    surface.diffuse_fresnel_f90 = record.diffuse_fresnel_f90;
    surface.diffuse_fresnel_uses_max = record.diffuse_fresnel_uses_max;
    surface.metallic = record.metallic;
    surface.roughness = record.roughness;
    surface.occlusion = record.occlusion;
    surface.shading_normal = record.shading_normal;
    DVec3 contribution = product(
        record.throughput,
        direct_lighting(
            frame.scene,
            hit,
            surface,
            record.outgoing,
            frame.error_code
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
            , frame.diagnostic_counters
#endif
            ));
    if (frame.scene.emissive_light_count > 0 ||
        environment_direct_active(frame.scene)) {
        const DShadowTask task = shadow_tasks[first];
        if (task.t_max > 0.0f &&
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
            (atomicAdd(
                frame.diagnostic_counters +
                    (task.diagnostic_kind == kDiagnosticShadowEnvironment
                        ? kDiagnosticEnvironmentShadowRays
                        : kDiagnosticEmissiveShadowRays),
                1ULL), true) &&
#endif
            (!task.casts_shadows || !occluded_scene(
                frame.scene,
                task.ray,
                0.0f,
                task.t_max,
                frame.error_code))) {
            contribution = add(
                contribution,
                task.contribution);
        }
    }
    frame.sample_radiance[record.pixel_index] = add(
        frame.sample_radiance[record.pixel_index],
        contribution);
}

__global__ void advance_wavefront_kernel(
    const DFrameParameters* parameters,
    const int* path_counts,
    int* bounce_index,
    cudaGraphConditionalHandle conditional_handle) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const DFrameParameters& frame = *parameters;
    const int next_bounce = *bounce_index + 1;
    *bounce_index = next_bounce;
    int count = next_bounce < frame.max_bounces
        ? path_counts[next_bounce & 1]
        : 0;
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = 0;
    }
    if (conditional_handle != 0) {
        cudaGraphSetConditional(
            conditional_handle,
            count > 0 && next_bounce < frame.max_bounces ? 1U : 0U);
    }
}

__global__ void finalize_sample_kernel(
    const DFrameParameters* parameters) {
    const DFrameParameters& frame = *parameters;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int sample_index = first;
         sample_index < frame.pixel_count;
         sample_index += stride) {
        const int pixel_index = frame.pixel_offset + sample_index;
        const DVec3 sum = add(
            frame.accumulation[pixel_index],
            frame.sample_radiance[sample_index]);
        frame.accumulation[pixel_index] = sum;
        if (frame.output_surface != 0) {
            const int completed_samples =
                frame.completed_samples + *frame.batch_sample_index;
            const DVec3 color = divv(
                sum,
                static_cast<float>(completed_samples + 1));
            const int x = pixel_index % frame.width;
            const int y = pixel_index / frame.width;
            const int output_x_begin =
                x * frame.output_width / frame.width;
            const int output_x_end =
                (x + 1) * frame.output_width / frame.width;
            const int output_y_begin =
                y * frame.output_height / frame.height;
            const int output_y_end =
                (y + 1) * frame.output_height / frame.height;
            const float4 output =
                make_float4(color.x, color.y, color.z, 1.0f);
            for (int output_y = output_y_begin;
                 output_y < output_y_end;
                 ++output_y) {
                for (int output_x = output_x_begin;
                     output_x < output_x_end;
                     ++output_x) {
                    surf2Dwrite(
                        output,
                        frame.output_surface,
                        output_x * static_cast<int>(sizeof(float4)),
                        output_y);
                }
            }
        }
    }
}

__global__ void prepare_sample_batch_kernel(int* batch_sample_index) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *batch_sample_index = 0;
    }
}

__global__ void advance_sample_batch_kernel(
    const DFrameParameters* parameters,
    cudaGraphConditionalHandle batch_handle) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const DFrameParameters& frame = *parameters;
    const int next_sample = *frame.batch_sample_index + 1;
    *frame.batch_sample_index = next_sample;
    if (batch_handle != 0) {
        cudaGraphSetConditional(
            batch_handle,
            next_sample < frame.batch_sample_count ? 1U : 0U);
    }
}

__global__ void update_frame_parameters_kernel(
    DFrameParameters* destination,
    DFrameParameters parameters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *destination = parameters;
    }
}

__global__ void resolve_frame_kernel(
    const DVec3* accumulation,
    DVec3* resolved,
    int pixel_count,
    int pixel_offset,
    int sample_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    resolved[index] = sample_count > 0
        ? divv(
            accumulation[pixel_offset + index],
            static_cast<float>(sample_count))
        : v3(0.0f, 0.0f, 0.0f);
}

__global__ void present_frame_kernel(
    const DVec3* accumulation,
    int width,
    int height,
    int sample_count,
    cudaSurfaceObject_t output_surface) {
    const int pixel_count = width * height;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int index = first; index < pixel_count; index += stride) {
        const DVec3 color = sample_count > 0
            ? divv(
                accumulation[index],
                static_cast<float>(sample_count))
            : v3(0.0f, 0.0f, 0.0f);
        const float4 output =
            make_float4(color.x, color.y, color.z, 1.0f);
        surf2Dwrite(
            output,
            output_surface,
            (index % width) * static_cast<int>(sizeof(float4)),
            index / width);
    }
}

class CudaFrameStorage {
public:
    explicit CudaFrameStorage(
        CudaPathStatistics& statistics,
        cudaStream_t shared_stream = nullptr)
        : statistics_(statistics),
          stream_(shared_stream),
          owns_stream_(shared_stream == nullptr) {
        if (owns_stream_) {
            check_cuda(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");
        }
    }

    ~CudaFrameStorage() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
        }
        destroy_graph();
        if (stream_ && owns_stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    void reset(
        int width,
        int height,
        std::uint64_t seed_offset,
        std::size_t requested_wavefront_capacity = 0) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("CUDA framebuffer dimensions must be positive");
        }
        const std::size_t count =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        if (count > static_cast<std::size_t>(INT_MAX)) {
            throw std::invalid_argument("CUDA framebuffer contains too many pixels");
        }
        width_ = width;
        height_ = height;
        const std::size_t wavefront_capacity =
            requested_wavefront_capacity == 0
            ? count
            : std::clamp<std::size_t>(
                requested_wavefront_capacity,
                1,
                count);
        reset_timer_.update(statistics_.reset_milliseconds);
        const bool record_timing = !reset_timer_.pending();
        if (record_timing) {
            reset_timer_.begin(stream_);
        }
        accumulation_.resize(count, statistics_);
        random_states_.resize(count, statistics_);
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        diagnostic_counters_.resize_exact(
            kDiagnosticCounterCount,
            statistics_);
        check_cuda(
            cudaMemsetAsync(
                diagnostic_counters_.get(),
                0,
                kDiagnosticCounterCount * sizeof(unsigned long long),
                stream_),
            "clear CUDA benchmark diagnostic counters");
#endif
        ensure_wavefront_capacity(
            wavefront_capacity,
            requested_wavefront_capacity != 0);
#if !defined(RENDERER_CUDA_SANITIZER_FALLBACK)
        if (!graph_exec_ ||
            wavefront_capacity >
                static_cast<std::size_t>(graph_capacity_)) {
            rebuild_graph(static_cast<int>(wavefront_capacity));
        }
#endif
        const int block_count = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        initialize_frame_kernel<<<block_count, kThreadsPerBlock, 0, stream_>>>(
            accumulation_.get(),
            random_states_.get(),
            width,
            height,
            seed_offset,
            error_code_);
        check_cuda(cudaGetLastError(), "initialize_frame_kernel launch");
        if (record_timing) {
            reset_timer_.end(stream_);
        }
        samples_ = 0;
    }

    void render_sample(
        const DScene& scene,
        const Camera& camera,
        const PathRenderSettings& path_settings,
        cudaSurfaceObject_t output_surface = 0) {
        render_samples(scene, camera, 1, path_settings, output_surface);
    }

    void render_samples(
        const DScene& scene,
        const Camera& camera,
        int sample_count,
        const PathRenderSettings& path_settings,
        cudaSurfaceObject_t output_surface = 0) {
        render_work(
            scene,
            camera,
            sample_count,
            0,
            static_cast<int>(accumulation_.size()),
            std::clamp(path_settings.max_bounces, 1, kMaxPathBounces),
            width_,
            height_,
            path_settings,
            output_surface,
            true);
    }

    void render_region(
        const DScene& scene,
        const Camera& camera,
        int pixel_offset,
        int pixel_count,
        int max_bounces,
        int output_width,
        int output_height,
        const PathRenderSettings& path_settings,
        cudaSurfaceObject_t output_surface = 0) {
        render_work(
            scene,
            camera,
            1,
            pixel_offset,
            pixel_count,
            max_bounces,
            output_width,
            output_height,
            path_settings,
            output_surface,
            false,
            true);
    }

    void render_region_quantum(
        const DScene& scene,
        const Camera& camera,
        int pixel_offset,
        int pixel_count,
        int max_bounces,
        int output_width,
        int output_height,
        const PathRenderSettings& path_settings) {
        if (pixel_count <= 0) {
            throw std::invalid_argument(
                "CUDA wavefront quantum must contain pixels");
        }
        const int chunk_capacity = static_cast<int>(
            std::min<std::size_t>(
                wavefront_capacity_pixels_,
                static_cast<std::size_t>(INT_MAX)));
        if (chunk_capacity <= 0) {
            throw std::logic_error(
                "CUDA wavefront quantum arena is empty");
        }
        update_trace_timing();
        const bool record_timing = !trace_timer_.pending();
        if (record_timing) {
            trace_timer_.begin(stream_);
        }
        int submitted = 0;
        while (submitted < pixel_count) {
            const int chunk =
                std::min(chunk_capacity, pixel_count - submitted);
            render_work(
                scene,
                camera,
                1,
                pixel_offset + submitted,
                chunk,
                max_bounces,
                output_width,
                output_height,
                path_settings,
                0,
                false,
                false);
            submitted += chunk;
        }
        if (record_timing) {
            trace_timer_.end(stream_);
        }
    }

    void complete_sample() {
        ++samples_;
    }

    void present_surface(cudaSurfaceObject_t output_surface) {
        if (output_surface == 0) {
            throw std::invalid_argument(
                "CUDA frame presentation requires a surface");
        }
        if (samples_ <= 0) {
            throw std::logic_error(
                "CUDA frame presentation requires a completed sample");
        }
        update_presentation_timing();
        const bool record_timing = !presentation_timer_.pending();
        if (record_timing) {
            presentation_timer_.begin(stream_);
        }
        const int pixel_count =
            static_cast<int>(accumulation_.size());
        const int capacity_blocks =
            (pixel_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const int block_count = occupancy_grid_size(
            reinterpret_cast<void*>(present_frame_kernel),
            capacity_blocks);
        present_frame_kernel<<<
            block_count,
            kThreadsPerBlock,
            0,
            stream_>>>(
                accumulation_.get(),
                width_,
                height_,
                samples_,
                output_surface);
        check_cuda(cudaGetLastError(), "present_frame_kernel launch");
        if (record_timing) {
            presentation_timer_.end(stream_);
        }
    }

private:
    void render_work(
        const DScene& scene,
        const Camera& camera,
        int sample_count,
        int pixel_offset,
        int pixel_count,
        int max_bounces,
        int output_width,
        int output_height,
        const PathRenderSettings& path_settings,
        cudaSurfaceObject_t output_surface,
        bool completes_sample,
        bool record_trace = true) {
        if (sample_count <= 0) {
            throw std::invalid_argument(
                "CUDA wavefront sample batch must be positive");
        }
        if (pixel_offset < 0 || pixel_count <= 0 ||
            static_cast<std::size_t>(pixel_offset) +
                    static_cast<std::size_t>(pixel_count) >
                accumulation_.size()) {
            throw std::invalid_argument(
                "CUDA wavefront work region is invalid");
        }
        if (static_cast<std::size_t>(pixel_count) >
            wavefront_capacity_pixels_) {
            throw std::invalid_argument(
                "CUDA wavefront work region exceeds arena capacity");
        }
        if (max_bounces <= 0 || max_bounces > kMaxPathBounces) {
            throw std::invalid_argument(
                "CUDA wavefront bounce limit is invalid");
        }
        if (output_width <= 0 || output_height <= 0) {
            throw std::invalid_argument(
                "CUDA output dimensions must be positive");
        }
        const int roulette_start_bounce = std::clamp(
            path_settings.russian_roulette_start_bounce,
            1,
            kMaxPathBounces);
        const float roulette_min_probability = std::clamp(
            path_settings.russian_roulette_min_probability,
            0.01f,
            1.0f);
        const float roulette_max_probability = std::clamp(
            path_settings.russian_roulette_max_probability,
            roulette_min_probability,
            1.0f);
        const DCamera packed_camera{
            to_device(camera.eye()),
            to_device(camera.forward()),
            to_device(camera.right()),
            to_device(camera.up()),
            camera.viewport_width(),
            camera.viewport_height()};
        check_completed_error();
        const DFrameParameters parameters{
            scene,
            packed_camera,
            accumulation_.get(),
            sample_radiance_,
            random_states_.get(),
            width_,
            height_,
            samples_,
            sample_count,
            batch_sample_index_,
            pixel_offset,
            pixel_count,
            static_cast<int>(wavefront_capacity_pixels_),
            max_bounces,
            roulette_start_bounce,
            roulette_min_probability,
            roulette_max_probability,
            output_width,
            output_height,
            error_code_,
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
            diagnostic_counters_.get(),
#endif
            output_surface};
        update_frame_parameters_kernel<<<1, 1, 0, stream_>>>(
            frame_parameters_,
            parameters);
        check_cuda(
            cudaGetLastError(),
            "update_frame_parameters_kernel launch");
        if (record_trace) {
            update_trace_timing();
        }
        const bool record_timing =
            record_trace && !trace_timer_.pending();
        if (record_timing) {
            trace_timer_.begin(stream_);
        }
#if defined(RENDERER_CUDA_SANITIZER_FALLBACK)
        launch_sanitizer_fixed_topology(sample_count);
#else
        check_cuda(
            cudaGraphLaunch(graph_exec_, stream_),
            "cudaGraphLaunch wavefront path");
#endif
        if (record_timing) {
            trace_timer_.end(stream_);
        }
        if (completes_sample) {
            samples_ += sample_count;
        }
    }

public:
    void synchronize_and_check_errors() {
        check_cuda(cudaStreamSynchronize(stream_), "path rendering stream synchronize");
        throw_if_wavefront_error();
        update_timings();
    }

    std::vector<Color> download_pixels(bool use_pinned_staging = true) {
        return download_region_pixels(
            0,
            static_cast<int>(accumulation_.size()),
            samples_,
            use_pinned_staging);
    }

    CudaPathDiagnosticProfile download_diagnostic_profile() {
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        std::array<unsigned long long, kDiagnosticCounterCount> counters{};
        diagnostic_counters_.download(counters.data(), stream_);
        synchronize_and_check_errors();
        CudaPathDiagnosticProfile profile;
        profile.primary_rays = counters[kDiagnosticPrimaryRays];
        profile.continuation_rays = counters[kDiagnosticContinuationRays];
        profile.directional_shadow_rays =
            counters[kDiagnosticDirectionalShadowRays];
        profile.point_shadow_rays = counters[kDiagnosticPointShadowRays];
        profile.spot_shadow_rays = counters[kDiagnosticSpotShadowRays];
        profile.emissive_shadow_rays =
            counters[kDiagnosticEmissiveShadowRays];
        profile.environment_shadow_rays =
            counters[kDiagnosticEnvironmentShadowRays];
        profile.primary_hits = counters[kDiagnosticPrimaryHits];
        profile.primary_misses = counters[kDiagnosticPrimaryMisses];
        for (std::size_t bounce = 0;
             bounce < CudaPathDiagnosticProfile::kBounceBins;
             ++bounce) {
            profile.rays_by_bounce[bounce] =
                counters[kDiagnosticRaysByBounce + bounce];
            profile.termination_by_bounce[bounce] =
                counters[kDiagnosticTerminationByBounce + bounce];
        }
        return profile;
#else
        throw std::runtime_error(
            "CUDA path diagnostics require the instrumented benchmark target");
#endif
    }

    std::vector<Color> download_region_pixels(
        int pixel_offset,
        int pixel_count,
        int sample_count,
        bool use_pinned_staging = true) {
        if (pixel_offset < 0 || pixel_count <= 0 ||
            static_cast<std::size_t>(pixel_offset) +
                    static_cast<std::size_t>(pixel_count) >
                accumulation_.size()) {
            throw std::invalid_argument(
                "CUDA download region is invalid");
        }
        const std::size_t count =
            static_cast<std::size_t>(pixel_count);
        resolved_.resize(count, statistics_);
        std::vector<DVec3> pageable_pixels;
        DVec3* packed_pixels = nullptr;
        if (use_pinned_staging) {
            pinned_pixels_.resize(count, statistics_);
            packed_pixels = pinned_pixels_.get();
        } else {
            pageable_pixels.resize(count);
            packed_pixels = pageable_pixels.data();
        }
        const int block_count =
            static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        resolve_frame_kernel<<<block_count, kThreadsPerBlock, 0, stream_>>>(
            accumulation_.get(),
            resolved_.get(),
            static_cast<int>(count),
            pixel_offset,
            sample_count);
        check_cuda(cudaGetLastError(), "resolve_frame_kernel launch");
        resolved_.download(packed_pixels, stream_);
        ++statistics_.framebuffer_downloads;
        synchronize_and_check_errors();
        std::vector<Color> pixels;
        pixels.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const DVec3 value = packed_pixels[index];
            pixels.emplace_back(value.x, value.y, value.z);
        }
        return pixels;
    }

    cudaStream_t stream() const { return stream_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int samples() const { return samples_; }
    std::uint64_t trace_generation() const {
        return trace_generation_;
    }
    float last_trace_milliseconds() const {
        return last_trace_milliseconds_;
    }

    void update_timings() {
        reset_timer_.update(statistics_.reset_milliseconds);
        update_trace_timing();
        update_presentation_timing();
        check_completed_error();
    }

private:
    bool update_trace_timing() {
        float milliseconds = last_trace_milliseconds_;
        if (!trace_timer_.update(milliseconds)) {
            return false;
        }
        last_trace_milliseconds_ = milliseconds;
        statistics_.trace_milliseconds = milliseconds;
        ++trace_generation_;
        return true;
    }

    bool update_presentation_timing() {
        float milliseconds =
            statistics_.presentation_milliseconds;
        if (!presentation_timer_.update(milliseconds)) {
            return false;
        }
        statistics_.presentation_milliseconds = milliseconds;
        return true;
    }

#if defined(RENDERER_CUDA_SANITIZER_FALLBACK)
    void launch_sanitizer_fixed_topology(int sample_count) {
        const int capacity =
            static_cast<int>(wavefront_capacity_pixels_);
        const int block_count =
            (capacity + kThreadsPerBlock - 1) / kThreadsPerBlock;
        prepare_sample_batch_kernel<<<1, 1, 0, stream_>>>(
            batch_sample_index_);
        for (int sample = 0; sample < sample_count; ++sample) {
            prepare_sample_kernel<<<
                block_count,
                kThreadsPerBlock,
                0,
                stream_>>>(
                frame_parameters_,
                path_queues_[0],
                path_counts_,
                path_counts_ + 1,
                bounce_index_,
                0,
                0);
            for (int bounce = 0;
                 bounce < kMaxPathBounces;
                 ++bounce) {
                intersect_wavefront_kernel<<<
                    block_count,
                    kThreadsPerBlock,
                    0,
                    stream_>>>(
                    frame_parameters_,
                    path_queues_[0],
                    path_queues_[1],
                    path_counts_,
                    bounce_index_,
                    hits_);
                shade_wavefront_kernel<<<
                    block_count,
                    kThreadsPerBlock,
                    0,
                    stream_>>>(
                    frame_parameters_,
                    path_queues_[0],
                    path_queues_[1],
                    path_counts_,
                    bounce_index_,
                    hits_,
                    shading_records_);
                sample_emissive_lights_kernel<<<
                    block_count,
                    kThreadsPerBlock,
                    0,
                    stream_>>>(
                    frame_parameters_,
                    shading_records_,
                    shadow_tasks_,
                    path_counts_,
                    bounce_index_);
                direct_visibility_kernel<<<
                    block_count,
                    kThreadsPerBlock,
                    0,
                    stream_>>>(
                    frame_parameters_,
                    shading_records_,
                    path_counts_,
                    bounce_index_,
                    shadow_tasks_);
                advance_wavefront_kernel<<<1, 1, 0, stream_>>>(
                    frame_parameters_,
                    path_counts_,
                    bounce_index_,
                    0);
            }
            finalize_sample_kernel<<<
                block_count,
                kThreadsPerBlock,
                0,
                stream_>>>(frame_parameters_);
            advance_sample_batch_kernel<<<1, 1, 0, stream_>>>(
                frame_parameters_,
                0);
        }
        check_cuda(
            cudaGetLastError(),
            "CUDA sanitizer fixed-topology launch");
    }
#endif

    void ensure_wavefront_capacity(
        std::size_t count,
        bool exact_capacity) {
        if ((!exact_capacity &&
             count <= wavefront_capacity_pixels_) ||
            (exact_capacity &&
             count == wavefront_capacity_pixels_)) {
            return;
        }
        check_cuda(
            cudaStreamSynchronize(stream_),
            "wavefront arena resize synchronize");
        destroy_graph();

        struct ArenaLayout {
            std::size_t sample_radiance = 0;
            std::size_t first_paths = 0;
            std::size_t second_paths = 0;
            std::size_t hits = 0;
            std::size_t shading_records = 0;
            std::size_t shadow_tasks = 0;
            std::size_t path_counts = 0;
            std::size_t bounce_index = 0;
            std::size_t batch_sample_index = 0;
            std::size_t error_code = 0;
            std::size_t frame_parameters = 0;
            std::size_t total_bytes = 0;
        } layout;

        std::size_t offset = 0;
        const auto reserve_region = [&offset](
                                        std::size_t& region_offset,
                                        std::size_t alignment,
                                        std::size_t element_size,
                                        std::size_t element_count) {
            const std::size_t aligned =
                (offset + alignment - 1) & ~(alignment - 1);
            if (element_count > (SIZE_MAX - aligned) / element_size) {
                throw std::overflow_error("CUDA wavefront arena is too large");
            }
            region_offset = aligned;
            offset = aligned + element_size * element_count;
        };
        reserve_region(
            layout.sample_radiance,
            alignof(DVec3),
            sizeof(DVec3),
            count);
        reserve_region(
            layout.first_paths, alignof(DPathState), sizeof(DPathState), count);
        reserve_region(
            layout.second_paths, alignof(DPathState), sizeof(DPathState), count);
        reserve_region(
            layout.hits, alignof(DWavefrontHit), sizeof(DWavefrontHit), count);
        reserve_region(
            layout.shading_records,
            alignof(DShadingRecord),
            sizeof(DShadingRecord),
            count);
        reserve_region(
            layout.shadow_tasks,
            alignof(DShadowTask),
            sizeof(DShadowTask),
            count);
        reserve_region(layout.path_counts, alignof(int), sizeof(int), 2);
        reserve_region(layout.bounce_index, alignof(int), sizeof(int), 1);
        reserve_region(
            layout.batch_sample_index, alignof(int), sizeof(int), 1);
        reserve_region(layout.error_code, alignof(int), sizeof(int), 1);
        reserve_region(
            layout.frame_parameters,
            alignof(DFrameParameters),
            sizeof(DFrameParameters),
            1);
        layout.total_bytes = offset;

        if (exact_capacity) {
            wavefront_arena_.resize_exact(
                layout.total_bytes,
                statistics_);
        } else {
            wavefront_arena_.resize(
                layout.total_bytes,
                statistics_);
        }
        unsigned char* const base = wavefront_arena_.get();
        sample_radiance_ =
            reinterpret_cast<DVec3*>(base + layout.sample_radiance);
        path_queues_[0] =
            reinterpret_cast<DPathState*>(base + layout.first_paths);
        path_queues_[1] =
            reinterpret_cast<DPathState*>(base + layout.second_paths);
        hits_ = reinterpret_cast<DWavefrontHit*>(base + layout.hits);
        shading_records_ =
            reinterpret_cast<DShadingRecord*>(base + layout.shading_records);
        shadow_tasks_ =
            reinterpret_cast<DShadowTask*>(base + layout.shadow_tasks);
        path_counts_ = reinterpret_cast<int*>(base + layout.path_counts);
        bounce_index_ = reinterpret_cast<int*>(base + layout.bounce_index);
        batch_sample_index_ =
            reinterpret_cast<int*>(base + layout.batch_sample_index);
        error_code_ = reinterpret_cast<int*>(base + layout.error_code);
        frame_parameters_ =
            reinterpret_cast<DFrameParameters*>(base + layout.frame_parameters);
        wavefront_capacity_pixels_ = count;
    }

    void rebuild_graph(int capacity) {
        if (capacity <= 0) {
            throw std::invalid_argument(
                "CUDA wavefront graph capacity must be positive");
        }
        check_cuda(
            cudaStreamSynchronize(stream_),
            "wavefront graph rebuild synchronize");
        destroy_graph();
        const int block_count =
            (capacity + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const int prepare_block_count = occupancy_grid_size(
            reinterpret_cast<void*>(prepare_sample_kernel),
            block_count);
        const int intersect_block_count = block_count;
        const int emissive_block_count = block_count;
        const int direct_block_count = block_count;
        const int finalize_block_count = occupancy_grid_size(
            reinterpret_cast<void*>(finalize_sample_kernel),
            block_count);
        check_cuda(
            cudaGraphCreate(&graph_, 0),
            "cudaGraphCreate wavefront graph");
        check_cuda(
            cudaGraphConditionalHandleCreate(
                &batch_conditional_handle_,
                graph_,
                1U,
                cudaGraphCondAssignDefault),
            "cudaGraphConditionalHandleCreate sample batch");
        check_cuda(
            cudaGraphConditionalHandleCreate(
                &wavefront_conditional_handle_,
                graph_,
                0U,
                0),
            "cudaGraphConditionalHandleCreate wavefront loop");
        check_cuda(
            cudaGraphConditionalHandleCreate(
                &emissive_conditional_handle_,
                graph_,
                0U,
                0),
            "cudaGraphConditionalHandleCreate emissive sampling");

        DFrameParameters* frame_parameters = frame_parameters_;
        DPathState* first_paths = path_queues_[0];
        DPathState* second_paths = path_queues_[1];
        int* path_counts = path_counts_;
        int* primary_count = path_counts;
        int* secondary_count = path_counts + 1;
        int* bounce_index = bounce_index_;
        int* batch_sample_index = batch_sample_index_;
        DWavefrontHit* hits = hits_;
        DShadingRecord* shading_records = shading_records_;
        DShadowTask* shadow_tasks = shadow_tasks_;

        void* prepare_batch_arguments[] = {&batch_sample_index};
        cudaGraphNode_t prepare_batch_node = add_kernel_node(
            graph_,
            nullptr,
            reinterpret_cast<void*>(prepare_sample_batch_kernel),
            dim3(1),
            dim3(1),
            prepare_batch_arguments);

        cudaGraphNodeParams batch_conditional_parameters{};
        batch_conditional_parameters.type = cudaGraphNodeTypeConditional;
        batch_conditional_parameters.conditional.handle =
            batch_conditional_handle_;
        batch_conditional_parameters.conditional.type =
            cudaGraphCondTypeWhile;
        batch_conditional_parameters.conditional.size = 1;
        cudaGraphNode_t batch_conditional_node = nullptr;
        check_cuda(
            cudaGraphAddNode(
                &batch_conditional_node,
                graph_,
                &prepare_batch_node,
                nullptr,
                1,
                &batch_conditional_parameters),
            "cudaGraphAddNode sample batch while");
        cudaGraph_t sample_body =
            batch_conditional_parameters.conditional.phGraph_out[0];

        cudaGraphConditionalHandle wavefront_handle =
            wavefront_conditional_handle_;
        cudaGraphConditionalHandle emissive_handle =
            emissive_conditional_handle_;
        void* prepare_arguments[] = {
            &frame_parameters,
            &first_paths,
            &primary_count,
            &secondary_count,
            &bounce_index,
            &wavefront_handle,
            &emissive_handle};
        cudaGraphNode_t prepare_node = add_kernel_node(
            sample_body,
            nullptr,
            reinterpret_cast<void*>(prepare_sample_kernel),
            dim3(prepare_block_count),
            dim3(kThreadsPerBlock),
            prepare_arguments);

        cudaGraphNodeParams conditional_parameters{};
        conditional_parameters.type = cudaGraphNodeTypeConditional;
        conditional_parameters.conditional.handle =
            wavefront_conditional_handle_;
        conditional_parameters.conditional.type = cudaGraphCondTypeWhile;
        conditional_parameters.conditional.size = 1;
        cudaGraphNode_t conditional_node = nullptr;
        check_cuda(
            cudaGraphAddNode(
                &conditional_node,
                sample_body,
                &prepare_node,
                nullptr,
                1,
                &conditional_parameters),
            "cudaGraphAddNode wavefront while");
        cudaGraph_t body = conditional_parameters.conditional.phGraph_out[0];

        void* intersect_arguments[] = {
            &frame_parameters,
            &first_paths,
            &second_paths,
            &path_counts,
            &bounce_index,
            &hits};
        cudaGraphNode_t body_tail = add_kernel_node(
            body,
            nullptr,
            reinterpret_cast<void*>(intersect_wavefront_kernel),
            dim3(intersect_block_count),
            dim3(kThreadsPerBlock),
            intersect_arguments);

        void* shade_arguments[] = {
            &frame_parameters,
            &first_paths,
            &second_paths,
            &path_counts,
            &bounce_index,
            &hits,
            &shading_records};
        body_tail = add_kernel_node(
            body,
            body_tail,
            reinterpret_cast<void*>(shade_wavefront_kernel),
            dim3(block_count),
            dim3(kThreadsPerBlock),
            shade_arguments);

        cudaGraphNodeParams emissive_conditional_parameters{};
        emissive_conditional_parameters.type =
            cudaGraphNodeTypeConditional;
        emissive_conditional_parameters.conditional.handle =
            emissive_conditional_handle_;
        emissive_conditional_parameters.conditional.type =
            cudaGraphCondTypeIf;
        emissive_conditional_parameters.conditional.size = 1;
        cudaGraphNode_t emissive_conditional_node = nullptr;
        check_cuda(
            cudaGraphAddNode(
                &emissive_conditional_node,
                body,
                &body_tail,
                nullptr,
                1,
                &emissive_conditional_parameters),
            "cudaGraphAddNode emissive sampling if");
        cudaGraph_t emissive_body =
            emissive_conditional_parameters.conditional.phGraph_out[0];

        void* emissive_arguments[] = {
            &frame_parameters,
            &shading_records,
            &shadow_tasks,
            &path_counts,
            &bounce_index};
        add_kernel_node(
            emissive_body,
            nullptr,
            reinterpret_cast<void*>(sample_emissive_lights_kernel),
            dim3(emissive_block_count),
            dim3(kThreadsPerBlock),
            emissive_arguments);
        body_tail = emissive_conditional_node;

        void* direct_arguments[] = {
            &frame_parameters,
            &shading_records,
            &path_counts,
            &bounce_index,
            &shadow_tasks};
        body_tail = add_kernel_node(
            body,
            body_tail,
            reinterpret_cast<void*>(direct_visibility_kernel),
            dim3(direct_block_count),
            dim3(kThreadsPerBlock),
            direct_arguments);

        cudaGraphConditionalHandle conditional_handle =
            wavefront_conditional_handle_;
        void* advance_arguments[] = {
            &frame_parameters,
            &path_counts,
            &bounce_index,
            &conditional_handle};
        add_kernel_node(
            body,
            body_tail,
            reinterpret_cast<void*>(advance_wavefront_kernel),
            dim3(1),
            dim3(1),
            advance_arguments);

        void* finalize_arguments[] = {&frame_parameters};
        cudaGraphNode_t finalize_node = add_kernel_node(
            sample_body,
            conditional_node,
            reinterpret_cast<void*>(finalize_sample_kernel),
            dim3(finalize_block_count),
            dim3(kThreadsPerBlock),
            finalize_arguments);

        cudaGraphConditionalHandle batch_handle =
            batch_conditional_handle_;
        void* advance_batch_arguments[] = {
            &frame_parameters,
            &batch_handle};
        add_kernel_node(
            sample_body,
            finalize_node,
            reinterpret_cast<void*>(advance_sample_batch_kernel),
            dim3(1),
            dim3(1),
            advance_batch_arguments);
        check_cuda(
            cudaGraphInstantiate(
                &graph_exec_,
                graph_,
                nullptr,
                nullptr,
                0),
            "cudaGraphInstantiate wavefront graph");
        graph_capacity_ = capacity;
    }

    int occupancy_grid_size(void* function, int capacity_blocks) {
        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        int multiprocessor_count = 0;
        check_cuda(
            cudaDeviceGetAttribute(
                &multiprocessor_count,
                cudaDevAttrMultiProcessorCount,
                device),
            "cudaDeviceGetAttribute multiprocessor count");
        int resident_blocks = 0;
        check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident_blocks,
                function,
                kThreadsPerBlock,
                0),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
        return std::max(
            1,
            std::min(
                capacity_blocks,
                2 * resident_blocks * multiprocessor_count));
    }

    cudaGraphNode_t add_kernel_node(
        cudaGraph_t graph,
        cudaGraphNode_t dependency,
        void* function,
        dim3 grid,
        dim3 block,
        void** arguments) {
        cudaKernelNodeParams parameters{};
        parameters.func = function;
        parameters.gridDim = grid;
        parameters.blockDim = block;
        parameters.kernelParams = arguments;
        cudaGraphNode_t node = nullptr;
        const cudaGraphNode_t* dependencies =
            dependency != nullptr ? &dependency : nullptr;
        check_cuda(
            cudaGraphAddKernelNode(
                &node,
                graph,
                dependencies,
                dependency != nullptr ? 1 : 0,
                &parameters),
            "cudaGraphAddKernelNode wavefront graph");
        return node;
    }

    void destroy_graph() noexcept {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        batch_conditional_handle_ = 0;
        wavefront_conditional_handle_ = 0;
        emissive_conditional_handle_ = 0;
        graph_capacity_ = 0;
    }

    void check_completed_error() {
        if (!stream_ || !error_code_) {
            return;
        }
        const cudaError_t query = cudaStreamQuery(stream_);
        if (query == cudaErrorNotReady) {
            return;
        }
        check_cuda(query, "CUDA wavefront stream query");
        throw_if_wavefront_error();
    }

    void throw_if_wavefront_error() {
        if (!error_code_) {
            return;
        }
        int error = 0;
        check_cuda(
            cudaMemcpy(
                &error,
                error_code_,
                sizeof(error),
                cudaMemcpyDeviceToHost),
            "download CUDA wavefront error");
        if (error == 0) {
            return;
        }
        if (error == 1) {
            throw std::runtime_error("CUDA wavefront path queue overflow");
        }
        if (error == 5) {
            throw std::runtime_error(
                "CUDA BVH4 traversal stack overflow");
        }
        if (error == 6) {
            throw std::runtime_error(
                "CUDA TLAS traversal stack overflow");
        }
        if (error == 7) {
            throw std::runtime_error(
                "CUDA instance or TLAS index is invalid");
        }
        throw std::runtime_error("CUDA wavefront queue count is invalid");
    }

    CudaPathStatistics& statistics_;
    cudaStream_t stream_ = nullptr;
    bool owns_stream_ = false;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaGraphConditionalHandle batch_conditional_handle_ = 0;
    cudaGraphConditionalHandle wavefront_conditional_handle_ = 0;
    cudaGraphConditionalHandle emissive_conditional_handle_ = 0;
    int graph_capacity_ = 0;
    CudaEventTimer reset_timer_;
    CudaEventTimer trace_timer_;
    CudaEventTimer presentation_timer_;
    float last_trace_milliseconds_ = 0.0f;
    std::uint64_t trace_generation_ = 0;
    DeviceBuffer<DVec3> accumulation_;
    DeviceBuffer<DVec3> resolved_;
    DeviceBuffer<DPcgState> random_states_;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    DeviceBuffer<unsigned long long> diagnostic_counters_;
#endif
    DeviceBuffer<unsigned char> wavefront_arena_;
    std::size_t wavefront_capacity_pixels_ = 0;
    DVec3* sample_radiance_ = nullptr;
    DPathState* path_queues_[2]{nullptr, nullptr};
    DWavefrontHit* hits_ = nullptr;
    DShadingRecord* shading_records_ = nullptr;
    DShadowTask* shadow_tasks_ = nullptr;
    int* path_counts_ = nullptr;
    int* bounce_index_ = nullptr;
    int* batch_sample_index_ = nullptr;
    int* error_code_ = nullptr;
    DFrameParameters* frame_parameters_ = nullptr;
    PinnedHostBuffer<DVec3> pinned_pixels_;
    int width_ = 0;
    int height_ = 0;
    int samples_ = 0;
};

}  // namespace

bool cuda_path_backend_compiled() {
    return true;
}

bool cuda_path_backend_available(std::string* reason) {
    return cuda_path_backend_available(0, reason);
}

bool cuda_path_backend_available(int device_id, std::string* reason) {
    return CudaDeviceContext::try_create(device_id, reason).has_value();
}

RenderResult render_cuda_path(
    const RenderSceneSnapshot& snapshot,
    const Camera& camera,
    const RenderSettings& settings) {
    const CudaDeviceContext device_context =
        CudaDeviceContext::create(settings.path.cuda_device);
    device_context.activate();
    Timer timer;
    CudaPathStatistics statistics;
    statistics.device_id = device_context.device_id();
    CudaFrameStorage frame(statistics);
    CudaSceneStorage device_scene(snapshot, frame.stream(), statistics);
    frame.reset(settings.width, settings.height, settings.path.sample_seed_offset);
    const int sample_count = std::max(1, settings.path.samples_per_pixel);
    frame.render_samples(
        device_scene.view(),
        camera,
        sample_count,
        settings.path);
    Image image(settings.width, settings.height);
    image.set_pixels(frame.download_pixels(false));
    return RenderResult{std::move(image), timer.elapsed_seconds()};
}

class CudaPathInteractiveRenderer::Impl {
public:
    explicit Impl(CudaDeviceContext device_context)
        : device_context_(activate(std::move(device_context))),
          frame_(statistics_),
          preview_frame_(statistics_, frame_.stream()) {
        statistics_.device_id = device_context_.device_id();
    }

    void reset(
        const RenderSceneSnapshot& snapshot,
        const RenderSettings& settings) {
        require_device(settings.path.cuda_device);
        scene_ = std::make_unique<CudaSceneStorage>(
            snapshot,
            frame_.stream(),
            statistics_);
        uploaded_source_id_ = snapshot.source_id;
        uploaded_revisions_ = snapshot.revisions;
        has_uploaded_snapshot_ = true;
        frame_.reset(
            settings.width,
            settings.height,
            settings.path.sample_seed_offset);
        reset_interactive_state(
            settings.width,
            settings.height,
            scene_->view().triangle_count);
        preview_has_run_ = false;
        has_progressive_key_ = false;
    }

    void render_next_frame(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) {
        render_next_frame_to_surface(
            snapshot,
            camera,
            settings,
            frame_state,
            0);
        download_current_frame(target);
    }

    void render_next_frame_to_surface(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        CudaSurfaceHandle surface) {
        require_device(settings.path.cuda_device);
        const SceneRevisions current_revisions = snapshot.revisions;
        const SceneChangeSet scene_changes = scene_changes_for_snapshot(
            uploaded_source_id_,
            uploaded_revisions_,
            has_uploaded_snapshot_,
            snapshot,
            frame_state.scene_changes);
        if (!scene_) {
            scene_ = std::make_unique<CudaSceneStorage>(
                snapshot,
                frame_.stream(),
                statistics_);
        } else if (scene_changes != SceneChange::None) {
            scene_->sync(snapshot, scene_changes);
        }
        uploaded_source_id_ = snapshot.source_id;
        uploaded_revisions_ = current_revisions;
        has_uploaded_snapshot_ = true;
        const ProgressiveRenderKey progressive_key =
            ProgressiveRenderKey::from(
                camera,
                settings,
                frame_state.automatic_interaction_quality,
                snapshot.source_id,
                current_revisions);
        const bool progressive_key_changed =
            !has_progressive_key_ ||
            !(progressive_key == progressive_key_);
        // Single invalidation predicate: both the accumulation reset and the
        // interaction/preview fast path below use the same five conditions.
        const bool interaction_changed =
            progressive_key_changed ||
            frame_state.camera_changed ||
            scene_changes != SceneChange::None ||
            frame_state.framebuffer_resized ||
            frame_state.reset_requested;
        if (interaction_changed) {
            frame_.reset(
                settings.width,
                settings.height,
                settings.path.sample_seed_offset,
                frame_state.automatic_interaction_quality
                    ? native_wavefront_capacity(
                        settings.width,
                        settings.height)
                    : static_cast<std::size_t>(settings.width) *
                        static_cast<std::size_t>(settings.height));
            reset_interactive_state(
                settings.width,
                settings.height,
                scene_->view().triangle_count);
            automatic_quality_enabled_ =
                frame_state.automatic_interaction_quality;
            progressive_key_ = progressive_key;
            has_progressive_key_ = true;
        }
        output_width_ = settings.width;
        output_height_ = settings.height;
        native_sweep_published_this_frame_ = false;
        statistics_.presentation_updated = false;

        if (!frame_state.automatic_interaction_quality) {
            active_frame_is_preview_ = false;
            frame_.render_sample(
                scene_->view(),
                camera,
                settings.path,
                static_cast<cudaSurfaceObject_t>(surface));
            statistics_.work_mode = CudaPathWorkMode::FullFrame;
            statistics_.internal_width = settings.width;
            statistics_.internal_height = settings.height;
            statistics_.tile_y = 0;
            statistics_.tile_rows = settings.height;
            statistics_.sweep_progress = 1.0f;
            statistics_.presentation_updated = true;
            scene_->update_timing();
            return;
        }

        if (interaction_changed) {
            idle_frames_ = 0;
            preview_dirty_ = true;
            if (!preview_has_run_ ||
                scene_changes != SceneChange::None) {
                preview_scale_tier_ =
                    initial_preview_scale_tier(
                        scene_->view().triangle_count);
                preview_trace_ema_ = 0.0f;
                slow_preview_frames_ = 0;
                fast_preview_frames_ = 0;
            }
        } else if (idle_frames_ < kIdleFramesBeforeNative) {
            ++idle_frames_;
        }

        frame_.update_timings();
        preview_frame_.update_timings();
        update_adaptive_work_size();
        if (idle_frames_ < kIdleFramesBeforeNative) {
            render_interaction_preview(
                camera,
                settings,
                static_cast<cudaSurfaceObject_t>(surface));
        } else {
            render_native_quantum(
                camera,
                settings,
                static_cast<cudaSurfaceObject_t>(surface));
        }
        scene_->update_timing();
    }

    void download_current_frame(Framebuffer& target) {
        const auto download_preview = [&]() {
            const std::vector<Color> preview =
                preview_frame_.download_pixels();
            if (target.width() != output_width_ ||
                target.height() != output_height_) {
                target.resize(output_width_, output_height_);
            }
            for (int y = 0; y < output_height_; ++y) {
                const int source_y = std::min(
                    preview_frame_.height() - 1,
                    y * preview_frame_.height() / output_height_);
                for (int x = 0; x < output_width_; ++x) {
                    const int source_x = std::min(
                        preview_frame_.width() - 1,
                        x * preview_frame_.width() / output_width_);
                    target.set_pixel(
                        x,
                        y,
                        preview[static_cast<std::size_t>(source_y) *
                                    static_cast<std::size_t>(
                                        preview_frame_.width()) +
                                static_cast<std::size_t>(source_x)]);
                }
            }
            host_presentation_initialized_ = true;
        };
        if (active_frame_is_preview_) {
            download_preview();
            return;
        }
        if (last_work_mode_ == CudaPathWorkMode::NativeTile &&
            automatic_quality_enabled_) {
            if (native_sweep_published_this_frame_) {
                if (target.width() != frame_.width() ||
                    target.height() != frame_.height()) {
                    target.resize(frame_.width(), frame_.height());
                }
                target.set_pixels(frame_.download_pixels());
                host_presentation_initialized_ = true;
            } else if (!host_presentation_initialized_ &&
                       preview_has_run_) {
                download_preview();
            }
            return;
        }
        if (target.width() != frame_.width() ||
            target.height() != frame_.height()) {
            target.resize(frame_.width(), frame_.height());
        }
        target.set_pixels(frame_.download_pixels());
        host_presentation_initialized_ = true;
    }

    int accumulated_samples() const {
        return frame_.samples();
    }

    CudaStreamHandle stream_handle() const {
        return reinterpret_cast<CudaStreamHandle>(frame_.stream());
    }

    const CudaPathStatistics& statistics() const {
        return statistics_;
    }

    void refresh_statistics() {
        frame_.update_timings();
        preview_frame_.update_timings();
        if (scene_) {
            scene_->update_timing();
        }
    }

    CudaPathDiagnosticProfile download_diagnostic_profile() {
        return frame_.download_diagnostic_profile();
    }

    void set_presentation_state(bool interop_active, bool fallback_active) {
        statistics_.interop_active = interop_active;
        statistics_.fallback_active = fallback_active;
    }

private:
    static CudaDeviceContext activate(CudaDeviceContext context) {
        context.activate();
        return context;
    }

    void require_device(int requested_device) {
        if (requested_device != device_context_.device_id()) {
            throw std::runtime_error(
                "CUDA renderer belongs to device " +
                std::to_string(device_context_.device_id()) +
                " but settings request device " +
                std::to_string(requested_device));
        }
        device_context_.activate();
        statistics_.device_id = device_context_.device_id();
    }

    struct ProgressiveRenderKey {
        Vec3 eye = Vec3::Zero();
        Vec3 forward = Vec3::Zero();
        Vec3 right = Vec3::Zero();
        Vec3 up = Vec3::Zero();
        float viewport_width = 0.0f;
        float viewport_height = 0.0f;
        float roulette_min = 0.0f;
        float roulette_max = 0.0f;
        std::uint64_t seed = 0;
        int device_id = -1;
        int width = 0;
        int height = 0;
        int max_bounces = 0;
        int roulette_start = 0;
        bool automatic_quality = false;
        std::uint64_t source_id = 0;
        SceneRevisions revisions;

        static ProgressiveRenderKey from(
            const Camera& camera,
            const RenderSettings& settings,
            bool automatic_quality_value,
            std::uint64_t source_id_value,
            const SceneRevisions& revision_value) {
            ProgressiveRenderKey result;
            result.eye = camera.eye();
            result.forward = camera.forward();
            result.right = camera.right();
            result.up = camera.up();
            result.viewport_width = camera.viewport_width();
            result.viewport_height = camera.viewport_height();
            result.roulette_min =
                settings.path.russian_roulette_min_probability;
            result.roulette_max =
                settings.path.russian_roulette_max_probability;
            result.seed = settings.path.sample_seed_offset;
            result.device_id = settings.path.cuda_device;
            result.width = settings.width;
            result.height = settings.height;
            result.max_bounces = settings.path.max_bounces;
            result.roulette_start =
                settings.path.russian_roulette_start_bounce;
            result.automatic_quality = automatic_quality_value;
            result.source_id = source_id_value;
            result.revisions = revision_value;
            return result;
        }

        bool operator==(const ProgressiveRenderKey& other) const {
            return eye == other.eye &&
                forward == other.forward &&
                right == other.right &&
                up == other.up &&
                viewport_width == other.viewport_width &&
                viewport_height == other.viewport_height &&
                roulette_min == other.roulette_min &&
                roulette_max == other.roulette_max &&
                seed == other.seed &&
                device_id == other.device_id &&
                width == other.width &&
                height == other.height &&
                max_bounces == other.max_bounces &&
                roulette_start == other.roulette_start &&
                automatic_quality == other.automatic_quality &&
                source_id == other.source_id &&
                revisions == other.revisions;
        }
    };

    static constexpr int kIdleFramesBeforeNative = 8;
    static constexpr int kPreviewBounceLimit = 2;
    static constexpr int kNativeArenaRows = 128;
    static constexpr int kMaximumQuantumRows = 512;
    static constexpr float kNativeTargetMilliseconds = 12.0f;
    static constexpr float kNativeHardLimitMilliseconds = 15.0f;

    static std::size_t native_wavefront_capacity(int width, int height) {
        return static_cast<std::size_t>(width) *
            static_cast<std::size_t>(
                std::min(height, kNativeArenaRows));
    }

    static int initial_preview_scale_tier(
        int triangle_count) {
        if (triangle_count >= 1'000'000) {
            return 3;
        }
        if (triangle_count >= 20'000) {
            return 2;
        }
        return 0;
    }

    static int initial_native_quantum_rows(int triangle_count) {
        return triangle_count >= 1'000'000 ? 32 : 64;
    }

    void reset_interactive_state(
        int width,
        int height,
        int triangle_count) {
        output_width_ = width;
        output_height_ = height;
        idle_frames_ = 0;
        tile_y_ = 0;
        quantum_rows_ = std::min(
            initial_native_quantum_rows(triangle_count),
            height);
        native_quantum_row_limit_ =
            triangle_count >= 1'000'000
            ? 80
            : kMaximumQuantumRows;
        last_native_quantum_rows_ = 0;
        last_native_quantum_was_tail_ = false;
        native_ms_per_row_ema_ = 0.0f;
        consumed_native_trace_generation_ =
            frame_.trace_generation();
        consumed_preview_trace_generation_ =
            preview_frame_.trace_generation();
        preview_dirty_ = true;
        active_frame_is_preview_ = false;
        native_sweep_published_this_frame_ = false;
        host_presentation_initialized_ = false;
        last_work_mode_ = CudaPathWorkMode::FullFrame;
        sweep_started_ = std::chrono::steady_clock::now();
        statistics_.internal_width = width;
        statistics_.internal_height = height;
        statistics_.tile_y = 0;
        statistics_.tile_rows = quantum_rows_;
        statistics_.sweep_progress = 0.0f;
        statistics_.complete_sweeps_per_second = 0.0f;
        statistics_.presentation_updated = false;
    }

    void update_adaptive_work_size() {
        if (preview_frame_.trace_generation() !=
            consumed_preview_trace_generation_) {
            consumed_preview_trace_generation_ =
                preview_frame_.trace_generation();
            const float elapsed =
                preview_frame_.last_trace_milliseconds();
            if (elapsed > 0.0f && std::isfinite(elapsed)) {
                preview_trace_ema_ = preview_trace_ema_ > 0.0f
                    ? preview_trace_ema_ * 0.4f + elapsed * 0.6f
                    : elapsed;
                if (preview_trace_ema_ > 12.0f) {
                    ++slow_preview_frames_;
                    fast_preview_frames_ = 0;
                    if (slow_preview_frames_ >= 2 &&
                        preview_scale_tier_ + 1 <
                            static_cast<int>(preview_scales_.size())) {
                        ++preview_scale_tier_;
                        slow_preview_frames_ = 0;
                        preview_dirty_ = true;
                    }
                } else if (preview_trace_ema_ < 8.0f) {
                    ++fast_preview_frames_;
                    slow_preview_frames_ = 0;
                    if (fast_preview_frames_ >= 30 &&
                        preview_scale_tier_ > 0) {
                        --preview_scale_tier_;
                        fast_preview_frames_ = 0;
                        preview_dirty_ = true;
                    }
                } else {
                    slow_preview_frames_ = 0;
                    fast_preview_frames_ = 0;
                }
            }
        }

        if (frame_.trace_generation() ==
            consumed_native_trace_generation_) {
            return;
        }
        consumed_native_trace_generation_ =
            frame_.trace_generation();
        const float elapsed = frame_.last_trace_milliseconds();
        if (last_native_quantum_rows_ <= 0 ||
            last_native_quantum_was_tail_ ||
            !(elapsed > 0.0f) ||
            !std::isfinite(elapsed)) {
            return;
        }
        if (elapsed > kNativeHardLimitMilliseconds) {
            quantum_rows_ =
                std::max(1, last_native_quantum_rows_ / 2);
            return;
        }
        const float milliseconds_per_row =
            elapsed /
            static_cast<float>(last_native_quantum_rows_);
        native_ms_per_row_ema_ =
            native_ms_per_row_ema_ > 0.0f
            ? native_ms_per_row_ema_ * 0.75f +
                milliseconds_per_row * 0.25f
            : milliseconds_per_row;
        int desired_rows = std::clamp(
            static_cast<int>(std::lround(
                kNativeTargetMilliseconds /
                native_ms_per_row_ema_)),
            1,
            native_quantum_row_limit_);
        if (desired_rows > kNativeArenaRows) {
            const int chunk_count = std::clamp(
                static_cast<int>(std::lround(
                    static_cast<float>(desired_rows) /
                    static_cast<float>(kNativeArenaRows))),
                1,
                std::max(
                    1,
                    native_quantum_row_limit_ /
                        kNativeArenaRows));
            desired_rows = chunk_count * kNativeArenaRows;
        }
        const int minimum_rows =
            std::max(1, last_native_quantum_rows_ / 2);
        const int maximum_rows = std::min(
            native_quantum_row_limit_,
            last_native_quantum_rows_ * 2);
        quantum_rows_ = std::clamp(
            desired_rows,
            minimum_rows,
            maximum_rows);
    }

    void render_interaction_preview(
        const Camera& camera,
        const RenderSettings& settings,
        cudaSurfaceObject_t surface) {
        const float scale =
            preview_scales_[static_cast<std::size_t>(preview_scale_tier_)];
        const int preview_width = std::max(
            1,
            static_cast<int>(std::lround(
                static_cast<float>(settings.width) * scale)));
        const int preview_height = std::max(
            1,
            static_cast<int>(std::lround(
                static_cast<float>(settings.height) * scale)));
        if (preview_dirty_ ||
            preview_frame_.width() != preview_width ||
            preview_frame_.height() != preview_height) {
            preview_frame_.reset(
                preview_width,
                preview_height,
                settings.path.sample_seed_offset);
            preview_dirty_ = false;
        }
        preview_frame_.render_region(
            scene_->view(),
            camera,
            0,
            preview_width * preview_height,
            std::min(
                kPreviewBounceLimit,
                std::clamp(settings.path.max_bounces, 1, kMaxPathBounces)),
            settings.width,
            settings.height,
            settings.path,
            surface);
        preview_frame_.complete_sample();
        preview_has_run_ = true;
        active_frame_is_preview_ = true;
        last_work_mode_ = CudaPathWorkMode::InteractionPreview;
        statistics_.work_mode = last_work_mode_;
        statistics_.internal_width = preview_width;
        statistics_.internal_height = preview_height;
        statistics_.tile_y = 0;
        statistics_.tile_rows = preview_height;
        statistics_.sweep_progress = 0.0f;
        statistics_.presentation_updated = true;
    }

    void render_native_quantum(
        const Camera& camera,
        const RenderSettings& settings,
        cudaSurfaceObject_t surface) {
        const int rows = std::min(
            quantum_rows_,
            settings.height - tile_y_);
        last_native_quantum_rows_ = rows;
        last_native_quantum_was_tail_ =
            rows < quantum_rows_;
        frame_.render_region_quantum(
            scene_->view(),
            camera,
            tile_y_ * settings.width,
            rows * settings.width,
            std::clamp(settings.path.max_bounces, 1, kMaxPathBounces),
            settings.width,
            settings.height,
            settings.path);
        frame_.synchronize_and_check_errors();
        active_frame_is_preview_ = false;
        last_work_mode_ = CudaPathWorkMode::NativeTile;
        tile_y_ += rows;
        if (tile_y_ >= settings.height) {
            frame_.complete_sample();
            if (surface != 0) {
                frame_.present_surface(surface);
            }
            native_sweep_published_this_frame_ = true;
            statistics_.presentation_updated = true;
            tile_y_ = 0;
            const auto completed_at = std::chrono::steady_clock::now();
            const float seconds =
                std::chrono::duration<float>(
                    completed_at - sweep_started_).count();
            if (seconds > 0.0f) {
                statistics_.complete_sweeps_per_second =
                    1.0f / seconds;
            }
            sweep_started_ = completed_at;
        }
        statistics_.work_mode = last_work_mode_;
        statistics_.internal_width = settings.width;
        statistics_.internal_height = rows;
        statistics_.tile_y = tile_y_;
        statistics_.tile_rows = rows;
        statistics_.sweep_progress =
            static_cast<float>(tile_y_) /
            static_cast<float>(settings.height);
    }

    CudaDeviceContext device_context_;
    CudaPathStatistics statistics_;
    CudaFrameStorage frame_;
    CudaFrameStorage preview_frame_;
    std::unique_ptr<CudaSceneStorage> scene_;
    const std::array<float, 4> preview_scales_{
        1.0f,
        0.75f,
        0.5f,
        0.25f};
    int preview_scale_tier_ = 0;
    int slow_preview_frames_ = 0;
    int fast_preview_frames_ = 0;
    int idle_frames_ = kIdleFramesBeforeNative;
    int tile_y_ = 0;
    int quantum_rows_ = 64;
    int native_quantum_row_limit_ = kMaximumQuantumRows;
    int last_native_quantum_rows_ = 0;
    int output_width_ = 1;
    int output_height_ = 1;
    float preview_trace_ema_ = 0.0f;
    float native_ms_per_row_ema_ = 0.0f;
    std::uint64_t consumed_native_trace_generation_ = 0;
    std::uint64_t consumed_preview_trace_generation_ = 0;
    bool preview_dirty_ = true;
    bool preview_has_run_ = false;
    bool active_frame_is_preview_ = false;
    bool last_native_quantum_was_tail_ = false;
    bool native_sweep_published_this_frame_ = false;
    bool host_presentation_initialized_ = false;
    bool automatic_quality_enabled_ = false;
    bool has_progressive_key_ = false;
    bool has_uploaded_snapshot_ = false;
    std::uint64_t uploaded_source_id_ = 0;
    SceneRevisions uploaded_revisions_;
    ProgressiveRenderKey progressive_key_;
    CudaPathWorkMode last_work_mode_ = CudaPathWorkMode::FullFrame;
    std::chrono::steady_clock::time_point sweep_started_{};
};

CudaPathInteractiveRenderer::CudaPathInteractiveRenderer()
    : CudaPathInteractiveRenderer(CudaDeviceContext::create(0)) {}
CudaPathInteractiveRenderer::CudaPathInteractiveRenderer(
    CudaDeviceContext device_context)
    : impl_(std::make_unique<Impl>(std::move(device_context))) {}
CudaPathInteractiveRenderer::~CudaPathInteractiveRenderer() = default;
CudaPathInteractiveRenderer::CudaPathInteractiveRenderer(CudaPathInteractiveRenderer&&) noexcept = default;
CudaPathInteractiveRenderer& CudaPathInteractiveRenderer::operator=(CudaPathInteractiveRenderer&&) noexcept = default;

void CudaPathInteractiveRenderer::reset(
    const RenderSceneSnapshot& snapshot,
    const RenderSettings& settings) {
    impl_->reset(snapshot, settings);
}

void CudaPathInteractiveRenderer::render_next_frame(
    const RenderSceneSnapshot& snapshot,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    Framebuffer& target) {
    impl_->render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        target);
}

void CudaPathInteractiveRenderer::render_next_frame_to_surface(
    const RenderSceneSnapshot& snapshot,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    CudaSurfaceHandle surface) {
    if (surface == 0) {
        throw std::invalid_argument("CUDA surface output requires a non-zero surface handle");
    }
    impl_->render_next_frame_to_surface(
        snapshot,
        camera,
        settings,
        frame_state,
        surface);
}

void CudaPathInteractiveRenderer::download_current_frame(Framebuffer& target) {
    impl_->download_current_frame(target);
}

int CudaPathInteractiveRenderer::accumulated_samples() const {
    return impl_->accumulated_samples();
}

CudaStreamHandle CudaPathInteractiveRenderer::stream_handle() const {
    return impl_->stream_handle();
}

int CudaPathInteractiveRenderer::device_id() const {
    return impl_->statistics().device_id;
}

const CudaPathStatistics& CudaPathInteractiveRenderer::statistics() const {
    return impl_->statistics();
}

void CudaPathInteractiveRenderer::refresh_statistics() {
    impl_->refresh_statistics();
}

CudaPathDiagnosticProfile
CudaPathInteractiveRenderer::download_diagnostic_profile() {
    return impl_->download_diagnostic_profile();
}

void CudaPathInteractiveRenderer::set_presentation_state(
    bool interop_active,
    bool fallback_active) {
    impl_->set_presentation_state(interop_active, fallback_active);
}

}  // namespace renderer
