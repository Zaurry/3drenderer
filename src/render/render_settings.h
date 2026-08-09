#pragma once

#include <cstdint>

namespace renderer {

enum class PathBackend {
    Auto,
    Cpu,
    Cuda
};

struct PathRenderSettings {
    int samples_per_pixel = 1;
    int tile_size = 16;
    int thread_count = 0;
    int max_bounces = 64;
    int russian_roulette_start_bounce = 3;
    float russian_roulette_min_probability = 0.05f;
    float russian_roulette_max_probability = 0.95f;
    std::uint64_t sample_seed_offset = 0;
    PathBackend backend = PathBackend::Auto;
};

struct RenderSettings {
    int width = 512;
    int height = 512;
    PathRenderSettings path;
};

}  // namespace renderer
