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
    std::uint64_t sample_seed_offset = 0;
    PathBackend backend = PathBackend::Auto;
};

struct RenderSettings {
    int width = 512;
    int height = 512;
    PathRenderSettings path;
};

}  // namespace renderer
