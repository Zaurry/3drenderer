#pragma once

#include "core/color.h"

namespace renderer {

enum class RenderMode {
    Raster,
    Ray,
    Path
};

struct RenderSettings {
    int width = 512;
    int height = 512;
    int samples_per_pixel = 1;
    int max_depth = 5;
    int tile_size = 16;
    int thread_count = 0;
    Color background = Color(0.02, 0.03, 0.05);
};

}  // namespace renderer
