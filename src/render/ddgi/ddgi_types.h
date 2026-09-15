#pragma once

#include "render/ddgi/ddgi_settings.h"
#include "scene/instanced_scene.h"
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace renderer {

constexpr int kDdgiIrradianceTexels = 8;
constexpr int kDdgiDistanceTexels = 16;
constexpr int kDdgiFixedRays = 32;

struct DdgiLayout {
    std::array<float, 3> origin{};
    std::array<float, 3> spacing{};
    std::array<int, 3> counts{};
    int columns = 0, rows = 0;
    int count() const {
        return counts[0] * counts[1] * counts[2];
    }
    int width(int texels) const {
        return columns * (texels + 2);
    }
    int height(int texels) const {
        return rows * (texels + 2);
    }
    Vec3 position(int index) const {
        return Vec3(origin[0] + spacing[0] * float(index % counts[0]),
                    origin[1] + spacing[1] * float((index / counts[0]) % counts[1]),
                    origin[2] + spacing[2] * float(index / (counts[0] * counts[1])));
    }
    bool operator==(const DdgiLayout &) const = default;
};

inline DdgiLayout make_ddgi_layout(const RenderSceneSnapshot &scene, DdgiSettings settings) {
    settings = normalized_ddgi_settings(settings);
    DdgiLayout layout;
    layout.counts = settings.probe_counts;
    Vec3 lo(settings.origin[0], settings.origin[1], settings.origin[2]);
    Vec3 extent(settings.extent[0], settings.extent[1], settings.extent[2]);
    if (settings.auto_fit && !scene.instances.empty()) {
        Bounds3 bounds;
        for (const auto &instance : scene.instances)
            bounds.expand(instance.world_bounds);
        if (bounds.min.allFinite() && bounds.max.allFinite() &&
            (bounds.max.array() >= bounds.min.array()).all()) {
            extent = bounds.max - bounds.min;
            const float scale = std::max(extent.maxCoeff(), 0.01f);
            extent = extent.cwiseMax(Vec3::Constant(scale * 0.1f));
            lo = (bounds.min + bounds.max - extent) * 0.5f - extent * 0.05f;
            extent *= 1.1f;
        }
    }
    for (int a = 0; a < 3; ++a) {
        layout.origin[a] = lo[a];
        layout.spacing[a] = extent[a] / float(layout.counts[a] - 1);
    }
    layout.columns = static_cast<int>(std::ceil(std::sqrt(float(layout.count()))));
    layout.rows = (layout.count() + layout.columns - 1) / layout.columns;
    return layout;
}

// Physical octahedral border texel -> matching interior texel.
inline std::array<int, 2> ddgi_border_source(int x, int y, int n) {
    if ((x == 0 || x == n + 1) && (y == 0 || y == n + 1))
        return {x == 0 ? n : 1, y == 0 ? n : 1};
    if (x == 0 || x == n + 1)
        return {x == 0 ? 1 : n, n + 1 - y};
    if (y == 0 || y == n + 1)
        return {n + 1 - x, y == 0 ? 1 : n};
    return {x, y};
}

struct DdgiStatistics {
    bool available = false, active = false;
    std::string status = "disabled", detail;
    int probe_count = 0, active_probes = 0, updated_probes = 0;
    int maximum_age = 0;
    std::uint64_t frame_index = 0, reset_count = 0, atlas_downloads = 0;
    std::uint64_t memory_bytes = 0, tlas_refits = 0, blas_builds = 0;
    float trace_ms = 0, blend_ms = 0, export_ms = 0, gather_ms = 0;
    DdgiLayout layout;
};

struct DdgiFrameResources {
    bool active = false;
    unsigned int irradiance = 0, distance = 0, metadata = 0;
    DdgiLayout layout;
    std::uint64_t frame_index = 0;
};

struct DdgiAtlasReadback {
    std::vector<std::array<float, 4>> irradiance, distance, metadata;
};

} // namespace renderer
