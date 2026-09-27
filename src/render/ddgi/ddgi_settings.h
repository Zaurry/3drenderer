#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace renderer {

enum class DdgiDebugView : std::uint8_t {
    Final,
    Indirect,
    ProbeState,
    ProbeAge,
    IrradianceAtlas,
    DistanceAtlas,
};

struct DdgiSettings {
    bool enabled = true;
    bool paused = false;
    bool auto_fit = true;
    std::array<float, 3> origin{-5.0f, -3.0f, -5.0f};
    std::array<float, 3> extent{10.0f, 6.0f, 10.0f};
    std::array<int, 3> probe_counts{12, 8, 12};
    int rays_per_probe = 128;
    int probes_per_frame = 256;
    float hysteresis = 0.95f;
    float normal_bias = 0.10f;
    float view_bias = 0.02f;
    float intensity = 1.0f;
    bool relocation = true;
    bool classification = true;
    bool show_probes = false;
    DdgiDebugView debug_view = DdgiDebugView::Final;
    // Commands are deliberately not persisted in viewer sessions.
    std::uint32_t reset_generation = 0;
    std::uint32_t fit_generation = 0;
    bool operator==(const DdgiSettings &) const = default;
};

inline void reset_ddgi_settings(DdgiSettings& settings) {
    // Commands must advance even when Reset is pressed twice in a row.
    const auto reset=settings.reset_generation+1,fit=settings.fit_generation+1;
    settings=DdgiSettings{};
    settings.reset_generation=reset;settings.fit_generation=fit;
}

inline DdgiSettings normalized_ddgi_settings(DdgiSettings s) {
    const auto bounded = [](float v, float fallback, float lo, float hi) {
        return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
    };
    for (int axis = 0; axis < 3; ++axis) {
        s.probe_counts[axis] = std::clamp(s.probe_counts[axis], 2, 32);
        s.origin[axis] = bounded(s.origin[axis], 0, -1.0e7f, 1.0e7f);
        s.extent[axis] = bounded(s.extent[axis], 10, 0.001f, 1.0e6f);
    }
    while (s.probe_counts[0] * s.probe_counts[1] * s.probe_counts[2] > 8192) {
        auto largest = std::max_element(s.probe_counts.begin(), s.probe_counts.end());
        --*largest;
    }
    s.rays_per_probe = std::clamp(s.rays_per_probe, 32, 512);
    s.probes_per_frame = std::clamp(s.probes_per_frame, 1,
                                    s.probe_counts[0] * s.probe_counts[1] * s.probe_counts[2]);
    s.hysteresis = bounded(s.hysteresis, 0.95f, 0, 0.999f);
    s.normal_bias = bounded(s.normal_bias, 0.1f, 0, 0.5f);
    s.view_bias = bounded(s.view_bias, 0.02f, 0, 0.5f);
    s.intensity = bounded(s.intensity, 1, 0, 10);
    if (static_cast<int>(s.debug_view) > 5)
        s.debug_view = DdgiDebugView::Final;
    return s;
}

} // namespace renderer
