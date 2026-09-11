#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace renderer {

enum class RealtimeDebugView : std::uint8_t {
    Final, Raw, Direct, IndirectDiffuse, Reflection, Transmission,
    Albedo, Normal, Depth, Motion, Variance, HistoryLength,
    Rejection, Reactive, Temporal, Filtered, Count
};

struct RealtimeRenderSettings {
    int samples_per_pixel = 1;
    int max_bounces = 8;
    int roulette_start = 3;
    int light_samples = 1;
    float internal_scale = 1.0f;
    bool direct_lighting = true;
    bool shadows = true;
    bool soft_shadows = true;
    bool indirect_diffuse = true;
    bool reflections = true;
    bool transmission = true;
    bool denoise = true;
    bool temporal = true;
    bool firefly_filter = true;
    float firefly_sigma = 6.0f;
    bool history_clamping = true;
    float history_sigma = 2.0f;
    int diffuse_history = 32;
    int specular_history = 8;
    int transmission_history = 4;
    float depth_threshold = 0.02f;
    float normal_threshold = 0.85f;
    float reactive_strength = 1.0f;
    int diffuse_iterations = 5;
    int specular_iterations = 3;
    float depth_sigma = 1.0f;
    float normal_power = 64.0f;
    float luminance_sigma = 4.0f;
    bool taa = true;
    bool temporal_upscale = true;
    float taa_current_weight = 0.15f;
    float taa_clip_sigma = 1.5f;
    float sharpening = 0.0f;
    RealtimeDebugView debug_view = RealtimeDebugView::Final;

    bool operator==(const RealtimeRenderSettings&) const = default;
};

inline RealtimeRenderSettings sanitize_realtime_settings(RealtimeRenderSettings s) {
    auto bounded = [](float v, float lo, float hi, float fallback) {
        return std::isfinite(v) ? std::clamp(v, lo, hi) : fallback;
    };
    s.samples_per_pixel = std::clamp(s.samples_per_pixel, 1, 64);
    s.max_bounces = std::clamp(s.max_bounces, 1, 64);
    s.roulette_start = std::clamp(s.roulette_start, 1, 64);
    s.light_samples = std::clamp(s.light_samples, 1, 16);
    s.internal_scale = bounded(s.internal_scale, 0.25f, 1.0f, 1.0f);
    s.firefly_sigma = bounded(s.firefly_sigma, 1.0f, 20.0f, 6.0f);
    s.history_sigma = bounded(s.history_sigma, 0.5f, 8.0f, 2.0f);
    s.diffuse_history = std::clamp(s.diffuse_history, 1, 128);
    s.specular_history = std::clamp(s.specular_history, 1, 64);
    s.transmission_history = std::clamp(s.transmission_history, 1, 32);
    s.depth_threshold = bounded(s.depth_threshold, 0.0001f, 0.2f, 0.02f);
    s.normal_threshold = bounded(s.normal_threshold, 0.0f, 0.9999f, 0.85f);
    s.reactive_strength = bounded(s.reactive_strength, 0.0f, 4.0f, 1.0f);
    s.diffuse_iterations = std::clamp(s.diffuse_iterations, 0, 5);
    s.specular_iterations = std::clamp(s.specular_iterations, 0, 5);
    s.depth_sigma = bounded(s.depth_sigma, 0.1f, 8.0f, 1.0f);
    s.normal_power = bounded(s.normal_power, 1.0f, 256.0f, 64.0f);
    s.luminance_sigma = bounded(s.luminance_sigma, 0.1f, 32.0f, 4.0f);
    s.taa_current_weight = bounded(s.taa_current_weight, 0.02f, 1.0f, 0.15f);
    s.taa_clip_sigma = bounded(s.taa_clip_sigma, 0.5f, 8.0f, 1.5f);
    s.sharpening = bounded(s.sharpening, 0.0f, 1.0f, 0.0f);
    if (static_cast<unsigned>(s.debug_view) >= static_cast<unsigned>(RealtimeDebugView::Count))
        s.debug_view = RealtimeDebugView::Final;
    return s;
}

struct RealtimeStatistics {
    bool active = false;
    std::uint64_t frames = 0;
    std::uint64_t history_resets = 0;
    std::uint64_t framebuffer_bytes = 0;
    float gbuffer_ms = 0;
    float lighting_ms = 0;
    float temporal_ms = 0;
    float filter_ms = 0;
    float reconstruction_ms = 0;
    float total_ms = 0;
};

}  // namespace renderer
