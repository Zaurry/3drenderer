#pragma once

#include <cstdint>

namespace renderer {

struct PathRenderSettings {
    int samples_per_pixel = 1;
    int max_bounces = 64;
    int russian_roulette_start_bounce = 3;
    float russian_roulette_min_probability = 0.05f;
    float russian_roulette_max_probability = 0.95f;
    std::uint64_t sample_seed_offset = 0;
    int cuda_device = 0;
};

enum class OpenGlShadowDebugView : std::uint8_t {
    Final = 0,
    Visibility,
    BlockerDepth,
    PenumbraRadius,
};

struct ShadowMapRenderSettings {
    bool enabled = true;
    int resolution = 1024;
    int max_shadow_lights = 8;
    float constant_bias = 0.0005f;
    float slope_bias = 0.0025f;
    float projection_padding = 0.05f;
    OpenGlShadowDebugView debug_view = OpenGlShadowDebugView::Final;
    int debug_shadow_slot = 0;

    bool operator==(const ShadowMapRenderSettings&) const = default;
};

struct PcssRenderSettings {
    bool enabled = true;
    int blocker_samples = 16;
    int filter_samples = 32;
    float max_penumbra_texels = 64.0f;
    float light_size_scale = 1.0f;

    bool operator==(const PcssRenderSettings&) const = default;
};

struct DominantLightExtractionRenderSettings {
    bool enabled = true;
    float peak_threshold_ev = 3.0f;
    float minimum_energy_fraction = 0.01f;
    float intensity_scale = 1.0f;

    bool operator==(const DominantLightExtractionRenderSettings&) const = default;
};

enum class OpenGlAmbientOcclusionMode : std::uint8_t {
    Off = 0,
    Ssao,
    Gtao,
};

enum class OpenGlAmbientOcclusionDebugView : std::uint8_t {
    Final = 0,
    Visibility,
    BentNormal,
    ViewNormal,
    LinearDepth,
};

struct SsaoRenderSettings {
    int sample_count = 32;
    float radius_scale = 0.10f;
    float depth_bias_fraction = 0.02f;
    float intensity = 1.0f;

    bool operator==(const SsaoRenderSettings&) const = default;
};

struct GtaoRenderSettings {
    int slice_count = 3;
    int samples_per_side = 3;
    float radius_scale = 0.10f;
    float falloff_fraction = 0.60f;
    float thickness_fraction = 0.20f;
    float intensity = 1.0f;
    bool bent_normals_enabled = true;

    bool operator==(const GtaoRenderSettings&) const = default;
};

struct AmbientOcclusionDenoiseSettings {
    bool enabled = true;
    int kernel_radius = 2;
    float depth_sigma_fraction = 0.10f;
    float normal_power = 8.0f;

    bool operator==(const AmbientOcclusionDenoiseSettings&) const = default;
};

struct AmbientOcclusionRenderSettings {
    OpenGlAmbientOcclusionMode mode = OpenGlAmbientOcclusionMode::Gtao;
    OpenGlAmbientOcclusionDebugView debug_view =
        OpenGlAmbientOcclusionDebugView::Final;
    SsaoRenderSettings ssao;
    GtaoRenderSettings gtao;
    AmbientOcclusionDenoiseSettings denoise;

    bool operator==(const AmbientOcclusionRenderSettings&) const = default;
};

enum class OpenGlSsrDebugView : std::uint8_t {
    Final = 0,
    Reflection,
    Confidence,
};

struct SsrRenderSettings {
    bool enabled = true;
    int max_steps = 64;
    int refinement_steps = 4;
    float max_distance_scale = 1.0f;
    float thickness_scale = 0.01f;
    float max_roughness = 0.9f;
    float intensity = 1.0f;
    float edge_fade = 0.15f;
    bool jitter = true;
    OpenGlSsrDebugView debug_view = OpenGlSsrDebugView::Final;

    bool operator==(const SsrRenderSettings&) const = default;
};

struct OpenGlRenderSettings {
    bool ibl_enabled = true;
    bool ltc_area_lights_enabled = true;
    ShadowMapRenderSettings shadow_map;
    PcssRenderSettings pcss;
    DominantLightExtractionRenderSettings dominant_light;
    AmbientOcclusionRenderSettings ambient_occlusion;
    SsrRenderSettings ssr;

    bool operator==(const OpenGlRenderSettings&) const = default;
};

struct RenderSettings {
    int width = 512;
    int height = 512;
    OpenGlRenderSettings opengl;
    PathRenderSettings path;
};

}  // namespace renderer
