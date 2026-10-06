#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace renderer {

enum class DxrReconstruction { Auto, NrdTaau, NrdDlss, DlssRr, Reference, Count };
enum class DxrDebugView { Final, Raw, Direct, Indirect, Albedo, Normal, Depth, Motion,
    SpecularMotion, ReservoirAge, ReservoirWeight, Rejection, NrdValidation, HitDistance, TaaHistory, Count };

struct DxrRenderSettings {
    int samples_per_pixel = 1;
    int max_bounces = 8;
    bool restir_di = true;
    bool restir_pt = true;
    bool shader_execution_reordering = true;
    bool opacity_micromaps = true;
    bool specular_antialiasing = true;
    bool full_resolution_materials = true;
    DxrReconstruction reconstruction = DxrReconstruction::Auto;
    float internal_scale = 2.0f / 3.0f;
    int di_candidates = 8;
    int spatial_samples = 4;
    int pt_spatial_samples = 2;
    int pt_disocclusion_samples = 4;
    int history_length = 20;
    DxrDebugView debug_view = DxrDebugView::Final;
    bool operator==(const DxrRenderSettings&) const = default;
};

inline DxrRenderSettings sanitize_dxr_settings(DxrRenderSettings s) {
    s.samples_per_pixel = std::clamp(s.samples_per_pixel, 1, 64);
    s.max_bounces = std::clamp(s.max_bounces, 1, 32);
    s.di_candidates = std::clamp(s.di_candidates, 1, 64);
    s.spatial_samples = std::clamp(s.spatial_samples, 0, 16);
    s.pt_spatial_samples = std::clamp(s.pt_spatial_samples, 0, 16);
    s.pt_disocclusion_samples = std::clamp(s.pt_disocclusion_samples, s.pt_spatial_samples, 16);
    s.history_length = std::clamp(s.history_length, 1, 64);
    s.internal_scale = std::isfinite(s.internal_scale) ? std::clamp(s.internal_scale, .25f, 1.f) : 2.f/3.f;
    if (static_cast<unsigned>(s.reconstruction) >= static_cast<unsigned>(DxrReconstruction::Count))
        s.reconstruction = DxrReconstruction::Auto;
    if (static_cast<unsigned>(s.debug_view) >= static_cast<unsigned>(DxrDebugView::Count))
        s.debug_view = DxrDebugView::Final;
    return s;
}

struct DxrDeviceCapabilities {
    bool available = false;
    std::string adapter, reason;
    std::uint64_t adapter_luid = 0;
    std::uint64_t dedicated_bytes = 0;
    std::string driver_version;
    unsigned raytracing_tier = 0, shader_model = 0;
    bool enhanced_barriers = false;
    bool ser_supported = false, ser_reorders = false, omm_supported = false;
    bool dlss_sr_supported = false, dlss_rr_supported = false;
    bool debug_layer_active=false,gpu_validation_active=false;
};

struct DxrRuntimeModule {
    std::string name,path,version,sha256,expected_sha256;
    bool pinned=false;
};

struct DxrStatistics {
    DxrDeviceCapabilities device;
    bool active = false, restir_di_active = false, restir_pt_active = false;
    bool ser_active = false, omm_active = false, nrd_active = false;
    bool ser_probe_complete=false;
    float ser_measured_speedup=0,ser_probe_ms=0,trace_probe_ms=0;
    bool dlss_sr_active = false, dlss_rr_active = false;
    bool enhanced_barriers_active=false;
    std::string reconstruction, detail;
    int internal_width = 0, internal_height = 0;
    std::uint64_t frames = 0, history_resets = 0;
    std::uint64_t blas_builds = 0, tlas_builds = 0, tlas_updates = 0;
    std::uint64_t omm_builds=0,omm_pending=0;
    std::array<std::uint64_t,4> omm_states{};
    std::uint64_t allocated_bytes = 0, readbacks = 0;
    bool video_memory_available=false;
    std::uint64_t video_memory_usage=0,video_memory_budget=0;
    std::uint64_t pooled_bytes=0,resource_creations=0,resource_reuses=0,pipeline_cache_hits=0;
    std::vector<DxrRuntimeModule> runtime_modules;
    bool dlss_runtime_pinned=true;
    float acceleration_ms = 0, gbuffer_ms = 0, direct_ms = 0, indirect_ms = 0;
    float pt_initial_ms=0,pt_temporal_ms=0,pt_spatial_ms=0;
    float reconstruction_ms = 0, presentation_ms=0,total_ms = 0;
};

DxrDeviceCapabilities query_dxr_capabilities();
bool dxr_available(std::string* reason = nullptr);

} // namespace renderer
