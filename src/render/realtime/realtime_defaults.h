#pragma once

#include "render/realtime/realtime_settings.h"

namespace renderer {

enum class RealtimeSettingsSection { Lighting, Denoising, Reconstruction, Diagnostics };

inline void reset_realtime_settings(RealtimeRenderSettings& s,RealtimeSettingsSection section) {
    const RealtimeRenderSettings d;
    switch(section) {
    case RealtimeSettingsSection::Lighting:
        s.low_discrepancy=d.low_discrepancy;
        s.shader_execution_reordering=d.shader_execution_reordering;
        s.split_dielectric=d.split_dielectric;
        s.specular_antialiasing=d.specular_antialiasing;
        s.regularize_indirect=d.regularize_indirect;
        s.samples_per_pixel=d.samples_per_pixel;
        s.max_bounces=d.max_bounces;
        s.roulette_start=d.roulette_start;
        s.light_samples=d.light_samples;
        s.direct_lighting=d.direct_lighting;
        s.shadows=d.shadows;
        s.soft_shadows=d.soft_shadows;
        s.indirect_diffuse=d.indirect_diffuse;
        s.reflections=d.reflections;
        s.transmission=d.transmission;
        break;
    case RealtimeSettingsSection::Denoising:
        s.denoise=d.denoise;s.denoiser=d.denoiser;s.temporal=d.temporal;
        s.firefly_filter=d.firefly_filter;s.firefly_sigma=d.firefly_sigma;
        s.history_clamping=d.history_clamping;s.history_sigma=d.history_sigma;
        s.diffuse_history=d.diffuse_history;s.specular_history=d.specular_history;
        s.transmission_history=d.transmission_history;
        s.depth_threshold=d.depth_threshold;s.normal_threshold=d.normal_threshold;
        s.reactive_strength=d.reactive_strength;
        s.diffuse_iterations=d.diffuse_iterations;s.specular_iterations=d.specular_iterations;
        s.depth_sigma=d.depth_sigma;s.normal_power=d.normal_power;s.luminance_sigma=d.luminance_sigma;
        break;
    case RealtimeSettingsSection::Reconstruction:
        s.internal_scale=d.internal_scale;s.taa=d.taa;
        s.temporal_upscale=d.temporal_upscale;s.full_resolution_materials=d.full_resolution_materials;
        s.taa_current_weight=d.taa_current_weight;s.taa_clip_sigma=d.taa_clip_sigma;
        s.sharpening=d.sharpening;
        break;
    case RealtimeSettingsSection::Diagnostics:
        s.debug_view=d.debug_view;
        break;
    }
}

} // namespace renderer
