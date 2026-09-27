#pragma once
#include "render/realtime/realtime_settings.h"
#include <nlohmann/json.hpp>
namespace renderer {
inline nlohmann::json realtime_settings_json(const RealtimeRenderSettings& input) {
    const auto s = sanitize_realtime_settings(input);
    return {
        {"low_discrepancy", s.low_discrepancy},
        {"shader_execution_reordering", s.shader_execution_reordering},
        {"split_dielectric", s.split_dielectric},
        {"specular_antialiasing", s.specular_antialiasing},
        {"regularize_indirect", s.regularize_indirect},
        {"samples_per_pixel", s.samples_per_pixel},
        {"max_bounces", s.max_bounces},
        {"roulette_start", s.roulette_start},
        {"light_samples", s.light_samples},
        {"internal_scale", s.internal_scale},
        {"direct_lighting", s.direct_lighting},
        {"shadows", s.shadows},
        {"soft_shadows", s.soft_shadows},
        {"indirect_diffuse", s.indirect_diffuse},
        {"reflections", s.reflections},
        {"transmission", s.transmission},
        {"denoise", s.denoise},
        {"denoiser", s.denoiser == RealtimeDenoiser::Optix ? "optix" : "svgf"},
        {"temporal", s.temporal},
        {"firefly_filter", s.firefly_filter},
        {"firefly_sigma", s.firefly_sigma},
        {"history_clamping", s.history_clamping},
        {"history_sigma", s.history_sigma},
        {"diffuse_history", s.diffuse_history},
        {"specular_history", s.specular_history},
        {"transmission_history", s.transmission_history},
        {"depth_threshold", s.depth_threshold},
        {"normal_threshold", s.normal_threshold},
        {"reactive_strength", s.reactive_strength},
        {"diffuse_iterations", s.diffuse_iterations},
        {"specular_iterations", s.specular_iterations},
        {"depth_sigma", s.depth_sigma},
        {"normal_power", s.normal_power},
        {"luminance_sigma", s.luminance_sigma},
        {"taa", s.taa},
        {"temporal_upscale", s.temporal_upscale},
        {"full_resolution_materials", s.full_resolution_materials},
        {"taa_current_weight", s.taa_current_weight},
        {"taa_clip_sigma", s.taa_clip_sigma},
        {"sharpening", s.sharpening},
        {"debug_view", static_cast<int>(s.debug_view)}
    };
}
inline RealtimeRenderSettings parse_realtime_settings(const nlohmann::json& j) {
    RealtimeRenderSettings s;
    if (!j.is_object()) return s;
    if(j.contains("low_discrepancy") && j.at("low_discrepancy").is_boolean()) s.low_discrepancy=j.at("low_discrepancy").get<bool>();
    if(j.contains("shader_execution_reordering") && j.at("shader_execution_reordering").is_boolean()) s.shader_execution_reordering=j.at("shader_execution_reordering").get<bool>();
    if(j.contains("split_dielectric") && j.at("split_dielectric").is_boolean()) s.split_dielectric=j.at("split_dielectric").get<bool>();
    if(j.contains("specular_antialiasing") && j.at("specular_antialiasing").is_boolean()) s.specular_antialiasing=j.at("specular_antialiasing").get<bool>();
    if(j.contains("regularize_indirect") && j.at("regularize_indirect").is_boolean()) s.regularize_indirect=j.at("regularize_indirect").get<bool>();
    const auto integer = [&](const char* name, int fallback) {
        const auto i=j.find(name);
        if (i==j.end() || !i->is_number()) return fallback;
        const double v=i->get<double>();
        return std::isfinite(v) ? static_cast<int>(std::clamp(v,-65536.0,65536.0)) : fallback;
    };
    s.samples_per_pixel = integer("samples_per_pixel", s.samples_per_pixel);
    s.max_bounces = integer("max_bounces", s.max_bounces);
    s.roulette_start = integer("roulette_start", s.roulette_start);
    s.light_samples = integer("light_samples", s.light_samples);
    if (j.contains("internal_scale") && j.at("internal_scale").is_number()) s.internal_scale = j.at("internal_scale").get<float>();
    if (j.contains("direct_lighting") && j.at("direct_lighting").is_boolean()) s.direct_lighting = j.at("direct_lighting").get<bool>();
    if (j.contains("shadows") && j.at("shadows").is_boolean()) s.shadows = j.at("shadows").get<bool>();
    if (j.contains("soft_shadows") && j.at("soft_shadows").is_boolean()) s.soft_shadows = j.at("soft_shadows").get<bool>();
    if (j.contains("indirect_diffuse") && j.at("indirect_diffuse").is_boolean()) s.indirect_diffuse = j.at("indirect_diffuse").get<bool>();
    if (j.contains("reflections") && j.at("reflections").is_boolean()) s.reflections = j.at("reflections").get<bool>();
    if (j.contains("transmission") && j.at("transmission").is_boolean()) s.transmission = j.at("transmission").get<bool>();
    if (j.contains("denoise") && j.at("denoise").is_boolean()) s.denoise = j.at("denoise").get<bool>();
    if (j.contains("denoiser") && j.at("denoiser").is_string()) {
        if(j.at("denoiser")=="optix")s.denoiser=RealtimeDenoiser::Optix;
        else if(j.at("denoiser")=="svgf")s.denoiser=RealtimeDenoiser::Svgf;
    }
    if (j.contains("temporal") && j.at("temporal").is_boolean()) s.temporal = j.at("temporal").get<bool>();
    if (j.contains("firefly_filter") && j.at("firefly_filter").is_boolean()) s.firefly_filter = j.at("firefly_filter").get<bool>();
    if (j.contains("firefly_sigma") && j.at("firefly_sigma").is_number()) s.firefly_sigma = j.at("firefly_sigma").get<float>();
    if (j.contains("history_clamping") && j.at("history_clamping").is_boolean()) s.history_clamping = j.at("history_clamping").get<bool>();
    if (j.contains("history_sigma") && j.at("history_sigma").is_number()) s.history_sigma = j.at("history_sigma").get<float>();
    s.diffuse_history = integer("diffuse_history", s.diffuse_history);
    s.specular_history = integer("specular_history", s.specular_history);
    s.transmission_history = integer("transmission_history", s.transmission_history);
    if (j.contains("depth_threshold") && j.at("depth_threshold").is_number()) s.depth_threshold = j.at("depth_threshold").get<float>();
    if (j.contains("normal_threshold") && j.at("normal_threshold").is_number()) s.normal_threshold = j.at("normal_threshold").get<float>();
    if (j.contains("reactive_strength") && j.at("reactive_strength").is_number()) s.reactive_strength = j.at("reactive_strength").get<float>();
    s.diffuse_iterations = integer("diffuse_iterations", s.diffuse_iterations);
    s.specular_iterations = integer("specular_iterations", s.specular_iterations);
    if (j.contains("depth_sigma") && j.at("depth_sigma").is_number()) s.depth_sigma = j.at("depth_sigma").get<float>();
    if (j.contains("normal_power") && j.at("normal_power").is_number()) s.normal_power = j.at("normal_power").get<float>();
    if (j.contains("luminance_sigma") && j.at("luminance_sigma").is_number()) s.luminance_sigma = j.at("luminance_sigma").get<float>();
    if (j.contains("taa") && j.at("taa").is_boolean()) s.taa = j.at("taa").get<bool>();
    if (j.contains("temporal_upscale") && j.at("temporal_upscale").is_boolean()) s.temporal_upscale = j.at("temporal_upscale").get<bool>();
    if (j.contains("full_resolution_materials") && j.at("full_resolution_materials").is_boolean()) s.full_resolution_materials = j.at("full_resolution_materials").get<bool>();
    if (j.contains("taa_current_weight") && j.at("taa_current_weight").is_number()) s.taa_current_weight = j.at("taa_current_weight").get<float>();
    if (j.contains("taa_clip_sigma") && j.at("taa_clip_sigma").is_number()) s.taa_clip_sigma = j.at("taa_clip_sigma").get<float>();
    if (j.contains("sharpening") && j.at("sharpening").is_number()) s.sharpening = j.at("sharpening").get<float>();
    if (j.contains("debug_view") && j.at("debug_view").is_number_integer())
        s.debug_view = static_cast<RealtimeDebugView>(std::clamp(integer("debug_view", 0), 0, int(RealtimeDebugView::Count)-1));
    return sanitize_realtime_settings(s);
}
} // namespace renderer
