#pragma once
#include "render/dxr/dxr_settings.h"
#include <nlohmann/json.hpp>
#include <array>

namespace renderer {
inline constexpr std::array<const char*, 5> dxr_reconstruction_names{
    "auto", "nrd_taau", "nrd_dlss", "dlss_rr", "reference"};
inline nlohmann::json dxr_settings_json(const DxrRenderSettings& settings) {
    const auto s = sanitize_dxr_settings(settings);
    return {{"samples_per_pixel",s.samples_per_pixel},{"max_bounces",s.max_bounces},
        {"restir_di",s.restir_di},{"restir_pt",s.restir_pt},
        {"shader_execution_reordering",s.shader_execution_reordering},
        {"opacity_micromaps",s.opacity_micromaps},{"specular_antialiasing",s.specular_antialiasing},
        {"full_resolution_materials",s.full_resolution_materials},
        {"reconstruction",dxr_reconstruction_names[static_cast<unsigned>(s.reconstruction)]},
        {"internal_scale",s.internal_scale},{"di_candidates",s.di_candidates},
        {"spatial_samples",s.spatial_samples},{"history_length",s.history_length},
        {"pt_spatial_samples",s.pt_spatial_samples},{"pt_disocclusion_samples",s.pt_disocclusion_samples},
        {"debug_view",static_cast<int>(s.debug_view)}};
}
inline DxrRenderSettings parse_dxr_settings(const nlohmann::json& j) {
    DxrRenderSettings s;
    if (!j.is_object()) return s;
    const auto integer=[&](const char* key,int fallback){
        const auto it=j.find(key);
        if(it==j.end() || !it->is_number())return fallback;
        const double v=it->get<double>();
        return std::isfinite(v)?static_cast<int>(std::clamp(v,-100000.,100000.)):fallback;
    };
    const auto flag=[&](const char* key,bool fallback){
        const auto it=j.find(key);return it!=j.end()&&it->is_boolean()?it->get<bool>():fallback;
    };
    s.samples_per_pixel=integer("samples_per_pixel",s.samples_per_pixel);
    s.max_bounces=integer("max_bounces",s.max_bounces);
    s.di_candidates=integer("di_candidates",s.di_candidates);
    s.spatial_samples=integer("spatial_samples",s.spatial_samples);
    // Older v8 settings used the same budget for DI and PT.
    s.pt_spatial_samples=integer("pt_spatial_samples",j.contains("spatial_samples")?s.spatial_samples:s.pt_spatial_samples);
    s.pt_disocclusion_samples=integer("pt_disocclusion_samples",std::max(s.pt_spatial_samples,4));
    s.history_length=integer("history_length",s.history_length);
    s.debug_view=static_cast<DxrDebugView>(integer("debug_view",0));
    s.restir_di=flag("restir_di",s.restir_di);s.restir_pt=flag("restir_pt",s.restir_pt);
    s.shader_execution_reordering=flag("shader_execution_reordering",true);
    s.opacity_micromaps=flag("opacity_micromaps",true);
    s.specular_antialiasing=flag("specular_antialiasing",true);
    s.full_resolution_materials=flag("full_resolution_materials",true);
    if(j.contains("internal_scale")&&j["internal_scale"].is_number())s.internal_scale=j["internal_scale"].get<float>();
    if(j.contains("reconstruction")&&j["reconstruction"].is_string())
        for(unsigned i=0;i<dxr_reconstruction_names.size();++i)
            if(j["reconstruction"]==dxr_reconstruction_names[i])s.reconstruction=static_cast<DxrReconstruction>(i);
    return sanitize_dxr_settings(s);
}
} // namespace renderer
