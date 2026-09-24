#pragma once

#include "render/realtime/realtime_settings.h"
#include <imgui.h>

namespace renderer {

inline void draw_realtime_panel(RealtimeRenderSettings& s) {
    if(ImGui::Button("1080p RT quality preset"))s=realtime_1080p_quality_settings();
    if(ImGui::IsItemHovered())ImGui::SetTooltip("960x540 broad lighting; native 1080p textures, mirrors and glass. Validated on RTX 5080 / San Miguel.");
    ImGui::PushItemWidth(std::max(80.0f, ImGui::GetContentRegionAvail().x * .42f));
    if (ImGui::CollapsingHeader("RTRT Lighting & Sampling", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Shader execution reordering", &s.shader_execution_reordering);
        ImGui::Checkbox("Stable glass sampling", &s.split_dielectric);
        if(ImGui::IsItemHovered())ImGui::SetTooltip("Samples reflection and refraction separately at the first two glass interfaces. Uses extra rays on glass pixels.");
        ImGui::Checkbox("Owen Sobol sampling", &s.low_discrepancy);
        ImGui::SliderInt("SPP per frame", &s.samples_per_pixel, 1, 16);
        ImGui::SliderInt("Path depth", &s.max_bounces, 1, 64);
        ImGui::SliderInt("Roulette start", &s.roulette_start, 1, 64);
        ImGui::SliderInt("Light samples", &s.light_samples, 1, 16);
        ImGui::Checkbox("Direct lighting", &s.direct_lighting);
        ImGui::Checkbox("Ray traced shadows", &s.shadows);
        ImGui::Checkbox("Soft shadows", &s.soft_shadows);
        ImGui::Checkbox("Indirect diffuse GI", &s.indirect_diffuse);
        ImGui::Checkbox("Reflections", &s.reflections);
        ImGui::Checkbox("Transmission / glass", &s.transmission);
        ImGui::TextWrapped("Light radius and sun angular radius control shadow softness. Every frame traces and publishes a complete image.");
    }
    if (ImGui::CollapsingHeader("RTRT Denoising", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable denoising", &s.denoise);
        int denoiser=static_cast<int>(s.denoiser);
        if(ImGui::Combo("Denoiser", &denoiser, "SVGF\0OptiX AI\0"))
            s.denoiser=static_cast<RealtimeDenoiser>(denoiser);
        ImGui::BeginDisabled(!s.denoise);
        ImGui::Checkbox("Temporal reuse", &s.temporal);
        if(s.denoiser==RealtimeDenoiser::Optix) {
            ImGui::TextWrapped("OptiX AI denoises HDR lighting with albedo, normals and motion. Disable temporal reuse for single-frame denoising. Native materials also denoise full-resolution glass and reflections. Falls back to SVGF if unavailable.");
        }
        ImGui::BeginDisabled(s.denoiser!=RealtimeDenoiser::Svgf);
        ImGui::SliderInt("Diffuse history", &s.diffuse_history, 1, 128);
        ImGui::SliderInt("Specular history", &s.specular_history, 1, 64);
        ImGui::SliderInt("Transmission history", &s.transmission_history, 1, 16);
        ImGui::SliderFloat("Depth rejection", &s.depth_threshold, .001f, .2f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Normal rejection", &s.normal_threshold, 0, 1, "%.3f");
        ImGui::SliderFloat("Reactive strength", &s.reactive_strength, 0, 4, "%.2f");
        ImGui::Checkbox("History clamping", &s.history_clamping);
        ImGui::SliderFloat("History clamp sigma", &s.history_sigma, .5f, 8, "%.2f");
        ImGui::Checkbox("Firefly suppression", &s.firefly_filter);
        ImGui::SliderFloat("Firefly sigma", &s.firefly_sigma, 1, 16, "%.2f");
        ImGui::SliderInt("Diffuse atrous passes", &s.diffuse_iterations, 0, 5);
        ImGui::SliderInt("Specular atrous passes", &s.specular_iterations, 0, 5);
        ImGui::SliderFloat("Filter depth sigma", &s.depth_sigma, .1f, 8, "%.2f");
        ImGui::SliderFloat("Filter normal power", &s.normal_power, 1, 256, "%.0f");
        ImGui::SliderFloat("Filter luminance sigma", &s.luminance_sigma, .1f, 16, "%.2f");
        ImGui::TextWrapped("Diffuse lighting is demodulated before filtering. Sharp reflections and transmission accumulate once in output TAA to preserve detail.");
        ImGui::EndDisabled();
        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("TAA / Temporal Upscaling", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable TAA", &s.taa);
        ImGui::SliderFloat("Internal resolution", &s.internal_scale, .25f, 1, "%.2fx");
        ImGui::Checkbox("Temporal upscaling", &s.temporal_upscale);
        ImGui::Checkbox("Native materials / sharp optics", &s.full_resolution_materials);
        ImGui::SliderFloat("Current frame weight", &s.taa_current_weight, .01f, 1, "%.2f");
        ImGui::SliderFloat("TAA clip sigma", &s.taa_clip_sigma, .5f, 8, "%.2f");
        ImGui::SliderFloat("Sharpening", &s.sharpening, 0, 1, "%.2f");
        ImGui::TextWrapped("Native materials preserve texture and emission detail, with extra rays for glass and polished metal. Other lighting uses the internal resolution. A higher current frame weight responds faster.");
    }
    if (ImGui::CollapsingHeader("RTRT Diagnostics", ImGuiTreeNodeFlags_DefaultOpen)) {
        int view = static_cast<int>(s.debug_view);
        if (ImGui::Combo("View", &view,
            "Final\0Raw lighting\0Direct lighting\0Indirect diffuse\0Reflection\0Transmission\0"
            "Albedo\0Normal\0Depth\0Motion (UV)\0Variance\0History length\0History rejection\0Reactive mask\0Temporal\0Filtered\0"))
            s.debug_view = static_cast<RealtimeDebugView>(view);
        ImGui::TextWrapped("Diagnostic views and exposure do not enter lighting history. Reset render clears all temporal history.");
        if(s.denoiser==RealtimeDenoiser::Optix)
            ImGui::TextWrapped("OptiX: Filtered shows the neural output before TAA. Temporal shows raw lighting; components show internal-resolution signals. SVGF variance and history diagnostics are unavailable.");
    }
    s = sanitize_realtime_settings(s);
    ImGui::PopItemWidth();
}

} // namespace renderer
