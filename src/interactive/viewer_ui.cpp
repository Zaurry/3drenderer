#include "interactive/viewer_ui.h"

#include "render/pathtracer/cuda_pathtracer.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <thread>

namespace renderer {

namespace {

constexpr ImGuiColorEditFlags kHdrColorFlags =
    ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR;

const char* backend_label(PathBackend backend) {
    switch (backend) {
        case PathBackend::Auto:
            return "Auto";
        case PathBackend::Cpu:
            return "CPU";
        case PathBackend::Cuda:
            return "CUDA";
    }
    return "Unknown";
}

const char* tone_mapper_label(ToneMapper tone_mapper) {
    switch (tone_mapper) {
        case ToneMapper::None:
            return "None";
        case ToneMapper::Reinhard:
            return "Reinhard";
        case ToneMapper::Aces:
            return "ACES";
    }
    return "Unknown";
}

void clamp_nonnegative(Color& color) {
    color = color.cwiseMax(0.0f);
}

float scene_radius(const Bounds3& bounds) {
    return std::max(0.5f, (bounds.max - bounds.min).norm() * 0.5f);
}

bool draw_path_backend(PathBackend& backend) {
    bool changed = false;
    if (!ImGui::BeginCombo("Path backend", backend_label(backend))) {
        return false;
    }

    const auto draw_choice = [&](PathBackend choice, bool enabled) {
        if (!enabled) {
            ImGui::BeginDisabled();
        }
        const bool selected = backend == choice;
        if (ImGui::Selectable(backend_label(choice), selected) && enabled) {
            backend = choice;
            changed = true;
        }
        if (selected) {
            ImGui::SetItemDefaultFocus();
        }
        if (!enabled) {
            ImGui::EndDisabled();
        }
    };

    draw_choice(PathBackend::Auto, true);
    draw_choice(PathBackend::Cpu, true);
    std::string cuda_reason;
    const bool cuda_available = cuda_path_backend_available(&cuda_reason);
    draw_choice(PathBackend::Cuda, cuda_available);
    if (!cuda_available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("CUDA unavailable: %s", cuda_reason.c_str());
    }
    ImGui::EndCombo();
    return changed;
}

}  // namespace

ViewerUiActions ViewerUi::draw(
    ViewerUiState& state,
    RenderSettings& render_settings,
    Scene& scene,
    OrbitCameraController& orbit_camera,
    FreeCameraController& free_camera,
    const Bounds3& bounds,
    const FrameRateSnapshot& performance,
    int accumulated_path_samples,
    ExecutionBackend active_path_backend) {
    ViewerUiActions actions;
    state.ui_font_scale = std::clamp(state.ui_font_scale, 0.75f, 2.0f);
    ImGui::GetStyle().FontScaleMain = state.ui_font_scale;
    if (!state.panel_visible) {
        return actions;
    }

    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360.0f, 680.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Renderer Controls", &state.panel_visible)) {
        ImGui::End();
        return actions;
    }

    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (performance.valid) {
            ImGui::Text("%.1f FPS  |  %.2f ms", performance.frames_per_second, performance.milliseconds_per_frame);
        } else {
            ImGui::TextUnformatted("Collecting frame timing...");
        }
        if (state.mode == InteractiveRenderMode::Path) {
            ImGui::Text(
                "%d spp  |  %s",
                accumulated_path_samples,
                active_path_backend == ExecutionBackend::Cuda ? "CUDA" : "CPU");
            if (ImGui::Button(state.path_accumulation_paused ? "Resume accumulation" : "Pause accumulation")) {
                state.path_accumulation_paused = !state.path_accumulation_paused;
            }
            ImGui::SameLine();
        }
        if (ImGui::Button("Reset render")) {
            actions.reset_requested = true;
        }
    }

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        int mode = static_cast<int>(state.mode);
        constexpr const char* modes[] = {"Raster", "Ray", "Path"};
        if (ImGui::Combo("Mode", &mode, modes, 3)) {
            state.mode = static_cast<InteractiveRenderMode>(mode);
            actions.mode_changed = true;
        }

        if (state.mode == InteractiveRenderMode::Path && draw_path_backend(render_settings.path_backend)) {
            actions.path_backend_changed = true;
        }
        if (state.mode == InteractiveRenderMode::Ray) {
            ImGui::SliderInt("Max depth", &render_settings.max_depth, 1, 32);
        }
        if (state.mode == InteractiveRenderMode::Path && render_settings.path_backend != PathBackend::Cuda) {
            const int hardware_threads = static_cast<int>(
                std::max(1U, std::thread::hardware_concurrency()));
            ImGui::SliderInt("CPU threads", &render_settings.thread_count, 0, hardware_threads, "%d");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("0 uses the hardware thread count");
            }
            ImGui::SliderInt("Tile size", &render_settings.tile_size, 4, 64);
        }
        int render_scale_percent = static_cast<int>(std::lround(state.render_scale * 100.0f));
        if (ImGui::SliderInt("Render scale", &render_scale_percent, 25, 100, "%d%%")) {
            state.render_scale = static_cast<float>(render_scale_percent) / 100.0f;
            actions.render_scale_changed = true;
        }
    }

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        int camera_mode = static_cast<int>(state.camera_mode);
        constexpr const char* camera_modes[] = {"Orbit", "Free"};
        if (ImGui::Combo("Control mode", &camera_mode, camera_modes, 2)) {
            state.camera_mode = static_cast<ViewerCameraMode>(camera_mode);
            actions.camera_mode_changed = true;
        }

        float fov = state.camera_mode == ViewerCameraMode::Orbit
            ? orbit_camera.vertical_fov_degrees()
            : free_camera.vertical_fov_degrees();
        if (ImGui::SliderFloat("Vertical FOV", &fov, 20.0f, 100.0f, "%.1f deg")) {
            orbit_camera.set_vertical_fov_degrees(fov);
            free_camera.set_vertical_fov_degrees(fov);
            actions.camera_parameters_changed = true;
        }

        const float radius = scene_radius(bounds);
        if (state.camera_mode == ViewerCameraMode::Orbit) {
            float distance = orbit_camera.distance();
            if (ImGui::SliderFloat(
                    "Orbit distance",
                    &distance,
                    radius * 0.05f,
                    radius * 20.0f,
                    "%.3f",
                    ImGuiSliderFlags_Logarithmic)) {
                orbit_camera.set_distance(distance);
                actions.camera_parameters_changed = true;
            }
        } else {
            float movement_speed = free_camera.movement_speed();
            if (ImGui::SliderFloat(
                    "Move speed",
                    &movement_speed,
                    radius * 0.01f,
                    radius * 10.0f,
                    "%.3f",
                    ImGuiSliderFlags_Logarithmic)) {
                free_camera.set_movement_speed(movement_speed);
            }
        }
        if (ImGui::Button("Reset camera")) {
            actions.camera_reset_requested = true;
        }
    }

    if (ImGui::CollapsingHeader("Lighting")) {
        if (ImGui::ColorEdit3("Environment", scene.environment.data(), kHdrColorFlags)) {
            clamp_nonnegative(scene.environment);
            actions.lighting_changed = true;
        }

        std::optional<std::size_t> directional_to_remove;
        for (std::size_t index = 0; index < scene.directional_lights.size(); ++index) {
            DirectionalLight& light = scene.directional_lights[index];
            ImGui::PushID(static_cast<int>(index));
            const std::string label = "Directional " + std::to_string(index + 1);
            if (ImGui::TreeNode(label.c_str())) {
                float direction[3]{light.direction.x(), light.direction.y(), light.direction.z()};
                if (ImGui::DragFloat3("Direction", direction, 0.01f, -1.0f, 1.0f)) {
                    const Vec3 candidate(direction[0], direction[1], direction[2]);
                    if (candidate.squaredNorm() > 1.0e-12f) {
                        light.direction = candidate.normalized();
                        actions.lighting_changed = true;
                    }
                }
                if (ImGui::ColorEdit3("Radiance", light.radiance.data(), kHdrColorFlags)) {
                    clamp_nonnegative(light.radiance);
                    actions.lighting_changed = true;
                }
                if (ImGui::Button("Delete")) {
                    directional_to_remove = index;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (directional_to_remove) {
            scene.directional_lights.erase(
                scene.directional_lights.begin() + static_cast<std::ptrdiff_t>(*directional_to_remove));
            actions.lighting_changed = true;
        }
        if (ImGui::Button("Add directional light")) {
            scene.directional_lights.push_back(DirectionalLight{
                Vec3(-0.5f, -1.0f, -0.25f).normalized(),
                Color(0.25f, 0.25f, 0.25f)});
            actions.lighting_changed = true;
        }

        std::optional<std::size_t> point_to_remove;
        for (std::size_t index = 0; index < scene.point_lights.size(); ++index) {
            PointLight& light = scene.point_lights[index];
            ImGui::PushID(static_cast<int>(index + scene.directional_lights.size()));
            const std::string label = "Point " + std::to_string(index + 1);
            if (ImGui::TreeNode(label.c_str())) {
                if (ImGui::DragFloat3("Position", light.position.data(), scene_radius(bounds) * 0.005f)) {
                    actions.lighting_changed = true;
                }
                if (ImGui::ColorEdit3("Intensity", light.intensity.data(), kHdrColorFlags)) {
                    clamp_nonnegative(light.intensity);
                    actions.lighting_changed = true;
                }
                if (ImGui::Button("Delete")) {
                    point_to_remove = index;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (point_to_remove) {
            scene.point_lights.erase(
                scene.point_lights.begin() + static_cast<std::ptrdiff_t>(*point_to_remove));
            actions.lighting_changed = true;
        }
        if (ImGui::Button("Add point light")) {
            const Vec3 center = (bounds.min + bounds.max) * 0.5f;
            const float radius = scene_radius(bounds);
            scene.point_lights.push_back(PointLight{
                center + Vec3(0.0f, radius, 0.0f),
                Color(10.0f, 10.0f, 10.0f)});
            actions.lighting_changed = true;
        }
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::SliderFloat("Exposure", &state.display.exposure_ev, -8.0f, 8.0f, "%+.2f EV")) {
            actions.display_changed = true;
        }
        int tone_mapper = static_cast<int>(state.display.tone_mapper);
        constexpr const char* tone_mappers[] = {"None", "Reinhard", "ACES"};
        if (ImGui::Combo("Tone mapping", &tone_mapper, tone_mappers, 3)) {
            state.display.tone_mapper = static_cast<ToneMapper>(tone_mapper);
            actions.display_changed = true;
        }
        ImGui::TextDisabled("Current: %s", tone_mapper_label(state.display.tone_mapper));
    }

    if (ImGui::CollapsingHeader("Interface", ImGuiTreeNodeFlags_DefaultOpen)) {
        int font_scale_percent = static_cast<int>(std::lround(state.ui_font_scale * 100.0f));
        if (ImGui::SliderInt("Font size", &font_scale_percent, 75, 200, "%d%%")) {
            state.ui_font_scale = static_cast<float>(font_scale_percent) / 100.0f;
            ImGui::GetStyle().FontScaleMain = state.ui_font_scale;
            actions.ui_style_changed = true;
        }
        if (ImGui::SmallButton("Reset font size")) {
            state.ui_font_scale = 1.0f;
            ImGui::GetStyle().FontScaleMain = state.ui_font_scale;
            actions.ui_style_changed = true;
        }
    }

    ImGui::Separator();
    ImGui::TextDisabled("Tab: toggle panel | R: reset | 1/2/3: mode");
    ImGui::TextDisabled("Orbit: LMB drag/wheel | Free: hold RMB + WASD");
    ImGui::End();
    return actions;
}

}  // namespace renderer
