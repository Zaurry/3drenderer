#include "interactive/viewer_ui.h"

#include "render/pathtracer/cuda_pathtracer.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <thread>

namespace renderer {

namespace {

struct ProjectedPoint {
    ImVec2 screen;
    float depth = 0.0f;
};

struct CubeFace {
    std::array<int, 4> corners;
    float depth = 0.0f;
    float shade = 1.0f;
};

std::optional<ProjectedPoint> project_to_screen(
    const Vec3& point,
    const Camera& camera,
    const ImVec2& display_size,
    const ImVec2& display_origin) {
    const Vec3 camera_to_point = point - camera.eye();
    const float depth = camera_to_point.dot(camera.forward());
    if (!std::isfinite(depth) || depth <= 1.0e-5f) {
        return std::nullopt;
    }

    const float normalized_x =
        2.0f * camera_to_point.dot(camera.right()) /
        (depth * camera.viewport_width());
    const float normalized_y =
        2.0f * camera_to_point.dot(camera.up()) /
        (depth * camera.viewport_height());
    if (!std::isfinite(normalized_x) || !std::isfinite(normalized_y)) {
        return std::nullopt;
    }

    return ProjectedPoint{
        ImVec2(
            display_origin.x +
                (normalized_x * 0.5f + 0.5f) * display_size.x,
            display_origin.y +
                (0.5f - normalized_y * 0.5f) * display_size.y),
        depth};
}

Color normalized_marker_color(const Color& intensity) {
    Color finite_intensity = Color::Zero();
    for (int channel = 0; channel < 3; ++channel) {
        if (std::isfinite(intensity[channel])) {
            finite_intensity[channel] = std::max(0.0f, intensity[channel]);
        }
    }

    const float maximum = finite_intensity.maxCoeff();
    if (maximum <= 1.0e-6f) {
        return Color(1.0f, 0.75f, 0.2f);
    }
    return Color::Constant(0.2f) + 0.8f * (finite_intensity / maximum);
}

ImU32 marker_face_color(const Color& color, float shade) {
    const Color shaded = (color * shade).cwiseMin(1.0f);
    return ImGui::ColorConvertFloat4ToU32(
        ImVec4(shaded.x(), shaded.y(), shaded.z(), 0.88f));
}

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

struct ColorStrengthEditResult {
    bool changed = false;
    bool finished = false;
};

ColorStrengthEditResult draw_color_and_strength(
    const char* id,
    const char* color_label,
    const char* strength_label,
    Color& value) {
    clamp_nonnegative(value);
    float strength = std::max(0.0f, value.maxCoeff());
    Color color = Color::Ones();
    if (strength > 1.0e-6f) {
        color = value / strength;
    }
    color = color.cwiseMax(0.0f).cwiseMin(1.0f);

    ColorStrengthEditResult result;
    ImGui::PushID(id);
    const auto fit_label = [](const char* label) {
        const float label_width =
            ImGui::CalcTextSize(label).x +
            ImGui::GetStyle().ItemInnerSpacing.x;
        ImGui::SetNextItemWidth(std::max(
            80.0f,
            ImGui::GetContentRegionAvail().x - label_width));
    };
    fit_label(color_label);
    if (ImGui::ColorEdit3(
            color_label,
            color.data(),
            ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel)) {
        color = color.cwiseMax(0.0f).cwiseMin(1.0f);
        if (strength <= 1.0e-6f) {
            strength = 1.0f;
        }
        result.changed = true;
    }
    result.finished =
        ImGui::IsItemDeactivatedAfterEdit() || result.finished;

    const float drag_speed = std::max(0.01f, strength * 0.01f);
    fit_label(strength_label);
    if (ImGui::DragFloat(
            strength_label,
            &strength,
            drag_speed,
            0.0f,
            100000.0f,
            "%.3f")) {
        strength = std::max(0.0f, strength);
        result.changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Scales all RGB channels together. Ctrl+click to enter an exact value.");
    }
    result.finished =
        ImGui::IsItemDeactivatedAfterEdit() || result.finished;

    if (ImGui::SmallButton("Reset strength")) {
        strength = 1.0f;
        result.changed = true;
        result.finished = true;
    }
    if (result.changed) {
        value = color * strength;
    }
    ImGui::PopID();
    return result;
}

float scene_radius(const Bounds3& bounds) {
    return std::max(0.5f, (bounds.max - bounds.min).norm() * 0.5f);
}

Mat4 camera_view_matrix(const Camera& camera) {
    Mat4 view = Mat4::Identity();
    view.row(0).head<3>() = camera.right().transpose();
    view.row(1).head<3>() = camera.up().transpose();
    view.row(2).head<3>() = (-camera.forward()).transpose();
    view(0, 3) = -camera.right().dot(camera.eye());
    view(1, 3) = -camera.up().dot(camera.eye());
    view(2, 3) = camera.forward().dot(camera.eye());
    return view;
}

Mat4 camera_projection_matrix(const Camera& camera, float near_plane, float far_plane) {
    Mat4 projection = Mat4::Zero();
    projection(0, 0) = 2.0f / camera.viewport_width();
    projection(1, 1) = 2.0f / camera.viewport_height();
    projection(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
    projection(2, 3) = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
    projection(3, 2) = -1.0f;
    return projection;
}

bool is_object_selected(const ViewerUiState& state, ObjectId id) {
    return std::find(
        state.selected_objects.begin(),
        state.selected_objects.end(),
        id) != state.selected_objects.end();
}

SceneChangeSet render_changes_for_subtree(
    const SceneDocument& document,
    ObjectId id,
    bool topology_changed) {
    const SceneObject* object = document.find(id);
    if (!object) {
        return SceneChange::None;
    }

    SceneChangeSet changes = SceneChange::None;
    if (object->type == SceneObjectType::Mesh) {
        changes |= SceneChange::Geometry;
        if (topology_changed) {
            changes |= SceneChange::MaterialBindings;
            changes |= SceneChange::Materials;
            changes |= SceneChange::Textures;
        }
    } else if (
        object->type == SceneObjectType::PointLight ||
        object->type == SceneObjectType::DirectionalLight) {
        changes |= SceneChange::Lighting;
    }
    for (ObjectId child : document.children(id)) {
        changes |= render_changes_for_subtree(document, child, topology_changed);
    }
    return changes;
}

void select_object(ViewerUiState& state, ObjectId id, bool additive) {
    if (!additive) {
        state.selected_objects.clear();
    }
    const auto found = std::find(
        state.selected_objects.begin(),
        state.selected_objects.end(),
        id);
    if (additive && found != state.selected_objects.end()) {
        state.selected_objects.erase(found);
        if (state.active_object == id) {
            state.active_object = state.selected_objects.empty()
                ? kInvalidObjectId
                : state.selected_objects.back();
        }
        return;
    }
    if (found == state.selected_objects.end()) {
        state.selected_objects.push_back(id);
    }
    state.active_object = id;
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

void build_default_dock_layout(ImGuiID dockspace_id, const ImGuiViewport& viewport) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    const ImGuiDockNodeFlags node_flags =
        static_cast<ImGuiDockNodeFlags>(ImGuiDockNodeFlags_DockSpace) |
        static_cast<ImGuiDockNodeFlags>(ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderAddNode(dockspace_id, node_flags);
    ImGui::DockBuilderSetNodeSize(dockspace_id, viewport.WorkSize);

    ImGuiID center = dockspace_id;
    ImGuiID left = 0;
    ImGuiID right = 0;
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.20f, &left, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.28f, &right, &center);

    ImGuiID right_bottom = 0;
    ImGuiID right_top = right;
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.52f, &right_bottom, &right_top);

    ImGui::DockBuilderDockWindow("Scene", left);
    ImGui::DockBuilderDockWindow("Rendering", right_top);
    ImGui::DockBuilderDockWindow("Inspector", right_bottom);
    ImGui::DockBuilderDockWindow("Camera & Lighting", right_bottom);
    ImGui::DockBuilderFinish(dockspace_id);
}

bool dock_layout_needs_default(const ImGuiDockNode* node) {
    if (!node) {
        return true;
    }
    if (!node->IsSplitNode()) {
        return false;
    }

    int populated_tool_nodes = 0;
    int usable_tool_nodes = 0;
    const std::function<void(const ImGuiDockNode*)> inspect =
        [&](const ImGuiDockNode* candidate) {
            if (!candidate) {
                return;
            }
            if (candidate->IsSplitNode()) {
                inspect(candidate->ChildNodes[0]);
                inspect(candidate->ChildNodes[1]);
                return;
            }
            if (!candidate->IsCentralNode() && candidate->Windows.Size > 0) {
                ++populated_tool_nodes;
                if (candidate->Size.x >= 180.0f &&
                    candidate->Size.y >= 120.0f) {
                    ++usable_tool_nodes;
                }
            }
        };
    inspect(node);
    return populated_tool_nodes >= 2 && usable_tool_nodes == 0;
}

}  // namespace

ViewerUiActions ViewerUi::draw(ViewerUiState& state,
                               RenderSettings& render_settings,
                               SceneDocument& document,
                               OrbitCameraController& orbit_camera,
                               FreeCameraController& free_camera,
                               const Bounds3& bounds,
                               const FrameRateSnapshot& performance,
                               int accumulated_path_samples,
                               ExecutionBackend active_path_backend,
                               const CudaOpenGlInteropUiState& interop_state,
                               const CudaPathStatistics& cuda_statistics,
                               OpenGlShaderUiState& shader_state,
                               bool scene_shortcuts_enabled) {
    ViewerUiActions actions;
    state.ui_font_scale = std::clamp(state.ui_font_scale, 0.75f, 2.0f);
    ImGui::GetStyle().FontScaleMain = state.ui_font_scale;

    bool reset_layout = false;
    if (state.panel_visible && ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene", nullptr, &state.scene_panel_visible);
            ImGui::MenuItem("Inspector", nullptr, &state.inspector_panel_visible);
            ImGui::MenuItem("Rendering", nullptr, &state.rendering_panel_visible);
            ImGui::MenuItem("Camera & Lighting", nullptr, &state.camera_lighting_panel_visible);
            ImGui::Separator();
            reset_layout = ImGui::MenuItem("Reset layout");
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    const ImGuiID dockspace_id = ImHashStr("RendererDockSpace");
    const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    const bool normal_window_size =
        main_viewport->WorkSize.x >= 800.0f &&
        main_viewport->WorkSize.y >= 480.0f;
    const bool dockspace_existed =
        ImGui::DockBuilderGetNode(dockspace_id) != nullptr;
    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
    if (!state.panel_visible) {
        dockspace_flags |= ImGuiDockNodeFlags_KeepAliveOnly;
    }
    ImGui::DockSpaceOverViewport(dockspace_id, main_viewport, dockspace_flags);
    const bool create_default_layout =
        normal_window_size &&
        (!dockspace_existed ||
         dock_layout_needs_default(ImGui::DockBuilderGetNode(dockspace_id)));
    if (create_default_layout || reset_layout) {
        state.scene_panel_visible = true;
        state.inspector_panel_visible = true;
        state.rendering_panel_visible = true;
        state.camera_lighting_panel_visible = true;
        build_default_dock_layout(dockspace_id, *main_viewport);
    }

    if (!state.panel_visible) {
        return actions;
    }

    const ImGuiIO& io = ImGui::GetIO();
    if (scene_shortcuts_enabled && !io.WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_G, false)) {
            state.gizmo_operation = 0;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_R, false) && !io.KeyCtrl) {
            state.gizmo_operation = 1;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
            state.gizmo_operation = 2;
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z, false) && document.undo()) {
            actions.scene_changes = SceneChange::All;
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Y, false) && document.redo()) {
            actions.scene_changes = SceneChange::All;
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
            actions.open_scene_requested = true;
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_I, false)) {
            actions.import_files_requested = true;
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S, false)) {
            actions.save_scene_requested = true;
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_D, false) &&
            state.active_object != kInvalidObjectId) {
            const ObjectId copy = document.duplicate_subtree(state.active_object);
            if (copy != kInvalidObjectId) {
                select_object(state, copy, false);
                actions.scene_changes = SceneChange::All;
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Delete, false) &&
            state.active_object != kInvalidObjectId &&
            document.erase_subtree(state.active_object)) {
            state.selected_objects.clear();
            state.active_object = kInvalidObjectId;
            actions.scene_changes = SceneChange::All;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
            actions.focus_object = state.active_object;
        }
    }

    if (state.rendering_panel_visible) {
        if (ImGui::Begin("Rendering", &state.rendering_panel_visible)) {
            const RenderModeCapability active_capabilities =
                render_mode_descriptor(state.mode).capabilities;
            if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (performance.valid) {
                    ImGui::Text("%.1f FPS  |  %.2f ms",
                                performance.frames_per_second,
                                performance.milliseconds_per_frame);
                } else {
                    ImGui::TextUnformatted("Collecting frame timing...");
                }
                if (has_capability(
                        active_capabilities,
                        RenderModeCapability::Progressive)) {
                    ImGui::Text("%d spp  |  %s",
                                accumulated_path_samples,
                                active_path_backend == ExecutionBackend::Cuda ? "CUDA" : "CPU");
                    if (active_path_backend == ExecutionBackend::Cuda) {
                        ImGui::Text("CUDA/OpenGL interop: %s", interop_state.status.c_str());
                        if (!interop_state.detail.empty()) {
                            ImGui::TextWrapped("%s", interop_state.detail.c_str());
                        }
                        ImGui::Text(
                            "GPU trace %.3f ms  |  reset %.3f ms  |  upload %.3f ms",
                            cuda_statistics.trace_milliseconds,
                            cuda_statistics.reset_milliseconds,
                            cuda_statistics.upload_milliseconds);
                        const char* work_mode = "full frame";
                        if (cuda_statistics.work_mode ==
                            CudaPathWorkMode::InteractionPreview) {
                            work_mode = "interaction preview";
                        } else if (cuda_statistics.work_mode ==
                                   CudaPathWorkMode::NativeTile) {
                            work_mode = "native tile";
                        }
                        ImGui::Text(
                            "%s  |  internal %dx%d",
                            work_mode,
                            cuda_statistics.internal_width,
                            cuda_statistics.internal_height);
                        if (cuda_statistics.work_mode ==
                            CudaPathWorkMode::NativeTile) {
                            ImGui::Text(
                                "Sweep %.1f%%  |  tile y=%d rows=%d  |  %.3f complete spp/s",
                                cuda_statistics.sweep_progress * 100.0f,
                                cuda_statistics.tile_y,
                                cuda_statistics.tile_rows,
                                cuda_statistics.complete_sweeps_per_second);
                        }
                        if (cuda_statistics.traversal_milliseconds > 0.0f ||
                            cuda_statistics.sort_milliseconds > 0.0f) {
                            ImGui::Text(
                                "Traversal %.3f ms  |  sort %.3f ms",
                                cuda_statistics.traversal_milliseconds,
                                cuda_statistics.sort_milliseconds);
                        }
                        ImGui::Text(
                            "Alloc generation %llu  |  downloads %llu",
                            static_cast<unsigned long long>(
                                cuda_statistics.allocation_generation),
                            static_cast<unsigned long long>(
                                cuda_statistics.framebuffer_downloads));
                        ImGui::Text(
                            "Upload KiB: geometry %.1f  BVH %.1f  material %.1f  "
                            "binding %.1f  texture %.1f  lighting %.1f",
                            static_cast<double>(
                                cuda_statistics.geometry_upload_bytes) /
                                1024.0,
                            static_cast<double>(
                                cuda_statistics.bvh_upload_bytes) /
                                1024.0,
                            static_cast<double>(
                                cuda_statistics.material_upload_bytes) /
                                1024.0,
                            static_cast<double>(
                                cuda_statistics.material_binding_upload_bytes) /
                                1024.0,
                            static_cast<double>(
                                cuda_statistics.texture_upload_bytes) /
                                1024.0,
                            static_cast<double>(
                                cuda_statistics.lighting_upload_bytes) /
                                1024.0);
                    }
                    if (ImGui::Button(state.path_accumulation_paused ? "Resume accumulation"
                                                                     : "Pause accumulation")) {
                        state.path_accumulation_paused = !state.path_accumulation_paused;
                    }
                    ImGui::SameLine();
                }
                if (ImGui::Button("Reset render")) {
                    actions.reset_requested = true;
                }
            }

            if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
                const RenderModeDescriptor& active_mode =
                    render_mode_descriptor(state.mode);
                if (ImGui::BeginCombo("Mode", active_mode.label)) {
                    for (const RenderModeDescriptor& descriptor :
                         interactive_render_modes()) {
                        const bool selected = descriptor.mode == state.mode;
                        if (ImGui::Selectable(descriptor.label, selected)) {
                            state.mode = descriptor.mode;
                            actions.mode_changed = true;
                        }
                        if (selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }

                if (has_capability(
                        active_capabilities,
                        RenderModeCapability::PathBackendSelection) &&
                    draw_path_backend(render_settings.path.backend)) {
                    actions.path_backend_changed = true;
                }
                if (has_capability(
                        active_capabilities,
                        RenderModeCapability::PathBackendSelection) &&
                    render_settings.path.backend != PathBackend::Cuda) {
                    const int hardware_threads =
                        static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
                    ImGui::SliderInt(
                        "CPU threads",
                        &render_settings.path.thread_count,
                        0,
                        hardware_threads,
                        "%d");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("0 uses the hardware thread count");
                    }
                    ImGui::SliderInt(
                        "Tile size",
                        &render_settings.path.tile_size,
                        4,
                        64);
                }
                if (active_path_backend == ExecutionBackend::Cuda &&
                    ImGui::Checkbox(
                        "Auto interaction quality",
                        &state.automatic_interaction_quality)) {
                    actions.automatic_interaction_quality_changed = true;
                }
                int render_scale_percent =
                    static_cast<int>(std::lround(state.render_scale * 100.0f));
                if (ImGui::SliderInt("Render scale", &render_scale_percent, 25, 100, "%d%%")) {
                    state.render_scale = static_cast<float>(render_scale_percent) / 100.0f;
                    actions.render_scale_changed = true;
                }
            }

            if (has_capability(
                    active_capabilities,
                    RenderModeCapability::ShaderReload) &&
                ImGui::CollapsingHeader("GLSL Shader", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextUnformatted(shader_state.valid ? "Program: active"
                                                          : "Program: unavailable");
                ImGui::TextWrapped("Vertex: %s", shader_state.vertex_path.c_str());
                ImGui::TextWrapped("Fragment: %s", shader_state.fragment_path.c_str());
                if (ImGui::Checkbox("Auto reload", &shader_state.auto_reload)) {
                    actions.shader_auto_reload_changed = true;
                }
                if (ImGui::Button("Reload shaders (F5)")) {
                    actions.shader_reload_requested = true;
                }
                if (!shader_state.error.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Compile/link error");
                    if (ImGui::BeginChild(
                            "ShaderError", ImVec2(0.0f, 130.0f), ImGuiChildFlags_Borders)) {
                        ImGui::TextUnformatted(shader_state.error.c_str());
                    }
                    ImGui::EndChild();
                } else {
                    ImGui::TextDisabled("Watching shader files every 250 ms");
                }
            }

            if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::SliderFloat(
                        "Exposure", &state.display.exposure_ev, -8.0f, 8.0f, "%+.2f EV")) {
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
                int font_scale_percent =
                    static_cast<int>(std::lround(state.ui_font_scale * 100.0f));
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
            ImGui::TextDisabled("G/R/S: transform | Ctrl+R: reset | 1/2/3/4: mode");
            ImGui::TextDisabled("Orbit: LMB drag/wheel | Free: hold RMB + WASD");
        }
        ImGui::End();
    }

    if (state.camera_lighting_panel_visible) {
        if (ImGui::Begin("Camera & Lighting", &state.camera_lighting_panel_visible)) {
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
                    if (ImGui::SliderFloat("Orbit distance",
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
                    if (ImGui::SliderFloat("Move speed",
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
                ImGui::Checkbox("Show point light markers", &state.show_point_light_markers);
                Color& environment = document.environment();
                const ColorStrengthEditResult environment_edit =
                    draw_color_and_strength(
                        "Environment",
                        "Environment color",
                        "Environment strength",
                        environment);
                if (environment_edit.changed) {
                    actions.scene_changes |= SceneChange::Lighting;
                }
                if (environment_edit.finished) {
                    document.checkpoint();
                }
                if (ImGui::Button("Add point light")) {
                    const Vec3 center = (bounds.min + bounds.max) * 0.5f;
                    const float radius = scene_radius(bounds);
                    const ObjectId id =
                        document.create_point_light("Point Light",
                                                    center + Vec3(0.0f, radius, 0.0f),
                                                    Color(10.0f, 10.0f, 10.0f));
                    document.checkpoint();
                    select_object(state, id, false);
                    actions.scene_changes |= SceneChange::Lighting;
                }
                ImGui::SameLine();
                if (ImGui::Button("Add directional light")) {
                    const ObjectId id =
                        document.create_directional_light("Directional Light",
                                                          Vec3(-0.5f, -1.0f, -0.25f),
                                                          Color(0.25f, 0.25f, 0.25f));
                    document.checkpoint();
                    select_object(state, id, false);
                    actions.scene_changes |= SceneChange::Lighting;
                }
            }
        }
        ImGui::End();
    }

    if (state.scene_panel_visible) {
        if (ImGui::Begin("Scene", &state.scene_panel_visible)) {
            if (ImGui::Button("Import OBJ...")) {
                actions.import_files_requested = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Import folder...")) {
                actions.import_folder_requested = true;
            }
            if (ImGui::Button("Open scene...")) {
                actions.open_scene_requested = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Save")) {
                actions.save_scene_requested = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Save as...")) {
                actions.save_scene_as_requested = true;
            }
            const std::string scene_name = document.file_path().empty()
                                               ? std::string("Untitled")
                                               : document.file_path().filename().string();
            ImGui::Text("%s%s", scene_name.c_str(), document.dirty() ? " *" : "");
            if (!state.scene_status.empty()) {
                ImGui::TextWrapped("%s", state.scene_status.c_str());
            }
            if (ImGui::TreeNode("Assets", "Assets (%zu)", document.assets().size())) {
                for (const auto& asset : document.assets()) {
                    if (!asset) {
                        continue;
                    }
                    ImGui::PushID(static_cast<int>(asset->id));
                    const bool missing =
                        !asset->source_path.empty() && !std::filesystem::exists(asset->source_path);
                    ImGui::TextWrapped("%s%s",
                                       missing ? "[missing] " : "",
                                       asset->source_path.empty()
                                           ? "<embedded>"
                                           : asset->source_path.string().c_str());
                    ImGui::PopID();
                }
                ImGui::TreePop();
            }
            ImGui::Separator();
            if (ImGui::Button("Undo") && document.undo()) {
                actions.scene_changes = SceneChange::All;
            }
            ImGui::SameLine();
            if (ImGui::Button("Redo") && document.redo()) {
                actions.scene_changes = SceneChange::All;
            }
            ImGui::SameLine();
            if (ImGui::Button("Add group")) {
                document.create_group("Group");
                document.checkpoint();
            }
            if (ImGui::RadioButton("Move (G)", state.gizmo_operation == 0)) {
                state.gizmo_operation = 0;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate (R)", state.gizmo_operation == 1)) {
                state.gizmo_operation = 1;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Scale (S)", state.gizmo_operation == 2)) {
                state.gizmo_operation = 2;
            }
            ImGui::Checkbox("Local coordinates", &state.gizmo_local);

            std::function<void(ObjectId)> draw_children = [&](ObjectId parent_id) {
                for (ObjectId id : document.children(parent_id)) {
                    SceneObject* object = document.find(id);
                    if (!object) {
                        continue;
                    }
                    ImGui::PushID(static_cast<int>(id));
                    const bool has_children = !document.children(id).empty();
                    ImGuiTreeNodeFlags flags =
                        ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
                    if (!has_children) {
                        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                    }
                    if (is_object_selected(state, id)) {
                        flags |= ImGuiTreeNodeFlags_Selected;
                    }
                    const bool open = ImGui::TreeNodeEx("object",
                                                        flags,
                                                        "%s%s",
                                                        object->visible ? "" : "[hidden] ",
                                                        object->name.c_str());
                    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                        select_object(state, id, ImGui::GetIO().KeyCtrl);
                    }
                    if (ImGui::BeginDragDropSource()) {
                        ImGui::SetDragDropPayload("SCENE_OBJECT", &id, sizeof(id));
                        ImGui::TextUnformatted(object->name.c_str());
                        ImGui::EndDragDropSource();
                    }
                    if (ImGui::BeginDragDropTarget()) {
                        if (const ImGuiPayload* payload =
                                ImGui::AcceptDragDropPayload("SCENE_OBJECT")) {
                            const ObjectId dropped = *static_cast<const ObjectId*>(payload->Data);
                            if (document.reparent(dropped, id)) {
                                actions.scene_changes = SceneChange::All;
                            }
                        }
                        ImGui::EndDragDropTarget();
                    }
                    if (open && has_children) {
                        draw_children(id);
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }
            };
            if (ImGui::BeginChild("Outliner", ImVec2(0.0f, 190.0f), ImGuiChildFlags_Borders)) {
                draw_children(kInvalidObjectId);
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }

    if (state.inspector_panel_visible) {
        if (ImGui::Begin("Inspector", &state.inspector_panel_visible)) {
            SceneObject* active = document.find(state.active_object);
            if (active) {
                char name_buffer[256]{};
                const std::size_t name_size =
                    std::min(active->name.size(), sizeof(name_buffer) - 1);
                std::memcpy(name_buffer, active->name.data(), name_size);
                name_buffer[name_size] = '\0';
                if (ImGui::InputText("Name", name_buffer, sizeof(name_buffer))) {
                    active->name = name_buffer;
                }
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    document.checkpoint();
                }
                bool visibility = active->visible;
                if (ImGui::Checkbox("Visible", &visibility)) {
                    active->visible = visibility;
                    document.checkpoint();
                    actions.scene_changes |= render_changes_for_subtree(
                        document,
                        active->id,
                        true);
                }
                ImGui::SameLine();
                bool locked = active->locked;
                if (ImGui::Checkbox("Locked", &locked)) {
                    active->locked = locked;
                    document.checkpoint();
                }

                ImGui::BeginDisabled(active->locked);
                bool transform_changed = false;
                bool transform_finished = false;
                transform_changed = ImGui::DragFloat3("Translation",
                                                      active->transform.translation.data(),
                                                      scene_radius(bounds) * 0.0025f) ||
                                    transform_changed;
                transform_finished = ImGui::IsItemDeactivatedAfterEdit() || transform_finished;
                transform_changed = ImGui::DragFloat3("Rotation",
                                                      active->transform.rotation_degrees.data(),
                                                      0.25f,
                                                      -3600.0f,
                                                      3600.0f,
                                                      "%.2f deg") ||
                                    transform_changed;
                transform_finished = ImGui::IsItemDeactivatedAfterEdit() || transform_finished;
                transform_changed =
                    ImGui::DragFloat3(
                        "Scale", active->transform.scale.data(), 0.01f, -1000.0f, 1000.0f) ||
                    transform_changed;
                transform_finished = ImGui::IsItemDeactivatedAfterEdit() || transform_finished;
                if (transform_changed) {
                    for (int axis = 0; axis < 3; ++axis) {
                        if (std::abs(active->transform.scale[axis]) < 1.0e-4f) {
                            active->transform.scale[axis] =
                                std::copysign(1.0e-4f, active->transform.scale[axis]);
                        }
                    }
                    actions.scene_changes |= render_changes_for_subtree(
                        document,
                        active->id,
                        false);
                }
                if (transform_finished) {
                    document.checkpoint();
                }
                ImGui::EndDisabled();

                if (active->type == SceneObjectType::Mesh) {
                    const SceneMeshAsset* asset = document.asset_for_object(active->id);
                    if (asset && !asset->local_scene.materials.empty()) {
                        if (state.material_editor_object != active->id) {
                            state.material_editor_object = active->id;
                            state.selected_material_slot = 0;
                        }
                        state.selected_material_slot = std::min(
                            state.selected_material_slot, asset->local_scene.materials.size() - 1);

                        ImGui::SeparatorText("Materials");
                        const auto material_name = [asset](std::size_t slot) {
                            return slot < asset->material_names.size() &&
                                           !asset->material_names[slot].empty()
                                       ? asset->material_names[slot]
                                       : "Material " + std::to_string(slot + 1);
                        };
                        const std::string current_name =
                            material_name(state.selected_material_slot);
                        if (ImGui::BeginCombo("Material slot", current_name.c_str())) {
                            for (std::size_t slot = 0; slot < asset->local_scene.materials.size();
                                 ++slot) {
                                const std::string name = material_name(slot);
                                const bool selected = slot == state.selected_material_slot;
                                ImGui::PushID(static_cast<int>(slot));
                                if (ImGui::Selectable(name.c_str(), selected)) {
                                    state.selected_material_slot = slot;
                                }
                                if (selected) {
                                    ImGui::SetItemDefaultFocus();
                                }
                                ImGui::PopID();
                            }
                            ImGui::EndCombo();
                        }

                        const std::size_t slot = state.selected_material_slot;
                        const bool has_override =
                            document.material_override(active->id, slot) != nullptr;
                        if (has_override) {
                            ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.2f, 1.0f), "Object override");
                            ImGui::SameLine();
                            ImGui::BeginDisabled(active->locked);
                            const bool reset_override = ImGui::SmallButton("Reset override");
                            ImGui::EndDisabled();
                            if (reset_override &&
                                document.clear_material_override(active->id, slot)) {
                                document.checkpoint();
                                actions.scene_changes |= SceneChange::Materials;
                                actions.scene_changes |= SceneChange::MaterialBindings;
                            }
                        } else {
                            ImGui::TextDisabled("Original OBJ/MTL material");
                        }

                        std::optional<SceneMaterialOverride> properties =
                            document.material_properties(active->id, slot);
                        if (properties) {
                            const Material& source_material = asset->local_scene.materials[slot];
                            properties->use_diffuse_texture =
                                properties->use_diffuse_texture &&
                                source_material.diffuse_texture_id >= 0;
                            properties->use_opacity_texture =
                                properties->use_opacity_texture &&
                                source_material.opacity_texture_id >= 0;
                            properties->use_bump_texture = properties->use_bump_texture &&
                                                           source_material.bump_texture_id >= 0;

                            bool material_changed = false;
                            bool material_edit_finished = false;
                            bool checkpoint_immediately = false;
                            ImGui::BeginDisabled(active->locked);

                            int material_type = static_cast<int>(properties->type);
                            constexpr const char* material_types[]{
                                "Diffuse", "Metal", "Dielectric", "Emissive"};
                            if (ImGui::Combo("Material type", &material_type, material_types, 4)) {
                                properties->type = static_cast<MaterialType>(material_type);
                                material_changed = true;
                                checkpoint_immediately = true;
                            }
                            if (ImGui::ColorEdit3("Base color / tint",
                                                  properties->base_color.data(),
                                                  ImGuiColorEditFlags_Float)) {
                                clamp_nonnegative(properties->base_color);
                                material_changed = true;
                            }
                            material_edit_finished =
                                ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("Multiplies the original diffuse texture; "
                                                  "white preserves its original colors.");
                            }

                            if (properties->type == MaterialType::Metal) {
                                material_changed =
                                    ImGui::SliderFloat(
                                        "Roughness", &properties->roughness, 0.0f, 1.0f) ||
                                    material_changed;
                                material_edit_finished =
                                    ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                                ImGui::TextDisabled("Physical roughness: Path");
                            } else if (properties->type == MaterialType::Dielectric) {
                                material_changed =
                                    ImGui::SliderFloat(
                                        "Index of refraction", &properties->ior, 1.0f, 3.0f) ||
                                    material_changed;
                                material_edit_finished =
                                    ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                                ImGui::TextDisabled("Physical refraction: Path");
                            } else if (properties->type == MaterialType::Emissive) {
                                const ColorStrengthEditResult emission_edit =
                                    draw_color_and_strength(
                                        "Emission",
                                        "Emission color",
                                        "Emission strength",
                                        properties->emission);
                                if (emission_edit.changed) {
                                    material_changed = true;
                                }
                                material_edit_finished = emission_edit.finished ||
                                    material_edit_finished;
                            }

                            material_changed =
                                ImGui::SliderFloat("Opacity", &properties->opacity, 0.0f, 1.0f) ||
                                material_changed;
                            material_edit_finished =
                                ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                            material_changed =
                                ImGui::SliderFloat(
                                    "Alpha cutoff", &properties->alpha_cutoff, 0.0f, 1.0f) ||
                                material_changed;
                            material_edit_finished =
                                ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                            material_changed =
                                ImGui::DragFloat("Bump scale", &properties->bump_scale, 0.01f) ||
                                material_changed;
                            material_edit_finished =
                                ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                            if (ImGui::Checkbox("Two-sided", &properties->two_sided)) {
                                material_changed = true;
                                checkpoint_immediately = true;
                            }

                            const auto texture_toggle =
                                [&](const char* label, int texture_id, bool& enabled) {
                                    const bool available = texture_id >= 0;
                                    ImGui::BeginDisabled(!available);
                                    if (!available) {
                                        enabled = false;
                                    }
                                    const bool changed = ImGui::Checkbox(label, &enabled);
                                    ImGui::EndDisabled();
                                    if (!available) {
                                        ImGui::SameLine();
                                        ImGui::TextDisabled("(none)");
                                    }
                                    return changed;
                                };
                            if (texture_toggle("Use diffuse texture",
                                               source_material.diffuse_texture_id,
                                               properties->use_diffuse_texture)) {
                                material_changed = true;
                                checkpoint_immediately = true;
                            }
                            if (texture_toggle("Use opacity texture",
                                               source_material.opacity_texture_id,
                                               properties->use_opacity_texture)) {
                                material_changed = true;
                                checkpoint_immediately = true;
                            }
                            if (texture_toggle("Use bump texture",
                                               source_material.bump_texture_id,
                                               properties->use_bump_texture)) {
                                material_changed = true;
                                checkpoint_immediately = true;
                            }
                            ImGui::EndDisabled();

                            if (material_changed &&
                                document.set_material_override(active->id, *properties)) {
                                actions.scene_changes |= SceneChange::Materials;
                                actions.scene_changes |= SceneChange::MaterialBindings;
                            }
                            if (material_edit_finished ||
                                (material_changed && checkpoint_immediately)) {
                                document.checkpoint();
                            }
                        }
                    }
                }

                if (active->type == SceneObjectType::PointLight ||
                    active->type == SceneObjectType::DirectionalLight) {
                    const ColorStrengthEditResult light_edit =
                        draw_color_and_strength(
                            "Light",
                            "Light color",
                            "Light intensity",
                            active->light_color);
                    if (light_edit.changed) {
                        actions.scene_changes |= SceneChange::Lighting;
                    }
                    if (light_edit.finished) {
                        document.checkpoint();
                    }
                }

                if (ImGui::Button("Duplicate")) {
                    const ObjectId copy = document.duplicate_subtree(active->id);
                    if (copy != kInvalidObjectId) {
                        select_object(state, copy, false);
                        actions.scene_changes = SceneChange::All;
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Delete")) {
                    const ObjectId removed = active->id;
                    if (document.erase_subtree(removed)) {
                        state.selected_objects.clear();
                        state.active_object = kInvalidObjectId;
                        actions.scene_changes = SceneChange::All;
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Focus")) {
                    actions.focus_object = active->id;
                }
            }
        }
        ImGui::End();
    }
    return actions;
}

bool ViewerUi::draw_scene_gizmo(ViewerUiState& state,
                                SceneDocument& document,
                                const Camera& camera,
                                const Bounds3& bounds) {
    SceneObject* active = document.find(state.active_object);
    if (!active || active->locked || state.selected_objects.empty()) {
        state.gizmo_was_using = false;
        state.gizmo_hovered = false;
        return false;
    }

    ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    const ImVec2 display_size = main_viewport->Size;
    if (display_size.x <= 0.0f || display_size.y <= 0.0f) {
        return false;
    }
    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList(main_viewport));
    ImGuizmo::SetRect(main_viewport->Pos.x, main_viewport->Pos.y, display_size.x, display_size.y);

    const float radius = scene_radius(bounds);
    const Mat4 view = camera_view_matrix(camera);
    const Mat4 projection = camera_projection_matrix(
        camera, std::max(1.0e-4f, radius * 1.0e-4f), std::max(1000.0f, radius * 100.0f));
    const Mat4 old_active_world = document.world_matrix(active->id);
    Mat4 manipulated = old_active_world;
    const ImGuizmo::OPERATION operation = state.gizmo_operation == 1   ? ImGuizmo::ROTATE
                                          : state.gizmo_operation == 2 ? ImGuizmo::SCALE
                                                                       : ImGuizmo::TRANSLATE;
    const ImGuizmo::MODE mode = state.gizmo_local ? ImGuizmo::LOCAL : ImGuizmo::WORLD;

    const bool changed =
        ImGuizmo::Manipulate(view.data(), projection.data(), operation, mode, manipulated.data());
    const bool using_gizmo = ImGuizmo::IsUsing();
    state.gizmo_hovered = ImGuizmo::IsOver() || using_gizmo;
    if (changed) {
        const Mat4 delta = manipulated * old_active_world.inverse();
        std::vector<ObjectId> roots;
        for (ObjectId id : state.selected_objects) {
            const SceneObject* object = document.find(id);
            if (!object || object->locked) {
                continue;
            }
            bool selected_ancestor = false;
            ObjectId parent_id = object->parent_id;
            while (parent_id != kInvalidObjectId) {
                if (is_object_selected(state, parent_id)) {
                    selected_ancestor = true;
                    break;
                }
                const SceneObject* parent = document.find(parent_id);
                parent_id = parent ? parent->parent_id : kInvalidObjectId;
            }
            if (!selected_ancestor) {
                roots.push_back(id);
            }
        }
        for (ObjectId id : roots) {
            document.set_world_matrix(id, delta * document.world_matrix(id));
        }
        document.rebuild_render_scene();
    }
    if (state.gizmo_was_using && !using_gizmo) {
        document.checkpoint();
    }
    state.gizmo_was_using = using_gizmo;
    return changed;
}

void ViewerUi::draw_scene_selection(const ViewerUiState& state,
                                    const SceneDocument& document,
                                    const Camera& camera) const {
    if (state.selected_objects.empty()) {
        return;
    }
    ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    const ImVec2 display_size = main_viewport->Size;
    const ImVec2 display_origin = main_viewport->Pos;
    constexpr std::array<std::array<int, 2>, 12> edges{{
        {{0, 1}},
        {{1, 3}},
        {{3, 2}},
        {{2, 0}},
        {{4, 5}},
        {{5, 7}},
        {{7, 6}},
        {{6, 4}},
        {{0, 4}},
        {{1, 5}},
        {{2, 6}},
        {{3, 7}},
    }};
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList(main_viewport);
    for (ObjectId id : state.selected_objects) {
        const Bounds3 bounds = document.world_bounds(id);
        if (!bounds.min.allFinite() || !bounds.max.allFinite()) {
            continue;
        }
        std::array<ImVec2, 8> points;
        bool valid = true;
        for (int corner = 0; corner < 8; ++corner) {
            const auto projected = project_to_screen(
                Vec3(
                    (corner & 1) != 0 ? bounds.max.x() : bounds.min.x(),
                    (corner & 2) != 0 ? bounds.max.y() : bounds.min.y(),
                    (corner & 4) != 0 ? bounds.max.z() : bounds.min.z()),
                camera,
                display_size,
                display_origin);
            if (!projected) {
                valid = false;
                break;
            }
            points[static_cast<std::size_t>(corner)] = projected->screen;
        }
        if (!valid) {
            continue;
        }
        const ImU32 color = id == state.active_object
            ? IM_COL32(255, 183, 40, 255)
            : IM_COL32(80, 190, 255, 230);
        for (const auto& edge : edges) {
            draw_list->AddLine(
                points[static_cast<std::size_t>(edge[0])],
                points[static_cast<std::size_t>(edge[1])],
                IM_COL32(10, 10, 10, 230),
                3.0f);
            draw_list->AddLine(
                points[static_cast<std::size_t>(edge[0])],
                points[static_cast<std::size_t>(edge[1])],
                color,
                1.25f);
        }
    }
}

void ViewerUi::draw_point_light_markers(
    const ViewerUiState& state,
    const Scene& scene,
    const Camera& camera) const {
    if (!state.show_point_light_markers || scene.point_lights.empty()) {
        return;
    }

    ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    const ImVec2 display_size = main_viewport->Size;
    const ImVec2 display_origin = main_viewport->Pos;
    if (!std::isfinite(display_size.x) || !std::isfinite(display_size.y) ||
        display_size.x <= 0.0f || display_size.y <= 0.0f) {
        return;
    }

    constexpr float marker_size_pixels = 14.0f;
    const std::array<Vec3, 8> corner_signs{
        Vec3(-1.0f, -1.0f, -1.0f),
        Vec3(1.0f, -1.0f, -1.0f),
        Vec3(1.0f, 1.0f, -1.0f),
        Vec3(-1.0f, 1.0f, -1.0f),
        Vec3(-1.0f, -1.0f, 1.0f),
        Vec3(1.0f, -1.0f, 1.0f),
        Vec3(1.0f, 1.0f, 1.0f),
        Vec3(-1.0f, 1.0f, 1.0f),
    };
    constexpr std::array<std::array<int, 2>, 12> edges{
        std::array<int, 2>{0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    constexpr std::array<CubeFace, 6> face_templates{
        CubeFace{{0, 3, 2, 1}, 0.0f, 0.68f},
        CubeFace{{4, 5, 6, 7}, 0.0f, 0.82f},
        CubeFace{{0, 4, 7, 3}, 0.0f, 0.74f},
        CubeFace{{1, 2, 6, 5}, 0.0f, 0.90f},
        CubeFace{{0, 1, 5, 4}, 0.0f, 0.62f},
        CubeFace{{3, 7, 6, 2}, 0.0f, 1.00f},
    };

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList(main_viewport);
    for (const PointLight& light : scene.point_lights) {
        const auto projected_center = project_to_screen(
            light.position,
            camera,
            display_size,
            display_origin);
        if (!projected_center ||
            projected_center->screen.x <
                display_origin.x - marker_size_pixels ||
            projected_center->screen.x >
                display_origin.x + display_size.x + marker_size_pixels ||
            projected_center->screen.y <
                display_origin.y - marker_size_pixels ||
            projected_center->screen.y >
                display_origin.y + display_size.y + marker_size_pixels) {
            continue;
        }

        const float half_extent =
            0.5f * marker_size_pixels * projected_center->depth *
            camera.viewport_height() / display_size.y;
        std::array<ProjectedPoint, 8> projected_corners;
        bool fully_visible = true;
        for (std::size_t index = 0; index < corner_signs.size(); ++index) {
            const auto projected = project_to_screen(
                light.position + corner_signs[index] * half_extent,
                camera,
                display_size,
                display_origin);
            if (!projected) {
                fully_visible = false;
                break;
            }
            projected_corners[index] = *projected;
        }
        if (!fully_visible) {
            continue;
        }

        std::array<CubeFace, 6> faces = face_templates;
        for (CubeFace& face : faces) {
            face.depth = 0.0f;
            for (const int corner : face.corners) {
                face.depth += projected_corners[static_cast<std::size_t>(corner)].depth;
            }
            face.depth *= 0.25f;
        }
        std::sort(
            faces.begin(),
            faces.end(),
            [](const CubeFace& a, const CubeFace& b) { return a.depth > b.depth; });

        const Color color = normalized_marker_color(light.intensity);
        for (const CubeFace& face : faces) {
            std::array<ImVec2, 4> polygon;
            for (std::size_t index = 0; index < face.corners.size(); ++index) {
                polygon[index] = projected_corners[
                    static_cast<std::size_t>(face.corners[index])].screen;
            }
            draw_list->AddConvexPolyFilled(
                polygon.data(),
                static_cast<int>(polygon.size()),
                marker_face_color(color, face.shade));
        }

        const ImU32 black_outline = IM_COL32(12, 12, 12, 235);
        const ImU32 bright_outline = marker_face_color(color, 1.0f);
        for (const auto& edge : edges) {
            const ImVec2 start = projected_corners[static_cast<std::size_t>(edge[0])].screen;
            const ImVec2 end = projected_corners[static_cast<std::size_t>(edge[1])].screen;
            draw_list->AddLine(start, end, black_outline, 3.0f);
            draw_list->AddLine(start, end, bright_outline, 1.25f);
        }
    }
}

}  // namespace renderer
