#include "interactive/viewer_ui.h"
#include "interactive/realtime_panel.h"
#include "render/optix/optix_realtime_renderer.h"

#include "render/opengl/opengl_shader_contract.h"
#include "render/pathtracer/cuda_pathtracer.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace renderer {

namespace {

constexpr float kPi = 3.14159265358979323846f;

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

std::optional<Vec3> directional_light_world_direction(
    const SceneDocument& document,
    ObjectId id) {
    const SceneObject* object = document.find(id);
    if (!object || object->type != SceneObjectType::DirectionalLight) {
        return std::nullopt;
    }
    const Vec3 direction =
        document.world_matrix(id).topLeftCorner<3, 3>() *
        Vec3(0.0f, 0.0f, -1.0f);
    if (!direction.allFinite() || direction.squaredNorm() <= 1.0e-10f) {
        return std::nullopt;
    }
    return direction.normalized();
}

void draw_outlined_arrow(
    ImDrawList* draw_list,
    const ImVec2& start,
    const ImVec2& end,
    ImU32 color,
    float thickness,
    float head_length) {
    const ImVec2 delta(end.x - start.x, end.y - start.y);
    const float length = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    if (!std::isfinite(length) || length <= head_length + 1.0f) {
        return;
    }
    const ImVec2 forward(delta.x / length, delta.y / length);
    const ImVec2 side(-forward.y, forward.x);
    const ImVec2 head_base(
        end.x - forward.x * head_length,
        end.y - forward.y * head_length);
    const float head_half_width = head_length * 0.48f;
    const ImVec2 left(
        head_base.x + side.x * head_half_width,
        head_base.y + side.y * head_half_width);
    const ImVec2 right(
        head_base.x - side.x * head_half_width,
        head_base.y - side.y * head_half_width);
    constexpr ImU32 outline = IM_COL32(12, 12, 12, 245);
    draw_list->AddLine(start, head_base, outline, thickness + 4.0f);
    draw_list->AddTriangleFilled(end, left, right, outline);
    draw_list->AddLine(start, head_base, color, thickness);
    const ImVec2 inner_left(
        end.x + (left.x - end.x) * 0.72f,
        end.y + (left.y - end.y) * 0.72f);
    const ImVec2 inner_right(
        end.x + (right.x - end.x) * 0.72f,
        end.y + (right.y - end.y) * 0.72f);
    draw_list->AddTriangleFilled(end, inner_left, inner_right, color);
}

void draw_direction_label(
    ImDrawList* draw_list,
    const ImVec2& anchor,
    const ImVec2& display_origin,
    const ImVec2& display_size,
    const char* text) {
    const ImVec2 text_size = ImGui::CalcTextSize(text);
    const ImVec2 padding(8.0f, 5.0f);
    ImVec2 minimum(anchor.x + 10.0f, anchor.y + 10.0f);
    const float maximum_x = std::max(
        display_origin.x + 6.0f,
        display_origin.x + display_size.x -
            text_size.x - padding.x * 2.0f - 6.0f);
    const float maximum_y = std::max(
        display_origin.y + 6.0f,
        display_origin.y + display_size.y -
            text_size.y - padding.y * 2.0f - 6.0f);
    minimum.x = std::clamp(
        minimum.x,
        display_origin.x + 6.0f,
        maximum_x);
    minimum.y = std::clamp(
        minimum.y,
        display_origin.y + 6.0f,
        maximum_y);
    const ImVec2 maximum(
        minimum.x + text_size.x + padding.x * 2.0f,
        minimum.y + text_size.y + padding.y * 2.0f);
    draw_list->AddRectFilled(
        minimum,
        maximum,
        IM_COL32(18, 18, 22, 220),
        5.0f);
    draw_list->AddRect(
        minimum,
        maximum,
        IM_COL32(255, 202, 74, 245),
        5.0f,
        0,
        1.5f);
    draw_list->AddText(
        ImVec2(minimum.x + padding.x, minimum.y + padding.y),
        IM_COL32(255, 239, 194, 255),
        text);
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

SceneLightProperties light_properties_from_object(const SceneObject& object) {
    return SceneLightProperties{
        object.light_color,
        object.light_range,
        object.spot_inner_cone_radians,
        object.spot_outer_cone_radians,
        object.light_source_radius,
        object.directional_angular_radius_radians,
        object.light_casts_shadows,
        object.light_shadow_priority,
        object.area_width,
        object.area_height,
        object.light_two_sided};
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

std::vector<std::size_t> referenced_material_slots(
    const SceneMeshAsset& asset) {
    std::vector<unsigned char> referenced(
        asset.local_scene.materials.size(),
        0U);
    const auto mark_referenced = [&referenced](int material_id) {
        if (material_id >= 0 &&
            static_cast<std::size_t>(material_id) < referenced.size()) {
            referenced[static_cast<std::size_t>(material_id)] = 1U;
        }
    };
    for (const Triangle& triangle : asset.local_scene.triangles) {
        mark_referenced(triangle.material_id());
    }
    for (const Sphere& sphere : asset.local_scene.spheres) {
        mark_referenced(sphere.material_id());
    }

    std::vector<std::size_t> slots;
    for (std::size_t slot = 0; slot < referenced.size(); ++slot) {
        if (referenced[slot] != 0U) {
            slots.push_back(slot);
        }
    }
    if (slots.empty() && !asset.local_scene.materials.empty()) {
        slots.push_back(0);
    }
    return slots;
}

bool contains_case_insensitive(
    std::string_view text,
    std::string_view query) {
    if (query.empty()) {
        return true;
    }
    const auto equal_folded = [](char lhs, char rhs) {
        return std::tolower(static_cast<unsigned char>(lhs)) ==
            std::tolower(static_cast<unsigned char>(rhs));
    };
    return std::search(
               text.begin(),
               text.end(),
               query.begin(),
               query.end(),
               equal_folded) != text.end();
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
        if (topology_changed) {
            changes |= SceneChange::Geometry;
            changes |= SceneChange::MaterialBindings;
            changes |= SceneChange::Materials;
            changes |= SceneChange::Textures;
        } else {
            changes |= SceneChange::InstanceTransforms;
        }
    } else if (object->type == SceneObjectType::RectAreaLight) {
        changes |= SceneChange::Lighting;
        changes |= SceneChange::Materials;
        changes |= SceneChange::InstanceTransforms;
        if (topology_changed) {
            changes |= SceneChange::Geometry;
            changes |= SceneChange::MaterialBindings;
        }
    } else if (
        object->type == SceneObjectType::PointLight ||
        object->type == SceneObjectType::DirectionalLight ||
        object->type == SceneObjectType::SpotLight) {
        changes |= SceneChange::Lighting;
    }
    for (ObjectId child : document.children(id)) {
        changes |= render_changes_for_subtree(document, child, topology_changed);
    }
    return changes;
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
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.24f, &left, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.28f, &right, &center);

    ImGuiID right_bottom = 0;
    ImGuiID right_top = right;
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.52f, &right_bottom, &right_top);

    ImGui::DockBuilderDockWindow("Scene", left);
    ImGui::DockBuilderDockWindow("Rendering", right_top);
    ImGui::DockBuilderDockWindow("Techniques", right_top);
    ImGui::DockBuilderDockWindow("Inspector", right_bottom);
    ImGui::DockBuilderDockWindow("Camera", right_bottom);
    ImGui::DockBuilderFinish(dockspace_id);
}

}  // namespace

ViewerUiActions ViewerUi::draw(ViewerUiState& state,
                               RenderSettings& render_settings,
                               SceneDocument& document,
                               OrbitCameraController& orbit_camera,
                               FreeCameraController& free_camera,
                               const Bounds3& bounds,
                               const FrameRateSnapshot& performance,
                               int /*accumulated_path_samples*/,
                               const CudaOpenGlInteropUiState& interop_state,
                               const CudaPathStatistics& cuda_statistics,
                               const OpenGlTechniqueDiagnostics& technique_diagnostics,
                               OpenGlShaderUiState& shader_state,
                               bool scene_shortcuts_enabled,
                               const DxrStatistics& dxr_statistics) {
    ViewerUiActions actions;
    state.ui_font_scale = std::clamp(state.ui_font_scale, 0.75f, 2.0f);
    ImGui::GetStyle().FontScaleMain = state.ui_font_scale;

    bool reset_layout = false;
    if (state.panel_visible && ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene", nullptr, &state.scene_panel_visible);
            ImGui::MenuItem("Inspector", nullptr, &state.inspector_panel_visible);
            ImGui::MenuItem("Rendering", nullptr, &state.rendering_panel_visible);
            ImGui::MenuItem("Techniques", nullptr, &state.techniques_panel_visible);
            ImGui::MenuItem("Camera", nullptr, &state.camera_lighting_panel_visible);
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
    // A loaded multi-viewport layout may report zero-sized dock nodes during
    // the first frame while its external platform windows are being created.
    // Treating that transient state as a broken layout re-docks every named
    // panel into the main viewport and leaves the saved floating nodes orphaned.
    // Only synthesize the default layout when no saved dockspace exists; users
    // can explicitly repair a real bad layout through View -> Reset layout.
    const bool create_default_layout = normal_window_size && !dockspace_existed;
    if (create_default_layout || reset_layout) {
        state.scene_panel_visible = true;
        state.inspector_panel_visible = true;
        state.rendering_panel_visible = true;
        state.camera_lighting_panel_visible = true;
        state.techniques_panel_visible = true;
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
                select_viewer_object(state, copy, false);
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
            if (settings_header("Performance",[&]{actions.reset_requested=true;},"Clear rendering history and restart accumulation.")) {
                if (performance.valid) {
                    ImGui::Text("%.1f FPS  |  %.2f ms",
                                performance.frames_per_second,
                                performance.milliseconds_per_frame);
                } else {
                    ImGui::TextUnformatted("Collecting frame timing...");
                }
                if(state.mode==InteractiveRenderMode::Dxr) {
                    const auto& d=dxr_statistics;
                    ImGui::TextWrapped("%s",d.device.adapter.c_str());
                    ImGui::Text("DXR %.1f | SM %.1f",d.device.raytracing_tier/10.0f,d.device.shader_model/16+(d.device.shader_model%16)/10.0f);
                    ImGui::Text("Internal %d x %d | GPU %.2f ms",d.internal_width,d.internal_height,d.total_ms);
                    ImGui::Text("PT: candidates %.2f | temporal %.2f | spatial %.2f ms",d.pt_initial_ms,d.pt_temporal_ms,d.pt_spatial_ms);
                    ImGui::Text("Reconstruction: %s",d.reconstruction.c_str());
                    ImGui::TextWrapped("%s",d.detail.c_str());
                    ImGui::Text("SER: requested %s | supported %s | active %s",render_settings.dxr.shader_execution_reordering?"yes":"no",d.device.ser_supported?"yes":"no",d.ser_active?"yes":"no");
                    if(render_settings.dxr.shader_execution_reordering && d.device.ser_reorders) {
                        if(d.ser_probe_complete)ImGui::Text("SER measured ray-stage speedup: %+.1f%% (%.2f / %.2f ms)",d.ser_measured_speedup*100,d.ser_probe_ms,d.trace_probe_ms);
                        else ImGui::TextUnformatted("SER Auto: measuring stable frames during warmup");
                    }
                    ImGui::Text("OMM: requested %s | supported %s | active %s",render_settings.dxr.opacity_micromaps?"yes":"no",d.device.omm_supported?"yes":"no",d.omm_active?"yes":"no");
                    ImGui::Text("OMM bakes: %llu | pending: %llu",static_cast<unsigned long long>(d.omm_builds),static_cast<unsigned long long>(d.omm_pending));
                    ImGui::Text("ReSTIR DI %s | PT %s",d.restir_di_active?"active":"off",d.restir_pt_active?"active":"off");
                    ImGui::Text("BLAS builds %llu | TLAS builds %llu / updates %llu",(unsigned long long)d.blas_builds,(unsigned long long)d.tlas_builds,(unsigned long long)d.tlas_updates);
                    ImGui::Text("Managed GPU allocations %.1f MiB | image readbacks %llu",double(d.allocated_bytes)/1048576,(unsigned long long)d.readbacks);
                    if(d.video_memory_available)ImGui::Text("Process local VRAM %.1f / %.1f MiB budget",double(d.video_memory_usage)/1048576,double(d.video_memory_budget)/1048576);
                    else ImGui::TextUnformatted("Process local VRAM: unavailable");
                }
                if (state.mode == InteractiveRenderMode::Rtrt) {
                    const auto& rt = cuda_statistics.realtime;
                    ImGui::Text("CUDA/OpenGL interop: %s", interop_state.status.c_str());
                    if (!interop_state.detail.empty()) ImGui::TextWrapped("%s", interop_state.detail.c_str());
                    ImGui::Text("%d SPP/frame | internal %dx%d", render_settings.realtime.samples_per_pixel,
                        cuda_statistics.internal_width, cuda_statistics.internal_height);
                    ImGui::Text("G-buffer %.2f ms | lighting %.2f ms", rt.gbuffer_ms, rt.lighting_ms);
                    ImGui::Text("Signal preparation %.2f ms | denoising %.2f ms", rt.temporal_ms, rt.filter_ms);
                    ImGui::Text("Denoiser: %s", !render_settings.realtime.denoise?"Off":
                        (rt.optix_denoiser_active?(rt.optix_denoiser_temporal?"OptiX AI (temporal)":"OptiX AI (single frame)"):"SVGF"));
                    if(!rt.optix_denoiser_detail.empty())ImGui::TextWrapped("%s",rt.optix_denoiser_detail.c_str());
                    ImGui::Text("Reconstruction %.2f ms | CUDA %.2f ms", rt.reconstruction_ms, rt.total_ms);
                    ImGui::Text("Primary: OptiX | RT Core version %u",rt.rt_core_version);
                    ImGui::Text("SER: %s",rt.ser_active?"active":rt.ser_supported?"disabled":"unsupported");
                    ImGui::Text("GAS builds %llu | IAS builds %llu / updates %llu",
                        (unsigned long long)rt.gas_builds,(unsigned long long)rt.ias_builds,(unsigned long long)rt.ias_updates);
                    ImGui::Text("Acceleration %.2f ms | %.1f MiB",rt.acceleration_ms,double(rt.hardware_ray_tracing_bytes)/1048576);
                    ImGui::Text("Ray traversal: %s",rt.hardware_ray_tracing_active?"RT cores (OptiX)":"unavailable");
                    if(!rt.hardware_ray_tracing_detail.empty())ImGui::TextWrapped("%s",rt.hardware_ray_tracing_detail.c_str());
                    ImGui::Text("Frames %llu | history resets %llu", (unsigned long long)rt.frames, (unsigned long long)rt.history_resets);
                    ImGui::Text("Frame buffers %.1f MiB | downloads %llu", double(rt.framebuffer_bytes)/1048576,
                        (unsigned long long)cuda_statistics.framebuffer_downloads);
                    ImGui::Text("Scene upload %.2f ms | TLAS refit %.2f ms", cuda_statistics.upload_milliseconds, cuda_statistics.tlas_refit_milliseconds);
                }
            }

            if (settings_header("Rendering",[&] {
                    const auto mode=optix_realtime_available(render_settings.path.cuda_device)
                        ?InteractiveRenderMode::Rtrt:InteractiveRenderMode::OpenGl;
                    actions.mode_changed=state.mode!=mode;state.mode=mode;
                    render_settings.opengl.npr=NprRenderSettings{};
                    actions.render_scale_changed=state.render_scale!=1.0f;state.render_scale=1.0f;
                })) {
                const RenderModeDescriptor& active_mode =
                    render_mode_descriptor(state.mode);
                if (ImGui::BeginCombo("Mode", active_mode.label)) {
                    std::string cuda_reason;
                    const bool cuda_available =
                        optix_realtime_available(render_settings.path.cuda_device,&cuda_reason);
                    std::string dxr_reason;const bool dxr_supported=dxr_available(&dxr_reason);
                    for (const RenderModeDescriptor& descriptor :
                         interactive_render_modes()) {
                        const bool selected = descriptor.mode == state.mode;
                        const bool enabled = descriptor.mode==InteractiveRenderMode::Dxr?dxr_supported:
                            descriptor.mode!=InteractiveRenderMode::Rtrt || cuda_available;
                        if (!enabled) {
                            ImGui::BeginDisabled();
                        }
                        if (ImGui::Selectable(descriptor.label, selected) && enabled) {
                            state.mode = descriptor.mode;
                            actions.mode_changed = true;
                        }
                        if (!enabled &&
                            ImGui::IsItemHovered(
                                ImGuiHoveredFlags_AllowWhenDisabled)) {
                            ImGui::SetTooltip(
                                "%s",descriptor.mode==InteractiveRenderMode::Dxr?dxr_reason.c_str():cuda_reason.c_str());
                        }
                        if (selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                        if (!enabled) {
                            ImGui::EndDisabled();
                        }
                    }
                    ImGui::EndCombo();
                }

                if (state.mode == InteractiveRenderMode::OpenGl) {
                auto& npr = render_settings.opengl.npr;
                int style = static_cast<int>(npr.style);
                if (ImGui::Combo("Style", &style, "Realistic\0Toon\0Sketch\0")) {
                    npr.style = static_cast<OpenGlRenderStyle>(style);
                    // A style selection should immediately show the final image.
                    render_settings.opengl.ddgi.debug_view = DdgiDebugView::Final;
                    render_settings.opengl.shadow_map.debug_view = OpenGlShadowDebugView::Final;
                    render_settings.opengl.ambient_occlusion.debug_view = OpenGlAmbientOcclusionDebugView::Final;
                    render_settings.opengl.ssr.debug_view = OpenGlSsrDebugView::Final;
                }
                if (npr.style != OpenGlRenderStyle::Realistic) {
                    ImGui::SliderFloat("Outline width (px)", &npr.outline_width, 0.0f, 4.0f);
                    ImGui::SliderFloat("Outline strength", &npr.outline_strength, 0.0f, 1.0f);
                    if (npr.style == OpenGlRenderStyle::Toon) {
                        ImGui::SliderInt("Diffuse bands", &npr.toon_levels, 2, 6);
                    } else {
                        ImGui::SliderFloat("Hatching scale", &npr.sketch_scale, 0.5f, 32.0f);
                        ImGui::SliderFloat("Pencil darkness", &npr.sketch_tone, 0.25f, 2.0f);
                        ImGui::Checkbox("Use mesh UV for strokes", &npr.sketch_use_uv);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("UV strokes follow object motion; requires usable UVs.\n"
                                "Otherwise world triplanar mapping supports meshes without UVs.");
                        }
                    }
                    ImGui::TextWrapped("NPR uses direct light, IBL and AO. SSGI / SSR resume in Realistic style.");
                }
                }

                int render_scale_percent =
                    static_cast<int>(std::lround(state.render_scale * 100.0f));
                if (ImGui::SliderInt("Output scale", &render_scale_percent, 25, 100, "%d%%")) {
                    state.render_scale = static_cast<float>(render_scale_percent) / 100.0f;
                    actions.render_scale_changed = true;
                }
            }

            if (has_capability(
                    active_capabilities,
                    RenderModeCapability::ShaderReload) &&
                settings_header("GLSL Shader",[&] {
                    shader_state.auto_reload=true;actions.shader_auto_reload_changed=true;
                    actions.shader_reload_requested=true;
                },"Enable automatic reload and reload the current shader files.")) {
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

            if (settings_header("Display",[&]{state.display=ViewerUiState{}.display;})) {
                // Display settings are applied directly via
                // ui_state.display at present time; no action flag needed.
                ImGui::SliderFloat(
                    "Exposure", &state.display.exposure_ev, -8.0f, 8.0f, "%+.2f EV");
                int tone_mapper = static_cast<int>(state.display.tone_mapper);
                constexpr const char* tone_mappers[] = {"None", "Reinhard", "ACES"};
                if (ImGui::Combo("Tone mapping", &tone_mapper, tone_mappers, 3)) {
                    state.display.tone_mapper = static_cast<ToneMapper>(tone_mapper);
                }
                ImGui::TextDisabled("Current: %s", tone_mapper_label(state.display.tone_mapper));
            }

            if (settings_header("Interface",[&] {
                    state.ui_font_scale=ViewerUiState{}.ui_font_scale;
                    ImGui::GetStyle().FontScaleMain=state.ui_font_scale;
                })) {
                int font_scale_percent =
                    static_cast<int>(std::lround(state.ui_font_scale * 100.0f));
                if (ImGui::SliderInt("Font size", &font_scale_percent, 75, 200, "%d%%")) {
                    state.ui_font_scale = static_cast<float>(font_scale_percent) / 100.0f;
                    ImGui::GetStyle().FontScaleMain = state.ui_font_scale;
                }
            }

            ImGui::Separator();
            ImGui::TextDisabled("G/R/S: transform | Ctrl+R: reset | 1/2/3/4: mode");
            ImGui::TextDisabled("Orbit: LMB rotate | RMB pan | wheel zoom");
            ImGui::TextDisabled("Free: hold RMB + WASD");
        }
        ImGui::End();
    }

    if (state.camera_lighting_panel_visible) {
        if (ImGui::Begin("Camera", &state.camera_lighting_panel_visible)) {
            if (settings_header("Camera",[&]{actions.camera_reset_requested=true;},
                    "Fit the current scene with default field of view, orbit distance and movement speed.")) {
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
            }

        }
        ImGui::End();
    }

    if (state.techniques_panel_visible) {
        if (ImGui::Begin("Techniques", &state.techniques_panel_visible)) {
            const auto reset_environment=[&] {
                const SceneDocument defaults;
                document.set_environment(document.environment_map()?Color::Ones():defaults.environment());
                document.set_environment_intensity(defaults.environment_intensity());
                document.set_environment_rotation_degrees(defaults.environment_rotation_degrees());
                document.set_environment_background_visible(defaults.environment_background_visible());
                render_settings.opengl.ibl_enabled=OpenGlRenderSettings{}.ibl_enabled;
                document.checkpoint();actions.scene_changes|=SceneChange::Environment;
            };
            if (settings_header("Direct Lighting",[&] {
                    const OpenGlRenderSettings defaults;
                    state.show_point_light_markers=ViewerUiState{}.show_point_light_markers;
                    render_settings.opengl.shadow_map=defaults.shadow_map;
                    render_settings.opengl.pcss=defaults.pcss;
                    render_settings.opengl.dominant_light=defaults.dominant_light;
                    render_settings.opengl.ltc_area_lights_enabled=defaults.ltc_area_lights_enabled;
                    reset_environment();
                },"Restore lighting controls and environment adjustments. Keep the HDRI source and scene lights.")) {
                ImGui::Checkbox("Show point light markers", &state.show_point_light_markers);
                settings_subsection("Environment IBL",reset_environment,
                    "Restore tint, intensity, rotation and background visibility. Keep the HDRI source.");
                const bool open_gl_mode = state.mode == InteractiveRenderMode::OpenGl;
                if (open_gl_mode) ImGui::Checkbox("Enable IBL", &render_settings.opengl.ibl_enabled);
                const std::string environment_path = document.environment_path().empty()
                    ? std::string("Constant color")
                    : document.environment_path().filename().string();
                ImGui::TextWrapped("Source: %s", environment_path.c_str());
                if (ImGui::Button("Load HDRI...")) {
                    actions.load_environment_requested = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Clear HDRI") && document.environment_map()) {
                    document.clear_environment_map();
                    document.checkpoint();
                    actions.scene_changes |= SceneChange::Environment;
                }
                bool environment_changed = false;
                bool environment_finished = false;
                Color environment = document.environment();
                if (ImGui::ColorEdit3(
                    "Environment tint",
                    environment.data(),
                    ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)) {
                    document.set_environment(environment);
                    environment_changed = true;
                }
                environment_finished = ImGui::IsItemDeactivatedAfterEdit() || environment_finished;
                float environment_intensity = document.environment_intensity();
                if (ImGui::SliderFloat(
                    "Environment intensity",
                    &environment_intensity,
                    0.0f,
                    32.0f,
                    "%.3f",
                    ImGuiSliderFlags_Logarithmic)) {
                    document.set_environment_intensity(environment_intensity);
                    environment_changed = true;
                }
                environment_finished = ImGui::IsItemDeactivatedAfterEdit() || environment_finished;
                float environment_rotation = document.environment_rotation_degrees();
                if (ImGui::SliderFloat(
                    "Environment rotation",
                    &environment_rotation,
                    -180.0f,
                    180.0f,
                    "%.1f deg")) {
                    document.set_environment_rotation_degrees(environment_rotation);
                    environment_changed = true;
                }
                environment_finished = ImGui::IsItemDeactivatedAfterEdit() || environment_finished;
                bool background_visible = document.environment_background_visible();
                if (ImGui::Checkbox(
                    "Show environment background",
                    &background_visible)) {
                    document.set_environment_background_visible(background_visible);
                    environment_changed = true;
                }
                if (environment_changed) {
                    actions.scene_changes |= SceneChange::Environment;
                }
                if (environment_finished ||
                    (environment_changed && ImGui::IsItemDeactivated())) {
                    document.checkpoint();
                }
                if (open_gl_mode) {
                settings_subsection("Shadow Map",[&]{render_settings.opengl.shadow_map=ShadowMapRenderSettings{};});
                ImGui::BeginDisabled(!open_gl_mode);
                ImGui::Checkbox(
                    "Enable Shadow Map",
                    &render_settings.opengl.shadow_map.enabled);
                ImGui::BeginDisabled(!render_settings.opengl.shadow_map.enabled);
                ImGui::SliderInt(
                    "Resolution",
                    &render_settings.opengl.shadow_map.resolution,
                    128,
                    4096);
                render_settings.opengl.shadow_map.resolution = std::clamp(
                    render_settings.opengl.shadow_map.resolution,
                    128,
                    4096);
                ImGui::SliderInt(
                    "Shadow light budget",
                    &render_settings.opengl.shadow_map.max_shadow_lights,
                    1,
                    32);
                ImGui::DragFloat(
                    "Constant bias",
                    &render_settings.opengl.shadow_map.constant_bias,
                    0.00001f,
                    0.0f,
                    0.05f,
                    "%.6f",
                    ImGuiSliderFlags_AlwaysClamp);
                ImGui::DragFloat(
                    "Slope bias",
                    &render_settings.opengl.shadow_map.slope_bias,
                    0.00005f,
                    0.0f,
                    0.1f,
                    "%.6f",
                    ImGuiSliderFlags_AlwaysClamp);
                ImGui::SliderFloat(
                    "Projection padding",
                    &render_settings.opengl.shadow_map.projection_padding,
                    0.0f,
                    0.5f,
                    "%.3f");
                constexpr const char* debug_views[] = {
                    "Final image",
                    "Visibility",
                    "Blocker depth",
                    "Penumbra radius"};
                int debug_view = static_cast<int>(
                    render_settings.opengl.shadow_map.debug_view);
                if (ImGui::Combo(
                        "Shadow debug view",
                        &debug_view,
                        debug_views,
                        static_cast<int>(std::size(debug_views)))) {
                    render_settings.opengl.shadow_map.debug_view =
                        static_cast<OpenGlShadowDebugView>(debug_view);
                    if (render_settings.opengl.shadow_map.debug_view !=
                        OpenGlShadowDebugView::Final) {
                        render_settings.opengl.ddgi.debug_view = DdgiDebugView::Final;
                        render_settings.opengl.ambient_occlusion.debug_view =
                            OpenGlAmbientOcclusionDebugView::Final;
                        render_settings.opengl.ssr.debug_view =
                            OpenGlSsrDebugView::Final;
                    }
                }
                ImGui::InputInt(
                    "Debug shadow slot",
                    &render_settings.opengl.shadow_map.debug_shadow_slot);
                render_settings.opengl.shadow_map.debug_shadow_slot = std::max(
                    0,
                    render_settings.opengl.shadow_map.debug_shadow_slot);
                ImGui::EndDisabled();
                ImGui::EndDisabled();
                if (open_gl_mode) {
                    ImGui::Text(
                        "Active slots: %d",
                        technique_diagnostics.active_shadow_slots);
                    if (technique_diagnostics.budget_excluded_lights > 0) {
                        ImGui::TextDisabled(
                            "%d light(s) remain lit but do not cast shadows (budget).",
                            technique_diagnostics.budget_excluded_lights);
                    }
                    if (technique_diagnostics.hardware_excluded_lights > 0) {
                        ImGui::TextDisabled(
                            "%d light(s) do not cast shadows (texture-layer limit).",
                            technique_diagnostics.hardware_excluded_lights);
                    }
                }

                settings_subsection("PCSS",[&]{render_settings.opengl.pcss=PcssRenderSettings{};});
                ImGui::BeginDisabled(
                    !open_gl_mode || !render_settings.opengl.shadow_map.enabled);
                ImGui::Checkbox("Enable PCSS", &render_settings.opengl.pcss.enabled);
                ImGui::BeginDisabled(!render_settings.opengl.pcss.enabled);
                ImGui::SliderInt(
                    "Blocker samples",
                    &render_settings.opengl.pcss.blocker_samples,
                    1,
                    64);
                ImGui::SliderInt(
                    "Filter samples",
                    &render_settings.opengl.pcss.filter_samples,
                    1,
                    64);
                ImGui::SliderFloat(
                    "Maximum penumbra",
                    &render_settings.opengl.pcss.max_penumbra_texels,
                    0.0f,
                    256.0f,
                    "%.1f texels");
                ImGui::SliderFloat(
                    "Light size multiplier",
                    &render_settings.opengl.pcss.light_size_scale,
                    0.0f,
                    8.0f,
                    "%.2f");
                ImGui::EndDisabled();
                ImGui::EndDisabled();

                settings_subsection("Dominant Light Extraction",[&]{render_settings.opengl.dominant_light=DominantLightExtractionRenderSettings{};});
                ImGui::BeginDisabled(!open_gl_mode);
                ImGui::Checkbox(
                    "Enable dominant environment light",
                    &render_settings.opengl.dominant_light.enabled);
                ImGui::BeginDisabled(!render_settings.opengl.dominant_light.enabled);
                ImGui::SliderFloat(
                    "Peak threshold",
                    &render_settings.opengl.dominant_light.peak_threshold_ev,
                    0.0f,
                    12.0f,
                    "%.2f EV below peak");
                ImGui::SliderFloat(
                    "Minimum environment energy",
                    &render_settings.opengl.dominant_light.minimum_energy_fraction,
                    0.0f,
                    0.25f,
                    "%.3f");
                ImGui::SliderFloat(
                    "Extracted light intensity",
                    &render_settings.opengl.dominant_light.intensity_scale,
                    0.0f,
                    8.0f,
                    "%.2f");
                ImGui::EndDisabled();
                ImGui::EndDisabled();
                if (open_gl_mode && technique_diagnostics.dominant_light_valid) {
                    const Vec3& direction =
                        technique_diagnostics.dominant_light_direction;
                    const Color& radiance =
                        technique_diagnostics.dominant_light_integrated_radiance;
                    ImGui::Text(
                        "Direction (environment local): %.3f, %.3f, %.3f",
                        direction.x(), direction.y(), direction.z());
                    ImGui::Text(
                        "Integrated RGB: %.3f, %.3f, %.3f",
                        radiance.x(), radiance.y(), radiance.z());
                    ImGui::Text(
                        "Energy: %.2f%%  |  angular radius: %.3f deg",
                        technique_diagnostics.dominant_light_energy_fraction * 100.0f,
                        technique_diagnostics.dominant_light_angular_radius_radians *
                            180.0f / kPi);
                } else if (open_gl_mode && document.environment_map()) {
                    ImGui::TextDisabled("No qualifying highlight region.");
                }

                settings_subsection("LTC Area Lights",[&]{render_settings.opengl.ltc_area_lights_enabled=OpenGlRenderSettings{}.ltc_area_lights_enabled;});
                ImGui::BeginDisabled(!open_gl_mode);
                ImGui::Checkbox(
                    "Enable LTC rectangular lights",
                    &render_settings.opengl.ltc_area_lights_enabled);
                ImGui::EndDisabled();
                ImGui::TextDisabled(
                    "64x64 GGX LTC matrix/amplitude LUT; rectangle shadows use a center cube map approximation.");

                }

                ImGui::SeparatorText("Create Lights");
                if (ImGui::Button("Add point light")) {
                    const Vec3 center = (bounds.min + bounds.max) * 0.5f;
                    const float radius = scene_radius(bounds);
                    const ObjectId id =
                        document.create_point_light("Point Light",
                                                    center + Vec3(0.0f, radius, 0.0f),
                                                    Color(10.0f, 10.0f, 10.0f));
                    document.checkpoint();
                    select_viewer_object(state, id, false);
                    actions.scene_changes |= SceneChange::Lighting;
                }
                ImGui::SameLine();
                if (ImGui::Button("Add directional light")) {
                    const ObjectId id =
                        document.create_directional_light("Directional Light",
                                                          Vec3(-0.5f, -1.0f, -0.25f),
                                                          Color(0.25f, 0.25f, 0.25f));
                    document.checkpoint();
                    select_viewer_object(state, id, false);
                    actions.scene_changes |= SceneChange::Lighting;
                }
                if (ImGui::Button("Add spot light")) {
                    const Vec3 center = (bounds.min + bounds.max) * 0.5f;
                    const float radius = scene_radius(bounds);
                    const Vec3 position = center + Vec3(0.0f, radius, radius);
                    const ObjectId id = document.create_spot_light(
                        "Spot Light",
                        position,
                        center - position,
                        Color(20.0f, 20.0f, 20.0f),
                        0.0f,
                        20.0f * kPi / 180.0f,
                        30.0f * kPi / 180.0f);
                    document.checkpoint();
                    select_viewer_object(state, id, false);
                    actions.scene_changes |= SceneChange::Lighting;
                }
                ImGui::SameLine();
                if (ImGui::Button("Add rectangle light")) {
                    const Vec3 center = (bounds.min + bounds.max) * 0.5f;
                    const float radius = scene_radius(bounds);
                    const Vec3 position = center + Vec3(0.0f, radius, 0.0f);
                    const ObjectId id = document.create_rect_area_light(
                        "Rectangle Light",
                        position,
                        center - position,
                        Color(10.0f, 10.0f, 10.0f),
                        std::max(radius * 0.5f, 0.1f),
                        std::max(radius * 0.5f, 0.1f));
                    document.checkpoint();
                    select_viewer_object(state, id, false);
                    actions.scene_changes = SceneChange::All;
                }
            }

            if (state.mode == InteractiveRenderMode::Rtrt) {
                draw_realtime_panel(render_settings.realtime);
            } else if(state.mode==InteractiveRenderMode::Dxr) {
                auto& s=render_settings.dxr;
                if(settings_header("DXR",[&]{s=DxrRenderSettings{};})) {
                    ImGui::SliderInt("Samples per pixel",&s.samples_per_pixel,1,16);
                    ImGui::SliderInt("Maximum bounces",&s.max_bounces,1,16);
                    ImGui::Checkbox("ReSTIR direct lighting",&s.restir_di);ImGui::Checkbox("ReSTIR path reuse",&s.restir_pt);
                    ImGui::Checkbox("Shader execution reordering (Auto)",&s.shader_execution_reordering);
                    ImGui::Checkbox("Opacity micromaps (Auto)",&s.opacity_micromaps);
                    ImGui::Checkbox("Specular motion / PSR",&s.specular_antialiasing);
                    ImGui::Checkbox("Material color detail (TAAU)",&s.full_resolution_materials);
                    const char* reconstructions[]={"Auto","NRD RELAX + TAAU","NRD RELAX + DLSS SR","DLSS Ray Reconstruction","Reference (raw path tracing)"};
                    int reconstruction=int(s.reconstruction);if(ImGui::Combo("Reconstruction",&reconstruction,reconstructions,5))s.reconstruction=DxrReconstruction(reconstruction);
                    if(s.reconstruction==DxrReconstruction::NrdTaau)ImGui::SliderFloat("Internal linear scale",&s.internal_scale,.25f,1.f,"%.3f");
                    ImGui::SliderInt("DI candidates",&s.di_candidates,1,32);ImGui::SliderInt("DI spatial samples",&s.spatial_samples,0,16);
                    ImGui::SliderInt("PT spatial samples",&s.pt_spatial_samples,0,16);
                    ImGui::SliderInt("PT disocclusion samples",&s.pt_disocclusion_samples,s.pt_spatial_samples,16);
                    ImGui::SliderInt("Reuse history length",&s.history_length,1,64);
                    const char* views[]={"Final","Raw","Light samples + visible emission","Path contribution (includes BRDF direct)","Albedo","Normal","Depth","Motion","Specular motion","Reservoir age","Reservoir weight","History rejection","NRD validation","BRDF hit distance (diffuse / specular)","TAA history (red age / green rejection)"};
                    int view=int(s.debug_view);if(ImGui::Combo("Debug view",&view,views,IM_ARRAYSIZE(views)))s.debug_view=DxrDebugView(view);
                }
            } else {
            if (settings_header("Ambient Occlusion",[&]{render_settings.opengl.ambient_occlusion=AmbientOcclusionRenderSettings{};})) {
                const bool open_gl_mode =
                    state.mode == InteractiveRenderMode::OpenGl;
                auto& ao = render_settings.opengl.ambient_occlusion;
                ImGui::BeginDisabled(!open_gl_mode);

                constexpr const char* modes[] = {"Off", "SSAO", "GTAO"};
                int mode = static_cast<int>(ao.mode);
                if (ImGui::Combo(
                        "AO mode",
                        &mode,
                        modes,
                        static_cast<int>(std::size(modes)))) {
                    ao.mode = static_cast<OpenGlAmbientOcclusionMode>(mode);
                    if (ao.mode == OpenGlAmbientOcclusionMode::Off) {
                        ao.debug_view =
                            OpenGlAmbientOcclusionDebugView::Final;
                    }
                }

                if (ao.mode == OpenGlAmbientOcclusionMode::Ssao) {
                    settings_subsection("SSAO",[&]{ao.ssao=SsaoRenderSettings{};});
                    ImGui::SliderInt(
                        "Samples##SSAO", &ao.ssao.sample_count, 8, 64);
                    ImGui::SliderFloat(
                        "Radius / scene radius##SSAO",
                        &ao.ssao.radius_scale,
                        0.005f,
                        0.5f,
                        "%.3f",
                        ImGuiSliderFlags_Logarithmic);
                    ImGui::SliderFloat(
                        "Depth bias / radius",
                        &ao.ssao.depth_bias_fraction,
                        0.0f,
                        0.2f,
                        "%.3f");
                    ImGui::SliderFloat(
                        "Intensity##SSAO",
                        &ao.ssao.intensity,
                        0.0f,
                        4.0f,
                        "%.2f");
                } else if (ao.mode == OpenGlAmbientOcclusionMode::Gtao) {
                    settings_subsection("GTAO + Bent Normal",[&]{ao.gtao=GtaoRenderSettings{};});
                    ImGui::SliderInt(
                        "Horizon slices", &ao.gtao.slice_count, 1, 8);
                    ImGui::SliderInt(
                        "Samples per side",
                        &ao.gtao.samples_per_side,
                        1,
                        8);
                    ImGui::SliderFloat(
                        "Radius / scene radius##GTAO",
                        &ao.gtao.radius_scale,
                        0.005f,
                        0.5f,
                        "%.3f",
                        ImGuiSliderFlags_Logarithmic);
                    ImGui::SliderFloat(
                        "Falloff fraction",
                        &ao.gtao.falloff_fraction,
                        0.05f,
                        1.0f,
                        "%.2f");
                    ImGui::SliderFloat(
                        "Thickness fraction",
                        &ao.gtao.thickness_fraction,
                        0.0f,
                        1.0f,
                        "%.2f");
                    ImGui::SliderFloat(
                        "Intensity##GTAO",
                        &ao.gtao.intensity,
                        0.0f,
                        4.0f,
                        "%.2f");
                    ImGui::Checkbox(
                        "Enable Bent Normal",
                        &ao.gtao.bent_normals_enabled);
                }

                settings_subsection("Spatial filter",[&]{ao.denoise=AmbientOcclusionDenoiseSettings{};});
                ImGui::Checkbox("Enable AO denoise", &ao.denoise.enabled);
                ImGui::BeginDisabled(!ao.denoise.enabled);
                ImGui::SliderInt(
                    "Kernel radius", &ao.denoise.kernel_radius, 1, 4);
                ImGui::SliderFloat(
                    "Depth sigma / radius",
                    &ao.denoise.depth_sigma_fraction,
                    0.01f,
                    1.0f,
                    "%.3f",
                    ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat(
                    "Normal power",
                    &ao.denoise.normal_power,
                    1.0f,
                    64.0f,
                    "%.1f",
                    ImGuiSliderFlags_Logarithmic);
                ImGui::EndDisabled();

                constexpr const char* debug_views[] = {
                    "Final image",
                    "AO visibility",
                    "Bent Normal",
                    "View Normal",
                    "Linear Depth"};
                int ao_debug_view = static_cast<int>(ao.debug_view);
                ImGui::BeginDisabled(
                    ao.mode == OpenGlAmbientOcclusionMode::Off);
                if (ImGui::Combo(
                        "AO debug view",
                        &ao_debug_view,
                        debug_views,
                        static_cast<int>(std::size(debug_views)))) {
                    ao.debug_view = static_cast<OpenGlAmbientOcclusionDebugView>(
                        ao_debug_view);
                    if (ao.debug_view !=
                        OpenGlAmbientOcclusionDebugView::Final) {
                        render_settings.opengl.ddgi.debug_view = DdgiDebugView::Final;
                        render_settings.opengl.shadow_map.debug_view =
                            OpenGlShadowDebugView::Final;
                        render_settings.opengl.ssr.debug_view =
                            OpenGlSsrDebugView::Final;
                    }
                }
                ImGui::EndDisabled();
                ImGui::EndDisabled();

                const float radius_scale =
                    ao.mode == OpenGlAmbientOcclusionMode::Ssao
                    ? ao.ssao.radius_scale
                    : ao.gtao.radius_scale;
                ImGui::Text(
                    "Resolution: %d x %d  |  world radius: %.4g",
                    render_settings.width,
                    render_settings.height,
                    scene_radius(bounds) * radius_scale);
                ImGui::TextDisabled(
                    "AO attenuates indirect lighting, including DDGI and environment IBL.");
            }

            if (settings_header("DDGI - Dynamic Diffuse GI",[&]{reset_ddgi_settings(render_settings.opengl.ddgi);},
                    "Restore DDGI defaults, refit the volume and restart probe history.")) {
                auto& ddgi = render_settings.opengl.ddgi;
                const auto& diagnostic = shader_state.ddgi;
                ImGui::Checkbox("Enable DDGI", &ddgi.enabled);
                ImGui::Text("Probes: %s", diagnostic.status.c_str());
                if (!diagnostic.detail.empty()) ImGui::TextWrapped("%s", diagnostic.detail.c_str());
                ImGui::BeginDisabled(!ddgi.enabled);
                if (ImGui::Checkbox("Fit volume to scene", &ddgi.auto_fit) && !ddgi.auto_fit && diagnostic.layout.count() > 0) {
                    ddgi.origin = diagnostic.layout.origin;
                    for (int a = 0; a < 3; ++a) ddgi.extent[a] = diagnostic.layout.spacing[a] * float(diagnostic.layout.counts[a] - 1);
                }
                ImGui::BeginDisabled(ddgi.auto_fit);
                ImGui::DragFloat3("Volume minimum", ddgi.origin.data(), 0.02f);
                ImGui::DragFloat3("Volume extent", ddgi.extent.data(), 0.02f, 0.001f, 1.0e6f);
                ImGui::EndDisabled();
                ImGui::SliderInt3("Probe grid", ddgi.probe_counts.data(), 2, 32);
                ImGui::SliderInt("Rays per probe", &ddgi.rays_per_probe, 32, 512);
                ImGui::SliderInt("Probes per frame", &ddgi.probes_per_frame, 1,
                    ddgi.probe_counts[0] * ddgi.probe_counts[1] * ddgi.probe_counts[2]);
                ImGui::SliderFloat("History weight", &ddgi.hysteresis, 0, 0.999f, "%.3f");
                ImGui::SliderFloat("Normal bias / cell", &ddgi.normal_bias, 0, 0.5f);
                ImGui::SliderFloat("View bias / cell", &ddgi.view_bias, 0, 0.5f);
                ImGui::SliderFloat("Indirect intensity", &ddgi.intensity, 0, 4);
                ImGui::Checkbox("Relocate probes", &ddgi.relocation); ImGui::SameLine();
                ImGui::Checkbox("Classify probes", &ddgi.classification);
                ImGui::Checkbox("Pause probe updates", &ddgi.paused);
                ImGui::Checkbox("Show probes", &ddgi.show_probes);
                if (ImGui::Button("Refit volume")) { ddgi.auto_fit = true; ++ddgi.fit_generation; }
                ImGui::SameLine(); if (ImGui::Button("Reset probes")) ++ddgi.reset_generation;
                const char* views[] = {"Final", "Diffuse indirect", "Probe state", "Update age", "Irradiance atlas", "Distance atlas"};
                int view = static_cast<int>(ddgi.debug_view);
                if (ImGui::Combo("DDGI debug view", &view, views, 6)) {
                    ddgi.debug_view = static_cast<DdgiDebugView>(view);
                    if (view != 0) {
                        render_settings.opengl.ssr.debug_view = OpenGlSsrDebugView::Final;
                        render_settings.opengl.shadow_map.debug_view = OpenGlShadowDebugView::Final;
                        render_settings.opengl.ambient_occlusion.debug_view = OpenGlAmbientOcclusionDebugView::Final;
                    }
                }
                ImGui::EndDisabled();
                ddgi = normalized_ddgi_settings(ddgi);
                ImGui::Text("Active %d / %d | updated %d | oldest %d frames", diagnostic.active_probes,
                    diagnostic.probe_count, diagnostic.updated_probes, diagnostic.maximum_age);
                ImGui::Text("Trace %.2f ms | blend %.2f ms | export %.2f ms | gather %.2f ms",
                    diagnostic.trace_ms, diagnostic.blend_ms, diagnostic.export_ms, diagnostic.gather_ms);
                ImGui::Text("Probe memory %.1f MiB | atlas readbacks %llu",
                    double(diagnostic.memory_bytes) / 1048576.0, static_cast<unsigned long long>(diagnostic.atlas_downloads));
                ImGui::TextWrapped("DDGI updates diffuse lighting in world space. SSR supplies glossy reflections. Camera motion keeps the probe history.");
            }

            if (settings_header("SSR - Screen Space Ray Tracing",[&]{render_settings.opengl.ssr=SsrRenderSettings{};})) {
                const bool open_gl_mode =
                    state.mode == InteractiveRenderMode::OpenGl;
                auto& ssr = render_settings.opengl.ssr;
                ImGui::BeginDisabled(!open_gl_mode);
                ImGui::Checkbox("Enable SSR", &ssr.enabled);
                ImGui::BeginDisabled(!ssr.enabled);
                ImGui::SliderInt(
                    "Rays per pixel##SSR",
                    &ssr.rays_per_pixel,
                    1,
                    kOpenGlMaxSsrRays);
                ImGui::SliderInt(
                    "Hi-Z cell visits##SSR",
                    &ssr.max_steps,
                    8,
                    kOpenGlMaxSsrSteps);
                ImGui::SliderFloat(
                    "Max distance / scene radius##SSR",
                    &ssr.max_distance_scale,
                    0.05f,
                    4.0f,
                    "%.3f",
                    ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat(
                    "Thickness / distance##SSR",
                    &ssr.thickness_scale,
                    0.0005f,
                    0.1f,
                    "%.5f",
                    ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat(
                    "Edge fade##SSR",
                    &ssr.edge_fade,
                    0.0f,
                    0.5f,
                    "%.3f");
                ImGui::SliderInt(
                    "History frames##SSR",
                    &ssr.max_history_frames,
                    1,
                    64);
                ImGui::SliderInt(
                    "A-trous passes##SSR",
                    &ssr.denoise_passes,
                    0,
                    kOpenGlMaxSsrDenoisePasses);
                ImGui::SliderFloat(
                    "Depth sigma / distance##SSR",
                    &ssr.denoise_depth_sigma_fraction,
                    0.005f,
                    0.5f,
                    "%.3f",
                    ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat(
                    "Normal power##SSR",
                    &ssr.denoise_normal_power,
                    1.0f,
                    64.0f,
                    "%.1f",
                    ImGuiSliderFlags_Logarithmic);
                ImGui::EndDisabled();

                ssr.rays_per_pixel = std::clamp(
                    ssr.rays_per_pixel, 1, kOpenGlMaxSsrRays);
                ssr.max_steps = std::clamp(
                    ssr.max_steps, 8, kOpenGlMaxSsrSteps);
                ssr.max_distance_scale = std::clamp(
                    ssr.max_distance_scale, 0.05f, 4.0f);
                ssr.thickness_scale = std::clamp(
                    ssr.thickness_scale, 0.0005f, 0.1f);
                ssr.edge_fade = std::clamp(ssr.edge_fade, 0.0f, 0.5f);
                ssr.max_history_frames = std::clamp(
                    ssr.max_history_frames, 1, 64);
                ssr.denoise_passes = std::clamp(
                    ssr.denoise_passes,
                    0,
                    kOpenGlMaxSsrDenoisePasses);
                ssr.denoise_depth_sigma_fraction = std::clamp(
                    ssr.denoise_depth_sigma_fraction, 0.005f, 0.5f);
                ssr.denoise_normal_power = std::clamp(
                    ssr.denoise_normal_power, 1.0f, 64.0f);

                constexpr const char* debug_views[] = {
                    "Final image",
                    "Raw indirect",
                    "Occluded ray fraction",
                    "Temporal indirect",
                    "Filtered indirect",
                    "History length"};
                int debug_view = static_cast<int>(ssr.debug_view);
                if (ImGui::Combo(
                        "SSR debug view",
                        &debug_view,
                        debug_views,
                        static_cast<int>(std::size(debug_views)))) {
                    ssr.debug_view = static_cast<OpenGlSsrDebugView>(
                        debug_view);
                    if (ssr.debug_view != OpenGlSsrDebugView::Final) {
                        render_settings.opengl.ddgi.debug_view = DdgiDebugView::Final;
                        render_settings.opengl.shadow_map.debug_view =
                            OpenGlShadowDebugView::Final;
                        render_settings.opengl.ambient_occlusion.debug_view =
                            OpenGlAmbientOcclusionDebugView::Final;
                    }
                }
                ImGui::EndDisabled();
                ImGui::Text(
                    "Trace resolution: %d x %d  |  max distance: %.4g",
                    render_settings.width,
                    render_settings.height,
                    scene_radius(bounds) * ssr.max_distance_scale);
                ImGui::TextWrapped(render_settings.opengl.ddgi.enabled && shader_state.ddgi.active
                    ? "DDGI supplies diffuse indirect lighting; SSR traces glossy reflections."
                    : "Diffuse and glossy indirect lighting share ray visibility on opaque/masked surfaces.");
                ImGui::TextDisabled(
                    "Known blockers replace environment lighting; AO applies only to environment fallback.");
                if (!render_settings.opengl.ibl_enabled) {
                    ImGui::TextColored(
                        ImVec4(1.0f, 0.72f, 0.28f, 1.0f),
                        "IBL is disabled: off-screen and missed-ray fallback is black.");
                }
                if (!open_gl_mode) {
                    ImGui::TextDisabled(
                        "Path mode uses traced indirect lighting and does not run SSR.");
                }
            }
            }
        }
        ImGui::End();
    }

    if (state.scene_panel_visible) {
        ImGui::SetNextWindowSizeConstraints(
            ImVec2(380.0f, 360.0f),
            ImVec2(
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()));
        if (ImGui::Begin("Scene", &state.scene_panel_visible)) {
            if (ImGui::Button("Import asset...")) {
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

            ImGui::SeparatorText("Objects");
            ImGui::TextDisabled(
                "%zu objects  |  Ctrl+click for multi-select",
                document.objects().size());
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::InputTextWithHint(
                "##SceneObjectFilter",
                "Filter objects...",
                state.scene_filter.data(),
                state.scene_filter.size());
            const std::string_view object_filter(state.scene_filter.data());
            std::unordered_map<ObjectId, const SceneObject*> objects_by_id;
            std::unordered_map<ObjectId, std::vector<ObjectId>> children_by_parent;
            objects_by_id.reserve(document.objects().size());
            children_by_parent.reserve(document.objects().size());
            for (const SceneObject& object : document.objects()) {
                objects_by_id.emplace(object.id, &object);
                children_by_parent[object.parent_id].push_back(object.id);
            }
            std::unordered_map<ObjectId, bool> filter_match_cache;
            filter_match_cache.reserve(document.objects().size());
            std::function<bool(ObjectId)> subtree_matches = [&](ObjectId id) {
                if (object_filter.empty()) {
                    return true;
                }
                if (const auto cached = filter_match_cache.find(id);
                    cached != filter_match_cache.end()) {
                    return cached->second;
                }
                const auto object = objects_by_id.find(id);
                bool matches = object != objects_by_id.end() &&
                    contains_case_insensitive(object->second->name, object_filter);
                if (const auto children = children_by_parent.find(id);
                    !matches && children != children_by_parent.end()) {
                    for (ObjectId child : children->second) {
                        if (subtree_matches(child)) {
                            matches = true;
                            break;
                        }
                    }
                }
                filter_match_cache.emplace(id, matches);
                return matches;
            };
            bool drew_filtered_object = false;

            std::function<void(ObjectId)> draw_children = [&](ObjectId parent_id) {
                const auto children = children_by_parent.find(parent_id);
                if (children == children_by_parent.end()) {
                    return;
                }
                for (ObjectId id : children->second) {
                    if (!subtree_matches(id)) {
                        continue;
                    }
                    const auto object_entry = objects_by_id.find(id);
                    if (object_entry == objects_by_id.end()) {
                        continue;
                    }
                    const SceneObject* object = object_entry->second;
                    drew_filtered_object = true;
                    ImGui::PushID(static_cast<int>(id));
                    const auto object_children = children_by_parent.find(id);
                    const bool has_children =
                        object_children != children_by_parent.end() &&
                        !object_children->second.empty();
                    ImGuiTreeNodeFlags flags =
                        ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
                    if (!has_children) {
                        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                    }
                    if (is_object_selected(state, id)) {
                        flags |= ImGuiTreeNodeFlags_Selected;
                    }
                    if (!object_filter.empty() && has_children) {
                        ImGui::SetNextItemOpen(true, ImGuiCond_Always);
                    }
                    const bool open = ImGui::TreeNodeEx("object",
                                                        flags,
                                                        "%s%s",
                                                        object->visible ? "" : "[hidden] ",
                                                        object->name.c_str());
                    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                        select_viewer_object(state, id, ImGui::GetIO().KeyCtrl);
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
            if (ImGui::BeginChild(
                    "Outliner",
                    ImVec2(0.0f, 0.0f),
                    ImGuiChildFlags_Borders,
                    ImGuiWindowFlags_HorizontalScrollbar)) {
                draw_children(kInvalidObjectId);
                if (!object_filter.empty() && !drew_filtered_object) {
                    ImGui::TextDisabled("No matching objects");
                }
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }

    if (state.inspector_panel_visible) {
        if (ImGui::Begin("Inspector", &state.inspector_panel_visible)) {
            const SceneObject* active = document.find(state.active_object);
            if (active) {
                char name_buffer[256]{};
                const std::size_t name_size =
                    std::min(active->name.size(), sizeof(name_buffer) - 1);
                std::memcpy(name_buffer, active->name.data(), name_size);
                name_buffer[name_size] = '\0';
                if (ImGui::InputText("Name", name_buffer, sizeof(name_buffer))) {
                    document.set_object_name(active->id, name_buffer);
                }
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    document.checkpoint();
                }
                bool visibility = active->visible;
                if (ImGui::Checkbox("Visible", &visibility)) {
                    document.set_object_visible(active->id, visibility);
                    document.checkpoint();
                    actions.scene_changes |= render_changes_for_subtree(
                        document,
                        active->id,
                        true);
                }
                ImGui::SameLine();
                bool locked = active->locked;
                if (ImGui::Checkbox("Locked", &locked)) {
                    document.set_object_locked(active->id, locked);
                    document.checkpoint();
                }

                ImGui::BeginDisabled(active->locked);
                bool transform_changed = false;
                bool transform_finished = false;
                std::optional<SceneTrs> editable_transform =
                    document.local_trs(active->id);
                if (editable_transform) {
                    transform_changed = ImGui::DragFloat3(
                        "Translation",
                        editable_transform->translation.data(),
                        scene_radius(bounds) * 0.0025f) || transform_changed;
                    transform_finished =
                        ImGui::IsItemDeactivatedAfterEdit() || transform_finished;
                    transform_changed = ImGui::DragFloat3(
                        "Rotation",
                        editable_transform->rotation_degrees.data(),
                        0.25f,
                        -3600.0f,
                        3600.0f,
                        "%.2f deg") || transform_changed;
                    transform_finished =
                        ImGui::IsItemDeactivatedAfterEdit() || transform_finished;
                    transform_changed = ImGui::DragFloat3(
                        "Scale",
                        editable_transform->scale.data(),
                        0.01f,
                        -1000.0f,
                        1000.0f) || transform_changed;
                    transform_finished =
                        ImGui::IsItemDeactivatedAfterEdit() || transform_finished;
                } else {
                    ImGui::TextWrapped(
                        "This object contains shear. Its affine matrix is preserved; "
                        "use the gizmo to edit it without lossy TRS decomposition.");
                }
                if (transform_changed && editable_transform) {
                    for (int axis = 0; axis < 3; ++axis) {
                        if (std::abs(editable_transform->scale[axis]) < 1.0e-4f) {
                            editable_transform->scale[axis] =
                                std::copysign(1.0e-4f, editable_transform->scale[axis]);
                        }
                    }
                    if (document.set_local_trs(active->id, *editable_transform)) {
                        actions.scene_changes |= render_changes_for_subtree(
                            document,
                            active->id,
                            false);
                    }
                }
                if (transform_finished) {
                    document.checkpoint();
                }
                ImGui::EndDisabled();

                if (active->type == SceneObjectType::DirectionalLight) {
                    ImGui::SeparatorText("Direction");
                    const auto direction =
                        directional_light_world_direction(document, active->id);
                    if (direction) {
                        ImGui::Text(
                            "World  X %.3f   Y %.3f   Z %.3f",
                            direction->x(),
                            direction->y(),
                            direction->z());
                    } else {
                        ImGui::TextDisabled("World direction unavailable");
                    }
                    ImGui::TextWrapped(
                        "The yellow scene arrows show the direction the light rays travel.");
                }

                if (active->type == SceneObjectType::Camera) {
                    ImGui::SeparatorText("Scene Camera");
                    const char* projection =
                        active->camera_projection == SceneCameraProjection::Orthographic
                            ? "Orthographic (perspective preview)"
                            : "Perspective";
                    ImGui::Text("Projection: %s", projection);

                    bool camera_changed = false;
                    bool camera_edit_finished = false;
                    SceneCameraProperties camera_properties{
                        active->camera_projection,
                        active->camera_vertical_fov_degrees,
                        active->camera_aspect_ratio,
                        active->camera_x_magnification,
                        active->camera_y_magnification,
                        active->camera_near_plane,
                        active->camera_far_plane};
                    ImGui::BeginDisabled(active->locked);
                    if (active->camera_projection == SceneCameraProjection::Perspective) {
                        camera_changed = ImGui::SliderFloat(
                            "Camera vertical FOV",
                            &camera_properties.vertical_fov_degrees,
                            1.0f,
                            179.0f,
                            "%.2f deg") || camera_changed;
                        camera_edit_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || camera_edit_finished;
                    } else {
                        camera_changed = ImGui::DragFloat(
                            "X magnification",
                            &camera_properties.x_magnification,
                            0.01f,
                            1.0e-4f,
                            100000.0f) || camera_changed;
                        camera_edit_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || camera_edit_finished;
                        camera_changed = ImGui::DragFloat(
                            "Y magnification",
                            &camera_properties.y_magnification,
                            0.01f,
                            1.0e-4f,
                            100000.0f) || camera_changed;
                        camera_edit_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || camera_edit_finished;
                    }
                    camera_changed = ImGui::DragFloat(
                        "Camera near plane",
                        &camera_properties.near_plane,
                        0.001f,
                        1.0e-5f,
                        100000.0f) || camera_changed;
                    camera_edit_finished =
                        ImGui::IsItemDeactivatedAfterEdit() || camera_edit_finished;
                    camera_changed = ImGui::DragFloat(
                        "Camera far plane",
                        &camera_properties.far_plane,
                        0.1f,
                        camera_properties.near_plane + 1.0e-4f,
                        1000000.0f) || camera_changed;
                    camera_edit_finished =
                        ImGui::IsItemDeactivatedAfterEdit() || camera_edit_finished;
                    ImGui::EndDisabled();

                    if (camera_changed) {
                        camera_properties.vertical_fov_degrees = std::clamp(
                            camera_properties.vertical_fov_degrees, 1.0f, 179.0f);
                        camera_properties.x_magnification =
                            std::max(camera_properties.x_magnification, 1.0e-4f);
                        camera_properties.y_magnification =
                            std::max(camera_properties.y_magnification, 1.0e-4f);
                        camera_properties.near_plane =
                            std::max(camera_properties.near_plane, 1.0e-5f);
                        camera_properties.far_plane = std::max(
                            camera_properties.far_plane,
                            camera_properties.near_plane + 1.0e-4f);
                        if (document.set_camera_properties(
                                active->id,
                                camera_properties)) {
                            actions.scene_changes = SceneChange::All;
                        }
                    }
                    if (camera_edit_finished) {
                        document.checkpoint();
                    }
                    if (ImGui::Button("Look through camera")) {
                        actions.look_through_camera = active->id;
                    }
                    if (active->camera_projection == SceneCameraProjection::Orthographic) {
                        ImGui::TextDisabled(
                            "RTRT/OpenGL camera rays are perspective; this view uses a 45 deg preview.");
                    }
                }

                if (active->type == SceneObjectType::Mesh) {
                    const SceneMeshAsset* asset = document.asset_for_object(active->id);
                    if (asset && !asset->local_scene.materials.empty()) {
                        const std::vector<std::size_t> material_slots =
                            referenced_material_slots(*asset);
                        if (state.material_editor_object != active->id) {
                            state.material_editor_object = active->id;
                            state.selected_material_slot = material_slots.front();
                        } else if (std::find(
                                       material_slots.begin(),
                                       material_slots.end(),
                                       state.selected_material_slot) ==
                                   material_slots.end()) {
                            state.selected_material_slot = material_slots.front();
                        }

                        ImGui::SeparatorText("Materials");
                        ImGui::TextDisabled(
                            "%zu used material%s",
                            material_slots.size(),
                            material_slots.size() == 1 ? "" : "s");
                        const auto material_name = [asset](std::size_t slot) {
                            return slot < asset->material_names.size() &&
                                           !asset->material_names[slot].empty()
                                       ? asset->material_names[slot]
                                       : "Material " + std::to_string(slot + 1);
                        };
                        const auto material_label = [&material_name](std::size_t slot) {
                            return material_name(slot) +
                                "  [slot " + std::to_string(slot) + "]";
                        };
                        const std::string current_name =
                            material_label(state.selected_material_slot);
                        if (ImGui::BeginCombo("Material slot", current_name.c_str())) {
                            for (std::size_t slot : material_slots) {
                                const std::string name = material_label(slot);
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
                            ImGui::TextDisabled("Original asset material");
                        }

                        std::optional<SceneMaterialOverride> properties =
                            document.material_properties(active->id, slot);
                        if (properties) {
                            const Material& source_material = asset->local_scene.materials[slot];
                            const bool source_specular_glossiness =
                                source_material.type == MaterialType::Pbr &&
                                source_material.pbr_workflow == PbrWorkflow::SpecularGlossiness;
                            if (source_material.type == MaterialType::Pbr) {
                                ImGui::TextDisabled(
                                    "PBR workflow: %s",
                                    source_specular_glossiness
                                        ? "Specular-Glossiness"
                                        : "Metallic-Roughness");
                            }
                            properties->use_diffuse_texture =
                                properties->use_diffuse_texture &&
                                source_material.diffuse_texture_id >= 0;
                            properties->use_opacity_texture =
                                properties->use_opacity_texture &&
                                source_material.opacity_texture_id >= 0;
                            properties->use_bump_texture = properties->use_bump_texture &&
                                                           source_material.bump_texture_id >= 0;
                            properties->use_base_color_texture =
                                properties->use_base_color_texture &&
                                source_material.base_color_texture_id >= 0;
                            properties->use_metallic_roughness_texture =
                                properties->use_metallic_roughness_texture &&
                                source_material.metallic_roughness_texture_id >= 0;
                            properties->use_normal_texture =
                                properties->use_normal_texture &&
                                source_material.normal_texture_id >= 0;
                            properties->use_occlusion_texture =
                                properties->use_occlusion_texture &&
                                source_material.occlusion_texture_id >= 0;
                            properties->use_emissive_texture =
                                properties->use_emissive_texture &&
                                source_material.emissive_texture_id >= 0;

                            bool material_changed = false;
                            bool material_edit_finished = false;
                            bool checkpoint_immediately = false;
                            ImGui::BeginDisabled(active->locked);

                            int material_type = static_cast<int>(properties->type);
                            constexpr const char* material_types[]{
                                "Diffuse", "Metal", "Dielectric", "Emissive", "PBR"};
                            if (ImGui::Combo("Material type", &material_type, material_types, 5)) {
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

                            if (properties->type == MaterialType::Metal ||
                                properties->type == MaterialType::Pbr) {
                                if (properties->type == MaterialType::Pbr &&
                                    !source_specular_glossiness) {
                                    material_changed =
                                        ImGui::SliderFloat(
                                            "Metallic", &properties->metallic, 0.0f, 1.0f) ||
                                        material_changed;
                                    material_edit_finished =
                                        ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                                }
                                material_changed =
                                    ImGui::SliderFloat(
                                        "Roughness", &properties->roughness, 0.0f, 1.0f) ||
                                    material_changed;
                                material_edit_finished =
                                    ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                                ImGui::TextDisabled("GGX roughness: OpenGL / CUDA RTRT");
                            } else if (properties->type == MaterialType::Dielectric) {
                                material_changed =
                                    ImGui::SliderFloat(
                                        "Index of refraction", &properties->ior, 1.0f, 3.0f) ||
                                    material_changed;
                                material_edit_finished =
                                    ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                                ImGui::TextDisabled("Physical refraction: RTRT");
                            }
                            if (properties->type == MaterialType::Emissive ||
                                properties->type == MaterialType::Pbr) {
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

                            AlphaMode effective_alpha_mode = properties->alpha_mode;
                            if (properties->type == MaterialType::Pbr) {
                                int alpha_mode = static_cast<int>(properties->alpha_mode);
                                constexpr const char* alpha_modes[]{"Opaque", "Mask", "Blend"};
                                if (ImGui::Combo("Alpha mode", &alpha_mode, alpha_modes, 3)) {
                                    properties->alpha_mode = static_cast<AlphaMode>(alpha_mode);
                                    effective_alpha_mode = properties->alpha_mode;
                                    material_changed = true;
                                    checkpoint_immediately = true;
                                }
                            } else {
                                effective_alpha_mode =
                                    (properties->use_opacity_texture &&
                                     source_material.opacity_texture_id >= 0) ||
                                            properties->opacity < 1.0f
                                        ? AlphaMode::Mask
                                        : AlphaMode::Opaque;
                                ImGui::TextDisabled(
                                    "Alpha mode: %s (legacy)",
                                    effective_alpha_mode == AlphaMode::Mask
                                        ? "Mask"
                                        : "Opaque");
                            }

                            const bool opacity_enabled =
                                properties->type != MaterialType::Pbr ||
                                effective_alpha_mode != AlphaMode::Opaque;
                            ImGui::BeginDisabled(!opacity_enabled);
                            material_changed =
                                ImGui::SliderFloat("Opacity", &properties->opacity, 0.0f, 1.0f) ||
                                material_changed;
                            material_edit_finished =
                                ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                            if (!opacity_enabled &&
                                ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                                ImGui::SetTooltip("Opacity is ignored by Opaque materials.");
                            }
                            ImGui::EndDisabled();

                            const bool cutoff_enabled =
                                effective_alpha_mode == AlphaMode::Mask;
                            ImGui::BeginDisabled(!cutoff_enabled);
                            material_changed =
                                ImGui::SliderFloat(
                                    "Alpha cutoff", &properties->alpha_cutoff, 0.0f, 1.0f) ||
                                material_changed;
                            material_edit_finished =
                                ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                            if (!cutoff_enabled &&
                                ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                                ImGui::SetTooltip("Alpha cutoff is used only by Mask materials.");
                            }
                            ImGui::EndDisabled();
                            material_changed =
                                ImGui::DragFloat("Bump scale", &properties->bump_scale, 0.01f) ||
                                material_changed;
                            material_edit_finished =
                                ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                            material_changed =
                                ImGui::DragFloat("Normal scale", &properties->normal_scale, 0.01f) ||
                                material_changed;
                            material_edit_finished =
                                ImGui::IsItemDeactivatedAfterEdit() || material_edit_finished;
                            material_changed =
                                ImGui::SliderFloat(
                                    "Occlusion strength",
                                    &properties->occlusion_strength,
                                    0.0f,
                                    1.0f) || material_changed;
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
                            if (texture_toggle("Use base color texture",
                                               source_material.base_color_texture_id,
                                               properties->use_base_color_texture)) {
                                material_changed = true;
                                checkpoint_immediately = true;
                            }
                            if (texture_toggle("Use metallic-roughness texture",
                                               source_material.metallic_roughness_texture_id,
                                               properties->use_metallic_roughness_texture)) {
                                material_changed = true;
                                checkpoint_immediately = true;
                            }
                            if (texture_toggle("Use normal texture",
                                               source_material.normal_texture_id,
                                               properties->use_normal_texture)) {
                                material_changed = true;
                                checkpoint_immediately = true;
                            }
                            if (texture_toggle("Use occlusion texture",
                                               source_material.occlusion_texture_id,
                                               properties->use_occlusion_texture)) {
                                material_changed = true;
                                checkpoint_immediately = true;
                            }
                            if (texture_toggle("Use emissive texture",
                                               source_material.emissive_texture_id,
                                               properties->use_emissive_texture)) {
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
                    active->type == SceneObjectType::DirectionalLight ||
                    active->type == SceneObjectType::SpotLight ||
                    active->type == SceneObjectType::RectAreaLight) {
                    SceneLightProperties light_properties =
                        light_properties_from_object(*active);
                    const ColorStrengthEditResult light_edit =
                        draw_color_and_strength(
                            "Light",
                            "Light color",
                            "Light intensity",
                            light_properties.color);
                    if (light_edit.changed && document.set_light_properties(
                            active->id,
                            light_properties)) {
                        actions.scene_changes |= SceneChange::Lighting;
                    }
                    if (light_edit.finished) {
                        document.checkpoint();
                    }
                }

                if (active->type == SceneObjectType::PointLight ||
                    active->type == SceneObjectType::DirectionalLight ||
                    active->type == SceneObjectType::SpotLight ||
                    active->type == SceneObjectType::RectAreaLight) {
                    SceneLightProperties light_properties =
                        light_properties_from_object(*active);
                    bool shape_changed = false;
                    bool shape_finished = false;
                    ImGui::BeginDisabled(active->locked);
                    if (active->type == SceneObjectType::PointLight ||
                        active->type == SceneObjectType::SpotLight) {
                        shape_changed = ImGui::DragFloat(
                            "Range (0 = unlimited)",
                            &light_properties.range,
                            0.05f,
                            0.0f,
                            1000000.0f,
                            "%.3f",
                            ImGuiSliderFlags_AlwaysClamp) || shape_changed;
                        shape_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || shape_finished;
                        shape_changed = ImGui::DragFloat(
                            "Source radius",
                            &light_properties.source_radius,
                            0.005f,
                            0.0f,
                            100000.0f,
                            "%.4f",
                            ImGuiSliderFlags_AlwaysClamp) || shape_changed;
                        shape_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || shape_finished;
                    }
                    if (active->type == SceneObjectType::SpotLight) {
                        shape_changed = ImGui::SliderAngle(
                            "Inner cone",
                            &light_properties.spot_inner_cone_radians,
                            0.0f,
                            90.0f) || shape_changed;
                        shape_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || shape_finished;
                        shape_changed = ImGui::SliderAngle(
                            "Outer cone",
                            &light_properties.spot_outer_cone_radians,
                            0.0f,
                            90.0f) || shape_changed;
                        shape_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || shape_finished;
                    }
                    if (active->type == SceneObjectType::DirectionalLight) {
                        shape_changed = ImGui::SliderAngle(
                            "Angular radius",
                            &light_properties.directional_angular_radius_radians,
                            0.0f,
                            10.0f,
                            "%.3f deg") || shape_changed;
                        shape_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || shape_finished;
                    }
                    if (active->type == SceneObjectType::RectAreaLight) {
                        shape_changed = ImGui::DragFloat(
                            "Width",
                            &light_properties.area_width,
                            0.01f,
                            0.0001f,
                            100000.0f,
                            "%.3f",
                            ImGuiSliderFlags_AlwaysClamp) || shape_changed;
                        shape_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || shape_finished;
                        shape_changed = ImGui::DragFloat(
                            "Height",
                            &light_properties.area_height,
                            0.01f,
                            0.0001f,
                            100000.0f,
                            "%.3f",
                            ImGuiSliderFlags_AlwaysClamp) || shape_changed;
                        shape_finished =
                            ImGui::IsItemDeactivatedAfterEdit() || shape_finished;
                        if (ImGui::Checkbox("Two-sided emission", &light_properties.two_sided)) {
                            shape_changed = true;
                            shape_finished = true;
                        }
                    }
                    if (ImGui::Checkbox("Cast shadows", &light_properties.casts_shadows)) {
                        shape_changed = true;
                        shape_finished = true;
                    }
                    if (ImGui::InputInt(
                            "Shadow priority",
                            &light_properties.shadow_priority)) {
                        shape_changed = true;
                    }
                    shape_finished =
                        ImGui::IsItemDeactivatedAfterEdit() || shape_finished;
                    ImGui::EndDisabled();
                    if (shape_changed) {
                        light_properties.range = std::max(0.0f, light_properties.range);
                        light_properties.source_radius =
                            std::max(0.0f, light_properties.source_radius);
                        light_properties.spot_inner_cone_radians = std::clamp(
                            light_properties.spot_inner_cone_radians,
                            0.0f,
                            0.5f * kPi);
                        light_properties.directional_angular_radius_radians =
                            std::clamp(
                                light_properties.directional_angular_radius_radians,
                                0.0f,
                                0.5f * kPi);
                        light_properties.area_width =
                            std::max(0.0001f, light_properties.area_width);
                        light_properties.area_height =
                            std::max(0.0001f, light_properties.area_height);
                        light_properties.spot_outer_cone_radians = std::clamp(
                            light_properties.spot_outer_cone_radians,
                            light_properties.spot_inner_cone_radians,
                            0.5f * kPi);
                        if (document.set_light_properties(
                                active->id,
                                light_properties)) {
                            actions.scene_changes |= SceneChange::Lighting;
                        }
                    }
                    if (shape_finished) {
                        document.checkpoint();
                    }
                }

                if (ImGui::Button("Duplicate")) {
                    const ObjectId copy = document.duplicate_subtree(active->id);
                    if (copy != kInvalidObjectId) {
                        select_viewer_object(state, copy, false);
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

SceneChangeSet ViewerUi::draw_scene_gizmo(
    ViewerUiState& state,
    SceneDocument& document,
    const Camera& camera,
    const Bounds3& bounds) {
    std::erase_if(
        state.selected_objects,
        [&document](ObjectId id) { return document.find(id) == nullptr; });
    if (state.selected_objects.empty()) {
        state.active_object = kInvalidObjectId;
        state.gizmo_was_using = false;
        state.gizmo_hovered = false;
        return SceneChange::None;
    }
    if (std::find(
            state.selected_objects.begin(),
            state.selected_objects.end(),
            state.active_object) == state.selected_objects.end()) {
        state.active_object = state.selected_objects.back();
        state.gizmo_was_using = false;
    }
    const SceneObject* active = document.find(state.active_object);
    if (!active || active->locked) {
        state.gizmo_was_using = false;
        state.gizmo_hovered = false;
        return SceneChange::None;
    }

    ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    const ImVec2 display_size = main_viewport->Size;
    if (display_size.x <= 0.0f || display_size.y <= 0.0f) {
        return SceneChange::None;
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
    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
    switch (state.gizmo_operation) {
        case 1:
            operation = ImGuizmo::ROTATE;
            break;
        case 2:
            operation = ImGuizmo::SCALE;
            break;
        default:
            state.gizmo_operation = 0;
            break;
    }
    const ImGuizmo::MODE mode = state.gizmo_local ? ImGuizmo::LOCAL : ImGuizmo::WORLD;

    // Keep ImGuizmo's active handle state isolated per object and operation.
    // Without a scoped ID, switching from rotate to scale can inherit the
    // previous handle's internal state and make the displayed/active tool lag.
    ImGuizmo::PushID(
        reinterpret_cast<const void*>(
            static_cast<std::uintptr_t>(active->id)));
    ImGuizmo::PushID(state.gizmo_operation);
    const bool changed =
        ImGuizmo::Manipulate(view.data(), projection.data(), operation, mode, manipulated.data());
    const bool using_gizmo = ImGuizmo::IsUsing();
    state.gizmo_hovered = ImGuizmo::IsOver(operation) || using_gizmo;
    ImGuizmo::PopID();
    ImGuizmo::PopID();
    std::vector<ObjectId> roots;
    std::vector<ObjectId> transformed_roots;
    if (changed) {
        const Mat4 delta = manipulated * old_active_world.inverse();
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
            const Mat4 target_world = id == active->id
                ? manipulated
                : delta * document.world_matrix(id);
            if (document.set_world_matrix(id, target_world)) {
                transformed_roots.push_back(id);
            }
        }
    }
    if (state.gizmo_was_using && !using_gizmo) {
        document.checkpoint();
    }
    state.gizmo_was_using = using_gizmo;
    if (!changed || transformed_roots.empty()) {
        return SceneChange::None;
    }
    SceneChangeSet changes = SceneChange::None;
    for (ObjectId id : transformed_roots) {
        changes |= render_changes_for_subtree(
            document,
            id,
            false);
    }
    return changes;
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
        const ImU32 color = id == state.active_object
            ? IM_COL32(255, 183, 40, 255)
            : IM_COL32(80, 190, 255, 230);
        const SceneObject* object = document.find(id);
        if (object && object->type == SceneObjectType::RectAreaLight) {
            const Mat4 world = document.world_matrix(id);
            const std::array<Vec3, 4> local_corners{
                Vec3(-0.5f * object->area_width, -0.5f * object->area_height, 0.0f),
                Vec3(0.5f * object->area_width, -0.5f * object->area_height, 0.0f),
                Vec3(0.5f * object->area_width, 0.5f * object->area_height, 0.0f),
                Vec3(-0.5f * object->area_width, 0.5f * object->area_height, 0.0f)};
            std::array<ImVec2, 4> rectangle_points;
            bool rectangle_valid = true;
            for (std::size_t corner = 0; corner < local_corners.size(); ++corner) {
                const Vec4 transformed = world * Vec4(
                    local_corners[corner].x(),
                    local_corners[corner].y(),
                    local_corners[corner].z(),
                    1.0f);
                const auto projected = project_to_screen(
                    transformed.head<3>() / transformed.w(),
                    camera,
                    display_size,
                    display_origin);
                if (!projected) {
                    rectangle_valid = false;
                    break;
                }
                rectangle_points[corner] = projected->screen;
            }
            if (rectangle_valid) {
                for (std::size_t corner = 0; corner < rectangle_points.size(); ++corner) {
                    const ImVec2 start = rectangle_points[corner];
                    const ImVec2 end = rectangle_points[
                        (corner + 1U) % rectangle_points.size()];
                    draw_list->AddLine(
                        start,
                        end,
                        IM_COL32(10, 10, 10, 230),
                        3.0f);
                    draw_list->AddLine(start, end, color, 1.25f);
                }
            }
            continue;
        }
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
    const SceneDocument& document,
    const Camera& camera) const {
    const auto& point_lights =
        document.render_scene_snapshot().point_lights;
    if (!state.show_point_light_markers || point_lights.empty()) {
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
    for (const PointLight& light : point_lights) {
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

void ViewerUi::draw_directional_light_indicator(
    const ViewerUiState& state,
    const SceneDocument& document,
    const Camera& camera,
    const Bounds3& bounds) const {
    if (!is_object_selected(state, state.active_object)) {
        return;
    }
    const SceneObject* active = document.find(state.active_object);
    const auto direction =
        directional_light_world_direction(document, state.active_object);
    if (!active || !active->visible || !direction) {
        return;
    }

    ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    const ImVec2 display_size = main_viewport->Size;
    const ImVec2 display_origin = main_viewport->Pos;
    if (!std::isfinite(display_size.x) || !std::isfinite(display_size.y) ||
        display_size.x <= 0.0f || display_size.y <= 0.0f) {
        return;
    }
    ImVec2 label_region_origin = display_origin;
    ImVec2 label_region_size = display_size;
    if (const ImGuiDockNode* central_node =
            ImGui::DockBuilderGetCentralNode(ImHashStr("RendererDockSpace"));
        central_node &&
        central_node->Size.x > 0.0f &&
        central_node->Size.y > 0.0f) {
        label_region_origin = central_node->Pos;
        label_region_size = central_node->Size;
    }

    const Vec3 center = (bounds.min + bounds.max) * 0.5f;
    if (!center.allFinite()) {
        return;
    }
    const auto projected_center =
        project_to_screen(center, camera, display_size, display_origin);
    if (!projected_center) {
        return;
    }

    const float radius = scene_radius(bounds);
    const float half_length = radius * 0.72f;
    Vec3 side = camera.right() -
        direction.value() * camera.right().dot(direction.value());
    if (!side.allFinite() || side.squaredNorm() <= 1.0e-8f) {
        side = camera.up() -
            direction.value() * camera.up().dot(direction.value());
    }
    if (side.allFinite() && side.squaredNorm() > 1.0e-8f) {
        side.normalize();
    } else {
        side = Vec3::Zero();
    }

    const Color light_tint = normalized_marker_color(active->light_color);
    const Color warm_tint =
        (light_tint * 0.55f + Color(1.0f, 0.63f, 0.08f) * 0.45f)
            .cwiseMin(1.0f);
    const ImU32 color = ImGui::ColorConvertFloat4ToU32(
        ImVec4(warm_tint.x(), warm_tint.y(), warm_tint.z(), 1.0f));
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList(main_viewport);

    bool drew_planar_arrow = false;
    ImVec2 label_anchor = projected_center->screen;
    constexpr std::array<float, 3> ray_offsets{-1.0f, 0.0f, 1.0f};
    for (float ray_offset : ray_offsets) {
        const Vec3 offset = side * (ray_offset * radius * 0.075f);
        const auto start = project_to_screen(
            center - direction.value() * half_length + offset,
            camera,
            display_size,
            display_origin);
        const auto end = project_to_screen(
            center + direction.value() * half_length + offset,
            camera,
            display_size,
            display_origin);
        if (!start || !end) {
            continue;
        }
        const float screen_dx = end->screen.x - start->screen.x;
        const float screen_dy = end->screen.y - start->screen.y;
        const float screen_length =
            std::sqrt(screen_dx * screen_dx + screen_dy * screen_dy);
        if (!std::isfinite(screen_length) || screen_length < 42.0f) {
            continue;
        }
        const bool center_ray = ray_offset == 0.0f;
        draw_outlined_arrow(
            draw_list,
            start->screen,
            end->screen,
            color,
            center_ray ? 4.0f : 2.0f,
            center_ray ? 19.0f : 13.0f);
        if (center_ray) {
            label_anchor = ImVec2(
                start->screen.x + screen_dx * 0.58f,
                start->screen.y + screen_dy * 0.58f);
            draw_list->AddCircleFilled(
                start->screen,
                9.0f,
                IM_COL32(12, 12, 12, 245));
            draw_list->AddCircleFilled(start->screen, 6.0f, color);
            constexpr float two_pi = 6.28318530717958647692f;
            for (int spoke = 0; spoke < 8; ++spoke) {
                const float angle =
                    static_cast<float>(spoke) * (two_pi / 8.0f);
                const ImVec2 spoke_start(
                    start->screen.x + std::cos(angle) * 10.5f,
                    start->screen.y + std::sin(angle) * 10.5f);
                const ImVec2 spoke_end(
                    start->screen.x + std::cos(angle) * 15.0f,
                    start->screen.y + std::sin(angle) * 15.0f);
                draw_list->AddLine(
                    spoke_start,
                    spoke_end,
                    IM_COL32(12, 12, 12, 245),
                    4.5f);
                draw_list->AddLine(spoke_start, spoke_end, color, 2.0f);
            }
        }
        drew_planar_arrow = true;
    }

    const float view_depth = direction->dot(camera.forward());
    const char* label = "LIGHT TRAVELS THIS WAY";
    if (view_depth > 0.35f) {
        label = "LIGHT TRAVELS INTO SCENE";
    } else if (view_depth < -0.35f) {
        label = "LIGHT TRAVELS TOWARD CAMERA";
    }

    if (!drew_planar_arrow) {
        constexpr float symbol_radius = 22.0f;
        draw_list->AddCircleFilled(
            projected_center->screen,
            symbol_radius + 3.0f,
            IM_COL32(12, 12, 12, 235));
        draw_list->AddCircle(
            projected_center->screen,
            symbol_radius,
            color,
            0,
            4.0f);
        if (view_depth >= 0.0f) {
            const float arm = symbol_radius * 0.55f;
            draw_list->AddLine(
                ImVec2(
                    projected_center->screen.x - arm,
                    projected_center->screen.y - arm),
                ImVec2(
                    projected_center->screen.x + arm,
                    projected_center->screen.y + arm),
                color,
                4.0f);
            draw_list->AddLine(
                ImVec2(
                    projected_center->screen.x - arm,
                    projected_center->screen.y + arm),
                ImVec2(
                    projected_center->screen.x + arm,
                    projected_center->screen.y - arm),
                color,
                4.0f);
            label = "LIGHT TRAVELS INTO SCENE";
        } else {
            draw_list->AddCircleFilled(
                projected_center->screen,
                7.0f,
                color);
            label = "LIGHT TRAVELS TOWARD CAMERA";
        }
        label_anchor = ImVec2(
            projected_center->screen.x + symbol_radius,
            projected_center->screen.y);
    }

    draw_direction_label(
        draw_list,
        label_anchor,
        label_region_origin,
        label_region_size,
        label);
}

}  // namespace renderer
