#include "interactive/viewer_session.h"

#include "core/io/atomic_file.h"
#include "render/pathtracer/cuda_pathtracer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace renderer {

// Current viewer session schema version. Load applies version-gated
// upgrades oldest-first (v1 legacy backend note, v3 OpenGL techniques,
// v5 temporal SSGI);
// save() always writes this version.
constexpr int kViewerSessionVersion = 5;

namespace {

nlohmann::json vec3_json(const Vec3& value) {
    return nlohmann::json::array({value.x(), value.y(), value.z()});
}

Vec3 parse_vec3(const nlohmann::json& value, const char* field) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string(field) + " must contain three numbers");
    }
    const Vec3 result(
        value.at(0).get<float>(),
        value.at(1).get<float>(),
        value.at(2).get<float>());
    if (!result.allFinite()) {
        throw std::runtime_error(std::string(field) + " must be finite");
    }
    return result;
}

const char* render_mode_name(InteractiveRenderMode mode) {
    return render_mode_descriptor(mode).cli_name;
}

InteractiveRenderMode parse_render_mode(const std::string& value) {
    try {
        return parse_interactive_render_mode(value);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("unknown viewer render mode: " + value);
    }
}

const char* camera_mode_name(ViewerCameraMode mode) {
    return mode == ViewerCameraMode::Free ? "free" : "orbit";
}

ViewerCameraMode parse_camera_mode(const std::string& value) {
    if (value == "orbit") {
        return ViewerCameraMode::Orbit;
    }
    if (value == "free") {
        return ViewerCameraMode::Free;
    }
    throw std::runtime_error("unknown viewer camera mode: " + value);
}

const char* tone_mapper_name(ToneMapper tone_mapper) {
    switch (tone_mapper) {
        case ToneMapper::None:
            return "none";
        case ToneMapper::Reinhard:
            return "reinhard";
        case ToneMapper::Aces:
            return "aces";
    }
    return "none";
}

ToneMapper parse_tone_mapper(const std::string& value) {
    if (value == "none") {
        return ToneMapper::None;
    }
    if (value == "reinhard") {
        return ToneMapper::Reinhard;
    }
    if (value == "aces") {
        return ToneMapper::Aces;
    }
    throw std::runtime_error("unknown tone mapper: " + value);
}

float finite_clamped(
    const nlohmann::json& object,
    const char* field,
    float minimum,
    float maximum) {
    const float value = object.at(field).get<float>();
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(field) + " must be finite");
    }
    return std::clamp(value, minimum, maximum);
}

float finite_value_clamped(
    const nlohmann::json& object,
    const char* field,
    float fallback,
    float minimum,
    float maximum) {
    const float value = object.value(field, fallback);
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(field) + " must be finite");
    }
    return std::clamp(value, minimum, maximum);
}

float finite_value_or_default_clamped(
    const nlohmann::json& object,
    const char* field,
    float fallback,
    float minimum,
    float maximum) {
    const auto iterator = object.find(field);
    if (iterator == object.end() || !iterator->is_number()) {
        return std::clamp(fallback, minimum, maximum);
    }
    const float value = iterator->get<float>();
    return std::isfinite(value)
        ? std::clamp(value, minimum, maximum)
        : std::clamp(fallback, minimum, maximum);
}

int integer_value_or_default_clamped(
    const nlohmann::json& object,
    const char* field,
    int fallback,
    int minimum,
    int maximum) {
    const auto iterator = object.find(field);
    if (iterator == object.end() || !iterator->is_number()) {
        return std::clamp(fallback, minimum, maximum);
    }
    const double value = iterator->get<double>();
    if (!std::isfinite(value)) {
        return std::clamp(fallback, minimum, maximum);
    }
    const double clamped = std::clamp(
        value,
        static_cast<double>(minimum),
        static_cast<double>(maximum));
    return static_cast<int>(clamped);
}

bool boolean_value_or_default(
    const nlohmann::json& object,
    const char* field,
    bool fallback) {
    const auto iterator = object.find(field);
    return iterator != object.end() && iterator->is_boolean()
        ? iterator->get<bool>()
        : fallback;
}

// v3 upgrade: restores the OpenGL techniques subtree introduced in session
// version 3. Sessions older than v3 keep the defaults in RenderSettings.
void restore_opengl_settings(
    OpenGlRenderSettings& settings,
    const nlohmann::json& render) {
    const auto& opengl = render.at("opengl");
    settings.ibl_enabled = opengl.value("ibl_enabled", true);
    settings.ltc_area_lights_enabled =
        opengl.value("ltc_area_lights_enabled", true);
        settings.ibl_enabled =
            opengl.value("ibl_enabled", true);
        settings.ltc_area_lights_enabled =
            opengl.value("ltc_area_lights_enabled", true);
        if (opengl.contains("npr")) {
            const auto& source = opengl.at("npr");
            auto& npr = settings.npr;
            npr.style = static_cast<OpenGlRenderStyle>(std::clamp(source.value("style", 0), 0, 2));
            npr.toon_levels = std::clamp(source.value("toon_levels", 3), 2, 6);
            npr.outline_width = finite_value_or_default_clamped(source, "outline_width", 1.5f, 0.0f, 4.0f);
            npr.outline_strength = finite_value_or_default_clamped(source, "outline_strength", 0.85f, 0.0f, 1.0f);
            npr.sketch_scale = finite_value_or_default_clamped(source, "sketch_scale", 8.0f, 0.5f, 32.0f);
            npr.sketch_tone = finite_value_or_default_clamped(source, "sketch_tone", 1.0f, 0.25f, 2.0f);
            npr.sketch_use_uv = source.value("sketch_use_uv", false);
        }
        if (opengl.contains("shadow_map")) {
            const auto& shadow = opengl.at("shadow_map");
            auto& target = settings.shadow_map;
            target.enabled = shadow.value("enabled", true);
            target.resolution = std::clamp(
                shadow.value("resolution", 1024), 128, 4096);
            target.max_shadow_lights = std::clamp(
                shadow.value("max_shadow_lights", 8), 1, 32);
            target.constant_bias = finite_value_clamped(
                shadow, "constant_bias", 0.0005f, 0.0f, 0.05f);
            target.slope_bias = finite_value_clamped(
                shadow, "slope_bias", 0.0025f, 0.0f, 0.1f);
            target.projection_padding = finite_value_clamped(
                shadow, "projection_padding", 0.05f, 0.0f, 0.5f);
            target.debug_view = static_cast<OpenGlShadowDebugView>(std::clamp(
                shadow.value("debug_view", 0), 0, 3));
            target.debug_shadow_slot = std::max(
                0, shadow.value("debug_shadow_slot", 0));
        }
        if (opengl.contains("pcss")) {
            const auto& pcss = opengl.at("pcss");
            auto& target = settings.pcss;
            target.enabled = pcss.value("enabled", true);
            target.blocker_samples = std::clamp(
                pcss.value("blocker_samples", 16), 1, 64);
            target.filter_samples = std::clamp(
                pcss.value("filter_samples", 32), 1, 64);
            target.max_penumbra_texels = finite_value_clamped(
                pcss, "max_penumbra_texels", 64.0f, 0.0f, 256.0f);
            target.light_size_scale = finite_value_clamped(
                pcss, "light_size_scale", 1.0f, 0.0f, 8.0f);
        }
        if (opengl.contains("dominant_light")) {
            const auto& dominant = opengl.at("dominant_light");
            auto& target = settings.dominant_light;
            target.enabled = dominant.value("enabled", true);
            target.peak_threshold_ev = finite_value_clamped(
                dominant, "peak_threshold_ev", 3.0f, 0.0f, 20.0f);
            target.minimum_energy_fraction = finite_value_clamped(
                dominant, "minimum_energy_fraction", 0.01f, 0.0f, 1.0f);
            target.intensity_scale = finite_value_clamped(
                dominant, "intensity_scale", 1.0f, 0.0f, 8.0f);
        }
        if (opengl.contains("ambient_occlusion")) {
            const auto& ao = opengl.at("ambient_occlusion");
            auto& target = settings.ambient_occlusion;
            target.mode = static_cast<OpenGlAmbientOcclusionMode>(std::clamp(
                ao.value("mode", static_cast<int>(OpenGlAmbientOcclusionMode::Gtao)),
                0,
                2));
            target.debug_view = static_cast<OpenGlAmbientOcclusionDebugView>(
                std::clamp(ao.value("debug_view", 0), 0, 4));
            if (ao.contains("ssao")) {
                const auto& source = ao.at("ssao");
                target.ssao.sample_count = std::clamp(
                    source.value("sample_count", 32), 8, 64);
                target.ssao.radius_scale = finite_value_or_default_clamped(
                    source, "radius_scale", 0.10f, 0.005f, 0.5f);
                target.ssao.depth_bias_fraction = finite_value_or_default_clamped(
                    source, "depth_bias_fraction", 0.02f, 0.0f, 0.2f);
                target.ssao.intensity = finite_value_or_default_clamped(
                    source, "intensity", 1.0f, 0.0f, 4.0f);
            }
            if (ao.contains("gtao")) {
                const auto& source = ao.at("gtao");
                target.gtao.slice_count = std::clamp(
                    source.value("slice_count", 3), 1, 8);
                target.gtao.samples_per_side = std::clamp(
                    source.value("samples_per_side", 3), 1, 8);
                target.gtao.radius_scale = finite_value_or_default_clamped(
                    source, "radius_scale", 0.10f, 0.005f, 0.5f);
                target.gtao.falloff_fraction = finite_value_or_default_clamped(
                    source, "falloff_fraction", 0.60f, 0.05f, 1.0f);
                target.gtao.thickness_fraction = finite_value_or_default_clamped(
                    source, "thickness_fraction", 0.20f, 0.0f, 1.0f);
                target.gtao.intensity = finite_value_or_default_clamped(
                    source, "intensity", 1.0f, 0.0f, 4.0f);
                target.gtao.bent_normals_enabled =
                    source.value("bent_normals_enabled", true);
            }
            if (ao.contains("denoise")) {
                const auto& source = ao.at("denoise");
                target.denoise.enabled = source.value("enabled", true);
                target.denoise.kernel_radius = std::clamp(
                    source.value("kernel_radius", 2), 1, 4);
                target.denoise.depth_sigma_fraction = finite_value_or_default_clamped(
                    source, "depth_sigma_fraction", 0.10f, 0.01f, 1.0f);
                target.denoise.normal_power = finite_value_or_default_clamped(
                    source, "normal_power", 8.0f, 1.0f, 64.0f);
            }
            if (target.debug_view != OpenGlAmbientOcclusionDebugView::Final) {
                settings.shadow_map.debug_view =
                    OpenGlShadowDebugView::Final;
                settings.ssr.debug_view = OpenGlSsrDebugView::Final;
                settings.ssgi.debug_view = OpenGlSsgiDebugView::Final;
            }
        }
        if (opengl.contains("ssr")) {
            const auto& ssr = opengl.at("ssr");
            auto& target = settings.ssr;
            target.enabled = ssr.value("enabled", true);
            target.max_steps = std::clamp(
                ssr.value("max_steps", 64), 8, 256);
            target.refinement_steps = std::clamp(
                ssr.value("refinement_steps", 4), 0, 16);
            target.max_distance_scale = finite_value_or_default_clamped(
                ssr, "max_distance_scale", 1.0f, 0.05f, 4.0f);
            target.thickness_scale = finite_value_or_default_clamped(
                ssr, "thickness_scale", 0.01f, 0.0005f, 0.1f);
            target.max_roughness = finite_value_clamped(
                ssr, "max_roughness", 0.9f, 0.0f, 1.0f);
            target.intensity = finite_value_clamped(
                ssr, "intensity", 1.0f, 0.0f, 4.0f);
            target.edge_fade = finite_value_clamped(
                ssr, "edge_fade", 0.15f, 0.0f, 0.5f);
            target.jitter = ssr.value("jitter", true);
            target.debug_view = static_cast<OpenGlSsrDebugView>(std::clamp(
                ssr.value("debug_view", 0), 0, 2));
            if (target.debug_view != OpenGlSsrDebugView::Final) {
                settings.shadow_map.debug_view =
                    OpenGlShadowDebugView::Final;
                settings.ambient_occlusion.debug_view =
                    OpenGlAmbientOcclusionDebugView::Final;
                settings.ssgi.debug_view = OpenGlSsgiDebugView::Final;
            }
        }
}

void restore_ssgi_settings(
    OpenGlRenderSettings& settings,
    const nlohmann::json& render) {
    const auto& opengl = render.at("opengl");
    if (!opengl.contains("ssgi")) {
        return;
    }
    const auto& ssgi = opengl.at("ssgi");
    auto& target = settings.ssgi;
    target.enabled = boolean_value_or_default(ssgi, "enabled", true);
    target.rays_per_pixel = integer_value_or_default_clamped(
        ssgi, "rays_per_pixel", 2, 1, 8);
    target.max_steps = integer_value_or_default_clamped(
        ssgi, "max_steps", 64, 8, 256);
    target.refinement_steps = integer_value_or_default_clamped(
        ssgi, "refinement_steps", 4, 0, 16);
    target.max_distance_scale = finite_value_or_default_clamped(
        ssgi, "max_distance_scale", 1.0f, 0.05f, 4.0f);
    target.thickness_scale = finite_value_or_default_clamped(
        ssgi, "thickness_scale", 0.01f, 0.0005f, 0.1f);
    target.edge_fade = finite_value_or_default_clamped(
        ssgi, "edge_fade", 0.15f, 0.0f, 0.5f);
    target.strength = finite_value_or_default_clamped(
        ssgi, "strength", 1.0f, 0.0f, 1.0f);
    target.max_history_frames = integer_value_or_default_clamped(
        ssgi, "max_history_frames", 32, 1, 64);
    target.denoise_passes = integer_value_or_default_clamped(
        ssgi, "denoise_passes", 3, 0, 4);
    target.denoise_depth_sigma_fraction =
        finite_value_or_default_clamped(
            ssgi,
            "denoise_depth_sigma_fraction",
            0.05f,
            0.005f,
            0.5f);
    target.denoise_normal_power = finite_value_or_default_clamped(
        ssgi, "denoise_normal_power", 16.0f, 1.0f, 64.0f);
    target.debug_view = static_cast<OpenGlSsgiDebugView>(
        integer_value_or_default_clamped(
            ssgi, "debug_view", 0, 0, 5));
    if (target.debug_view != OpenGlSsgiDebugView::Final) {
        settings.shadow_map.debug_view = OpenGlShadowDebugView::Final;
        settings.ambient_occlusion.debug_view =
            OpenGlAmbientOcclusionDebugView::Final;
        settings.ssr.debug_view = OpenGlSsrDebugView::Final;
    }
}

}  // namespace

ViewerSessionState ViewerSessionStore::load(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "failed to open viewer session: " + path.string());
    }
    nlohmann::json root;
    input >> root;
    const int version = root.value("version", 0);
    if (version < 1 || version > kViewerSessionVersion) {
        throw std::runtime_error("unsupported viewer session version");
    }

    ViewerSessionState state;
    const auto& window = root.at("window");
    state.window_width = std::clamp(window.at("width").get<int>(), 320, 8192);
    state.window_height = std::clamp(window.at("height").get<int>(), 240, 8192);

    const auto& document = root.at("document");
    state.document_path = document.value("file_path", std::string());
    state.document_dirty = document.value("dirty", false);
    state.document = SceneDocument::from_session_snapshot(
        document.at("snapshot"),
        state.window_width,
        state.window_height);
    state.document.restore_file_state(
        state.document_path,
        state.document_dirty);
    if (state.document.objects().empty()) {
        throw std::runtime_error("viewer session contains no restorable objects");
    }

    const auto& view = root.at("view");
    state.ui.mode = parse_render_mode(view.at("mode").get<std::string>());
    state.ui.camera_mode =
        parse_camera_mode(view.at("camera_mode").get<std::string>());
    state.ui.render_scale =
        finite_clamped(view, "render_scale", 0.25f, 1.0f);
    state.ui.ui_font_scale =
        finite_clamped(view, "ui_font_scale", 0.75f, 2.0f);
    state.ui.path_accumulation_paused =
        view.value("path_accumulation_paused", false);
    state.ui.automatic_interaction_quality =
        view.value("automatic_interaction_quality", true);
    state.ui.show_point_light_markers =
        view.value("show_point_light_markers", true);
    state.ui.panel_visible = view.value("panel_visible", true);
    state.ui.scene_panel_visible = view.value("scene_panel_visible", true);
    state.ui.inspector_panel_visible = view.value("inspector_panel_visible", true);
    state.ui.rendering_panel_visible = view.value("rendering_panel_visible", true);
    state.ui.camera_lighting_panel_visible = view.value(
        "camera_panel_visible",
        view.value("camera_lighting_panel_visible", true));
    state.ui.techniques_panel_visible =
        view.value("techniques_panel_visible", true);
    state.ui.active_object =
        view.value("active_object", kInvalidObjectId);
    state.ui.material_editor_object =
        view.value("material_editor_object", kInvalidObjectId);
    state.ui.selected_material_slot =
        view.value("selected_material_slot", std::size_t{0});
    state.ui.gizmo_operation =
        std::clamp(view.value("gizmo_operation", 0), 0, 2);
    state.ui.gizmo_local = view.value("gizmo_local", false);
    if (view.contains("selected_objects")) {
        for (const auto& id_json : view.at("selected_objects")) {
            const ObjectId id = id_json.get<ObjectId>();
            if (state.document.find(id)) {
                state.ui.selected_objects.push_back(id);
            }
        }
    }
    if (!state.document.find(state.ui.active_object)) {
        state.ui.active_object = kInvalidObjectId;
    }
    const SceneMeshAsset* material_asset =
        state.document.asset_for_object(state.ui.material_editor_object);
    if (!material_asset ||
        state.ui.selected_material_slot >= material_asset->local_scene.materials.size()) {
        state.ui.material_editor_object = kInvalidObjectId;
        state.ui.selected_material_slot = 0;
    }

    const auto& display = root.at("display");
    state.ui.display.exposure_ev =
        finite_clamped(display, "exposure_ev", -20.0f, 20.0f);
    state.ui.display.tone_mapper =
        parse_tone_mapper(display.at("tone_mapper").get<std::string>());

    const auto& render = root.at("render");
    state.render_settings.path.max_bounces = std::clamp(
        render.value("max_bounces", 64),
        1,
        64);
    state.render_settings.path.russian_roulette_start_bounce = std::clamp(
        render.value("rr_start_bounce", 3),
        1,
        64);
    state.render_settings.path.russian_roulette_min_probability = std::clamp(
        render.value("rr_min_probability", 0.05f),
        0.01f,
        1.0f);
    state.render_settings.path.russian_roulette_max_probability = std::clamp(
        render.value("rr_max_probability", 0.95f),
        state.render_settings.path.russian_roulette_min_probability,
        1.0f);
    state.render_settings.path.cuda_device = std::max(
        0,
        render.value("cuda_device", 0));
    state.render_settings.path.samples_per_pixel = 1;
    if (version >= 3 && render.contains("opengl")) {
        restore_opengl_settings(
            state.render_settings.opengl,
            render);
        if (version >= 5) {
            restore_ssgi_settings(
                state.render_settings.opengl,
                render);
        }
    }
    if (state.ui.mode == InteractiveRenderMode::Path) {
        std::string reason;
        if (!cuda_path_backend_available(
                state.render_settings.path.cuda_device,
                &reason)) {
            state.ui.mode = InteractiveRenderMode::OpenGl;
            state.migration_warning =
                "Saved Path session opened in OpenGL because CUDA Path is "
                "unavailable: " + reason;
        } else if (version == 1) {
            // v1 upgrade: the CPU/Auto path backends no longer exist.
            const std::string legacy_backend =
                render.value("path_backend", std::string("auto"));
            if (legacy_backend != "cuda") {
                state.migration_warning =
                    "Migrated v1 " + legacy_backend +
                    " Path session to the CUDA-only Path backend";
            }
        }
    }

    const auto& camera = root.at("camera");
    state.camera.eye = parse_vec3(camera.at("eye"), "camera.eye");
    state.camera.forward = parse_vec3(camera.at("forward"), "camera.forward");
    state.camera.up = parse_vec3(camera.at("up"), "camera.up");
    state.camera.vertical_fov_degrees =
        finite_clamped(camera, "vertical_fov_degrees", 5.0f, 175.0f);
    state.camera.orbit_distance =
        finite_clamped(camera, "orbit_distance", 0.05f, 1.0e9f);
    state.camera.free_movement_speed =
        finite_clamped(camera, "free_movement_speed", 1.0e-4f, 1.0e9f);
    if (state.camera.forward.squaredNorm() < 1.0e-12f ||
        state.camera.up.squaredNorm() < 1.0e-12f ||
        state.camera.forward.cross(state.camera.up).squaredNorm() < 1.0e-12f) {
        throw std::runtime_error("viewer session camera basis is invalid");
    }
    state.camera.forward.normalize();
    state.camera.up.normalize();
    return state;
}

void ViewerSessionStore::save(
    const std::filesystem::path& path,
    const SceneDocument& document,
    const ViewerSessionState& state) {
    nlohmann::json root;
    root["version"] = kViewerSessionVersion;
    root["document"] = {
        {"file_path", state.document_path.generic_string()},
        {"dirty", state.document_dirty},
        {"snapshot", document.session_snapshot()},
    };
    root["window"] = {
        {"width", std::clamp(state.window_width, 320, 8192)},
        {"height", std::clamp(state.window_height, 240, 8192)},
    };
    root["view"] = {
        {"mode", render_mode_name(state.ui.mode)},
        {"camera_mode", camera_mode_name(state.ui.camera_mode)},
        {"render_scale", state.ui.render_scale},
        {"ui_font_scale", state.ui.ui_font_scale},
        {"path_accumulation_paused", state.ui.path_accumulation_paused},
        {
            "automatic_interaction_quality",
            state.ui.automatic_interaction_quality,
        },
        {"show_point_light_markers", state.ui.show_point_light_markers},
        {"panel_visible", state.ui.panel_visible},
        {"scene_panel_visible", state.ui.scene_panel_visible},
        {"inspector_panel_visible", state.ui.inspector_panel_visible},
        {"rendering_panel_visible", state.ui.rendering_panel_visible},
        {"camera_panel_visible", state.ui.camera_lighting_panel_visible},
        {"techniques_panel_visible", state.ui.techniques_panel_visible},
        {"selected_objects", state.ui.selected_objects},
        {"active_object", state.ui.active_object},
        {"material_editor_object", state.ui.material_editor_object},
        {"selected_material_slot", state.ui.selected_material_slot},
        {"gizmo_operation", state.ui.gizmo_operation},
        {"gizmo_local", state.ui.gizmo_local},
    };
    root["display"] = {
        {"exposure_ev", state.ui.display.exposure_ev},
        {"tone_mapper", tone_mapper_name(state.ui.display.tone_mapper)},
    };
    root["render"] = {
        {"max_bounces", state.render_settings.path.max_bounces},
        {"cuda_device", state.render_settings.path.cuda_device},
        {
            "rr_start_bounce",
            state.render_settings.path.russian_roulette_start_bounce,
        },
        {
            "rr_min_probability",
            state.render_settings.path.russian_roulette_min_probability,
        },
        {
            "rr_max_probability",
            state.render_settings.path.russian_roulette_max_probability,
        },
    };
    root["render"]["opengl"] = {
        {"ibl_enabled", state.render_settings.opengl.ibl_enabled},
        {"ltc_area_lights_enabled", state.render_settings.opengl.ltc_area_lights_enabled},
        {"npr", {
            {"style", static_cast<int>(state.render_settings.opengl.npr.style)},
            {"toon_levels", state.render_settings.opengl.npr.toon_levels},
            {"outline_width", state.render_settings.opengl.npr.outline_width},
            {"outline_strength", state.render_settings.opengl.npr.outline_strength},
            {"sketch_scale", state.render_settings.opengl.npr.sketch_scale},
            {"sketch_tone", state.render_settings.opengl.npr.sketch_tone},
            {"sketch_use_uv", state.render_settings.opengl.npr.sketch_use_uv},
        }},
        {"shadow_map", {
            {"enabled", state.render_settings.opengl.shadow_map.enabled},
            {"resolution", state.render_settings.opengl.shadow_map.resolution},
            {"max_shadow_lights", state.render_settings.opengl.shadow_map.max_shadow_lights},
            {"constant_bias", state.render_settings.opengl.shadow_map.constant_bias},
            {"slope_bias", state.render_settings.opengl.shadow_map.slope_bias},
            {"projection_padding", state.render_settings.opengl.shadow_map.projection_padding},
            {"debug_view", static_cast<int>(state.render_settings.opengl.shadow_map.debug_view)},
            {"debug_shadow_slot", state.render_settings.opengl.shadow_map.debug_shadow_slot},
        }},
        {"pcss", {
            {"enabled", state.render_settings.opengl.pcss.enabled},
            {"blocker_samples", state.render_settings.opengl.pcss.blocker_samples},
            {"filter_samples", state.render_settings.opengl.pcss.filter_samples},
            {"max_penumbra_texels", state.render_settings.opengl.pcss.max_penumbra_texels},
            {"light_size_scale", state.render_settings.opengl.pcss.light_size_scale},
        }},
        {"dominant_light", {
            {"enabled", state.render_settings.opengl.dominant_light.enabled},
            {"peak_threshold_ev", state.render_settings.opengl.dominant_light.peak_threshold_ev},
            {"minimum_energy_fraction", state.render_settings.opengl.dominant_light.minimum_energy_fraction},
            {"intensity_scale", state.render_settings.opengl.dominant_light.intensity_scale},
        }},
        {"ambient_occlusion", {
            {"mode", static_cast<int>(state.render_settings.opengl.ambient_occlusion.mode)},
            {"debug_view", static_cast<int>(state.render_settings.opengl.ambient_occlusion.debug_view)},
            {"ssao", {
                {"sample_count", state.render_settings.opengl.ambient_occlusion.ssao.sample_count},
                {"radius_scale", state.render_settings.opengl.ambient_occlusion.ssao.radius_scale},
                {"depth_bias_fraction", state.render_settings.opengl.ambient_occlusion.ssao.depth_bias_fraction},
                {"intensity", state.render_settings.opengl.ambient_occlusion.ssao.intensity},
            }},
            {"gtao", {
                {"slice_count", state.render_settings.opengl.ambient_occlusion.gtao.slice_count},
                {"samples_per_side", state.render_settings.opengl.ambient_occlusion.gtao.samples_per_side},
                {"radius_scale", state.render_settings.opengl.ambient_occlusion.gtao.radius_scale},
                {"falloff_fraction", state.render_settings.opengl.ambient_occlusion.gtao.falloff_fraction},
                {"thickness_fraction", state.render_settings.opengl.ambient_occlusion.gtao.thickness_fraction},
                {"intensity", state.render_settings.opengl.ambient_occlusion.gtao.intensity},
                {"bent_normals_enabled", state.render_settings.opengl.ambient_occlusion.gtao.bent_normals_enabled},
            }},
            {"denoise", {
                {"enabled", state.render_settings.opengl.ambient_occlusion.denoise.enabled},
                {"kernel_radius", state.render_settings.opengl.ambient_occlusion.denoise.kernel_radius},
                {"depth_sigma_fraction", state.render_settings.opengl.ambient_occlusion.denoise.depth_sigma_fraction},
                {"normal_power", state.render_settings.opengl.ambient_occlusion.denoise.normal_power},
            }},
        }},
        {"ssgi", {
            {"enabled", state.render_settings.opengl.ssgi.enabled},
            {"rays_per_pixel", state.render_settings.opengl.ssgi.rays_per_pixel},
            {"max_steps", state.render_settings.opengl.ssgi.max_steps},
            {"refinement_steps", state.render_settings.opengl.ssgi.refinement_steps},
            {"max_distance_scale", state.render_settings.opengl.ssgi.max_distance_scale},
            {"thickness_scale", state.render_settings.opengl.ssgi.thickness_scale},
            {"edge_fade", state.render_settings.opengl.ssgi.edge_fade},
            {"strength", state.render_settings.opengl.ssgi.strength},
            {"max_history_frames", state.render_settings.opengl.ssgi.max_history_frames},
            {"denoise_passes", state.render_settings.opengl.ssgi.denoise_passes},
            {"denoise_depth_sigma_fraction", state.render_settings.opengl.ssgi.denoise_depth_sigma_fraction},
            {"denoise_normal_power", state.render_settings.opengl.ssgi.denoise_normal_power},
            {"debug_view", static_cast<int>(state.render_settings.opengl.ssgi.debug_view)},
        }},
        {"ssr", {
            {"enabled", state.render_settings.opengl.ssr.enabled},
            {"max_steps", state.render_settings.opengl.ssr.max_steps},
            {"refinement_steps", state.render_settings.opengl.ssr.refinement_steps},
            {"max_distance_scale", state.render_settings.opengl.ssr.max_distance_scale},
            {"thickness_scale", state.render_settings.opengl.ssr.thickness_scale},
            {"max_roughness", state.render_settings.opengl.ssr.max_roughness},
            {"intensity", state.render_settings.opengl.ssr.intensity},
            {"edge_fade", state.render_settings.opengl.ssr.edge_fade},
            {"jitter", state.render_settings.opengl.ssr.jitter},
            {"debug_view", static_cast<int>(state.render_settings.opengl.ssr.debug_view)},
        }},
    };
    root["camera"] = {
        {"eye", vec3_json(state.camera.eye)},
        {"forward", vec3_json(state.camera.forward)},
        {"up", vec3_json(state.camera.up)},
        {"vertical_fov_degrees", state.camera.vertical_fov_degrees},
        {"orbit_distance", state.camera.orbit_distance},
        {"free_movement_speed", state.camera.free_movement_speed},
    };
    write_file_atomically(path, root.dump(2) + '\n');
}

}  // namespace renderer
