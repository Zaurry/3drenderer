#pragma once

#include "render/framebuffer.h"
#include "render/interactive/render_mode.h"
#include "render/render_settings.h"
#include "scene/camera.h"
#include "scene/scene.h"

#include <cstdint>

namespace renderer {

enum class SceneChange : std::uint32_t {
    None = 0,
    Geometry = 1U << 0U,
    MaterialBindings = 1U << 1U,
    Materials = 1U << 2U,
    Textures = 1U << 3U,
    Lighting = 1U << 4U,
    InstanceTransforms = 1U << 5U,
    Environment = 1U << 6U,
    All = (1U << 7U) - 1U,
};

using SceneChangeSet = SceneChange;

constexpr SceneChange operator|(SceneChange left, SceneChange right) {
    return static_cast<SceneChange>(
        static_cast<std::uint32_t>(left) |
        static_cast<std::uint32_t>(right));
}

constexpr SceneChange operator&(SceneChange left, SceneChange right) {
    return static_cast<SceneChange>(
        static_cast<std::uint32_t>(left) &
        static_cast<std::uint32_t>(right));
}

constexpr SceneChange& operator|=(SceneChange& left, SceneChange right) {
    left = left | right;
    return left;
}

constexpr bool has_scene_change(SceneChangeSet changes, SceneChange change) {
    return (
        static_cast<std::uint32_t>(changes) &
        static_cast<std::uint32_t>(change)) != 0U;
}

struct InteractiveFrameState {
    bool camera_changed = false;
    SceneChangeSet scene_changes = SceneChange::None;
    bool framebuffer_resized = false;
    bool reset_requested = false;
    bool automatic_interaction_quality = false;
    float delta_seconds = 0.0f;
};

class InteractiveRenderSession {
public:
    virtual ~InteractiveRenderSession() = default;
    virtual void reset(const Scene& scene, const RenderSettings& settings) = 0;
    virtual void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) = 0;
};

}  // namespace renderer
