#pragma once

#include "render/framebuffer.h"
#include "render/render_settings.h"
#include "scene/camera.h"
#include "scene/scene.h"

namespace renderer {

enum class InteractiveRenderMode {
    Raster,
    Ray,
    Path,
    OpenGl,
};

struct InteractiveFrameState {
    bool camera_changed = false;
    bool scene_changed = false;
    bool lighting_changed = false;
    bool framebuffer_resized = false;
    bool reset_requested = false;
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
