#pragma once

#include "core/image.h"
#include "render/render_settings.h"
#include "scene/camera.h"
#include "scene/scene.h"

namespace renderer {

struct RenderResult {
    Image image;
    float seconds = 0.0f;
};

}  // namespace renderer
