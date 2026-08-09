#pragma once

#include "core/image.h"
#include "render/render_settings.h"
#include "scene/camera.h"
#include "scene/scene.h"

namespace renderer {

enum class ExecutionBackend {
    Cuda
};

struct RenderResult {
    Image image;
    float seconds = 0.0f;
    ExecutionBackend backend = ExecutionBackend::Cuda;
};

}  // namespace renderer
