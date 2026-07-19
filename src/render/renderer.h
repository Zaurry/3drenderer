#pragma once

#include "core/image.h"
#include "render/render_settings.h"
#include "scene/camera.h"
#include "scene/scene.h"

namespace renderer {

enum class ExecutionBackend {
    Cpu,
    Cuda
};

struct RenderResult {
    Image image;
    float seconds = 0.0f;
    ExecutionBackend backend = ExecutionBackend::Cpu;
};

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) = 0;
};

}  // namespace renderer
