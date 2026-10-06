#pragma once
#include "render/dxr/dxr_settings.h"
#include "render/interactive/render_frame_output.h"
#include "render/interactive/interactive_render_session.h"
#include "scene/instanced_scene.h"
#include <memory>

namespace renderer {
class D3d12Context;
class DxrRenderer {
public:
    explicit DxrRenderer(std::shared_ptr<D3d12Context> context);
    ~DxrRenderer();
    DxrRenderer(const DxrRenderer&) = delete;
    DxrRenderer& operator=(const DxrRenderer&) = delete;
    void reset(const RenderSceneSnapshot&, const RenderSettings&);
    const RenderFrameOutput& render(const RenderSceneSnapshot&, const Camera&,
        const RenderSettings&, const InteractiveFrameState&);
    const RenderFrameOutput& output() const;
    DxrStatistics statistics() const;
    void readback(Framebuffer& destination);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace renderer
