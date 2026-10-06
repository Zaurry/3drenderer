#include "render/dxr/dxr_renderer.h"
#include <stdexcept>
namespace renderer {
DxrDeviceCapabilities query_dxr_capabilities() {
    DxrDeviceCapabilities c;
    c.reason="DXR requires a Windows build with RENDERER_DXR enabled";
    return c;
}
bool dxr_available(std::string* reason) {
    if(reason)*reason=query_dxr_capabilities().reason;
    return false;
}
struct DxrRenderer::Impl {};
DxrRenderer::DxrRenderer(std::shared_ptr<D3d12Context>) {
    throw std::runtime_error(query_dxr_capabilities().reason);
}
DxrRenderer::~DxrRenderer()=default;
void DxrRenderer::reset(const RenderSceneSnapshot&,const RenderSettings&) {throw std::runtime_error(query_dxr_capabilities().reason);}
const RenderFrameOutput& DxrRenderer::render(const RenderSceneSnapshot&,const Camera&,const RenderSettings&,const InteractiveFrameState&) {throw std::runtime_error(query_dxr_capabilities().reason);}
const RenderFrameOutput& DxrRenderer::output() const {static RenderFrameOutput empty;return empty;}
DxrStatistics DxrRenderer::statistics() const {DxrStatistics s;s.device=query_dxr_capabilities();return s;}
void DxrRenderer::readback(Framebuffer&) {throw std::runtime_error(query_dxr_capabilities().reason);}
}
