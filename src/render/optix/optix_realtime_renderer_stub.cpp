#include "render/optix/optix_realtime_renderer.h"
#include <stdexcept>
namespace renderer {
namespace { [[noreturn]] void unavailable(){throw std::runtime_error("RTRT is unavailable: renderer was built without OptiX support");} }
bool optix_realtime_available(int, std::string* reason) {
    if(reason)*reason="renderer was built without OptiX support";
    return false;
}
class OptixRealtimeRenderer::Impl {};
OptixRealtimeRenderer::OptixRealtimeRenderer(CudaDeviceContext):impl_(std::make_unique<Impl>()){unavailable();}
OptixRealtimeRenderer::~OptixRealtimeRenderer()=default;
void OptixRealtimeRenderer::reset(const RenderSceneSnapshot&,const RenderSettings&){unavailable();}
void OptixRealtimeRenderer::render_next_frame_to_surface(const RenderSceneSnapshot&,const Camera&,const RenderSettings&,const InteractiveFrameState&,CudaSurfaceHandle){unavailable();}
void OptixRealtimeRenderer::render_next_frame(const RenderSceneSnapshot&,const Camera&,const RenderSettings&,const InteractiveFrameState&,Framebuffer&){unavailable();}
void OptixRealtimeRenderer::download_current_frame(Framebuffer&){unavailable();}
std::vector<RealtimeDiagnosticPixel> OptixRealtimeRenderer::download_diagnostics(){unavailable();}
CudaStreamHandle OptixRealtimeRenderer::stream_handle() const{unavailable();}
const CudaPathStatistics& OptixRealtimeRenderer::statistics() const{unavailable();}
void OptixRealtimeRenderer::refresh_statistics(){unavailable();}
void OptixRealtimeRenderer::set_presentation_state(bool,bool){unavailable();}
} // namespace renderer
