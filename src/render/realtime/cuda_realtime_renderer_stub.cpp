#include "render/realtime/cuda_realtime_renderer.h"
#include <stdexcept>
namespace renderer {
namespace { [[noreturn]] void unavailable(){throw std::runtime_error("RTRT is unavailable: renderer was built without CUDA support");} }
class CudaRealtimeRenderer::Impl {};
CudaRealtimeRenderer::CudaRealtimeRenderer(CudaDeviceContext):impl_(std::make_unique<Impl>()){unavailable();}
CudaRealtimeRenderer::~CudaRealtimeRenderer()=default;
void CudaRealtimeRenderer::reset(const RenderSceneSnapshot&,const RenderSettings&){unavailable();}
void CudaRealtimeRenderer::set_primary_visibility(std::shared_ptr<RealtimePrimaryVisibility>){unavailable();}
void CudaRealtimeRenderer::render_next_frame_to_surface(const RenderSceneSnapshot&,const Camera&,const RenderSettings&,const InteractiveFrameState&,CudaSurfaceHandle){unavailable();}
void CudaRealtimeRenderer::render_next_frame(const RenderSceneSnapshot&,const Camera&,const RenderSettings&,const InteractiveFrameState&,Framebuffer&){unavailable();}
void CudaRealtimeRenderer::download_current_frame(Framebuffer&){unavailable();}
std::vector<RealtimeDiagnosticPixel> CudaRealtimeRenderer::download_diagnostics(){unavailable();}
CudaStreamHandle CudaRealtimeRenderer::stream_handle() const{unavailable();}
const CudaPathStatistics& CudaRealtimeRenderer::statistics() const{unavailable();}
void CudaRealtimeRenderer::refresh_statistics(){unavailable();}
void CudaRealtimeRenderer::set_presentation_state(bool,bool){unavailable();}
} // namespace renderer
