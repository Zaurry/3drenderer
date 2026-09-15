#include "render/ddgi/cuda_ddgi_volume.h"
#include <stdexcept>
namespace renderer {
class CudaDdgiVolume::Impl {};
CudaDdgiVolume::CudaDdgiVolume(CudaDeviceContext) {
    throw std::runtime_error("DDGI requires a CUDA build");
}
CudaDdgiVolume::~CudaDdgiVolume() = default;
void CudaDdgiVolume::update(const RenderSceneSnapshot &, const OpenGlRenderSettings &,
                            const InteractiveFrameState &) {}
void CudaDdgiVolume::export_atlases(CudaSurfaceHandle, CudaSurfaceHandle, CudaSurfaceHandle) {}
DdgiAtlasReadback CudaDdgiVolume::download_atlases() {
    return {};
}
CudaStreamHandle CudaDdgiVolume::stream_handle() const {
    return 0;
}
const DdgiStatistics &CudaDdgiVolume::statistics() const {
    static const DdgiStatistics unavailable;
    return unavailable;
}
} // namespace renderer
