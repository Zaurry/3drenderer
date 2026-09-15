#pragma once
#include "render/ddgi/ddgi_types.h"
#include "render/pathtracer/cuda_pathtracer.h"

namespace renderer {

class CudaDdgiVolume {
  public:
    explicit CudaDdgiVolume(CudaDeviceContext context);
    ~CudaDdgiVolume();
    CudaDdgiVolume(const CudaDdgiVolume &) = delete;
    CudaDdgiVolume &operator=(const CudaDdgiVolume &) = delete;
    void update(const RenderSceneSnapshot &, const OpenGlRenderSettings &,
                const InteractiveFrameState &);
    void export_atlases(CudaSurfaceHandle irradiance, CudaSurfaceHandle distance,
                        CudaSurfaceHandle metadata);
    DdgiAtlasReadback download_atlases();
    CudaStreamHandle stream_handle() const;
    const DdgiStatistics &statistics() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace renderer
