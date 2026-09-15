#pragma once
#include "platform/opengl/cuda_opengl_interop.h"
#include "render/ddgi/cuda_ddgi_volume.h"

namespace renderer {
class OpenGlDdgiPass {
  public:
    explicit OpenGlDdgiPass(bool disable_interop = false);
    ~OpenGlDdgiPass();
    const DdgiFrameResources &update(const RenderSceneSnapshot &, const RenderSettings &,
                                     const InteractiveFrameState &);
    DdgiStatistics statistics() const {
        return statistics_;
    }

  private:
    std::unique_ptr<CudaDdgiVolume> volume_;
    std::array<CudaOpenGlInteropTexture, 3> interop_;
    std::array<unsigned int, 3> host_textures_{};
    DdgiLayout host_layout_;
    DdgiFrameResources frame_;
    DdgiStatistics statistics_;
    bool force_host_ = false, host_ = false, failed_ = false, was_enabled_ = false;
    std::uint32_t reset_generation_ = 0;
    int device_id_ = -2;
};
} // namespace renderer
