#pragma once

#include "render/pathtracer/cuda_pathtracer.h"
#include <vector>

namespace renderer {

// Checks both the OptiX driver and RT Core capability on the selected device.
bool optix_realtime_available(int device, std::string* reason = nullptr);

// Readback is explicit and intended for validation/export, never steady presentation.
struct RealtimeDiagnosticPixel {
    float depth = 0;
    float motion_x = 0, motion_y = 0;
    float history = 0, variance = 0, reactive = 0, rejected = 0;
};

class OptixRealtimeRenderer {
public:
    explicit OptixRealtimeRenderer(CudaDeviceContext context);
    ~OptixRealtimeRenderer();
    OptixRealtimeRenderer(const OptixRealtimeRenderer&) = delete;
    OptixRealtimeRenderer& operator=(const OptixRealtimeRenderer&) = delete;
    void reset(const RenderSceneSnapshot&, const RenderSettings&);
    void render_next_frame_to_surface(const RenderSceneSnapshot&, const Camera&,
        const RenderSettings&, const InteractiveFrameState&, CudaSurfaceHandle);
    void render_next_frame(const RenderSceneSnapshot&, const Camera&,
        const RenderSettings&, const InteractiveFrameState&, Framebuffer&);
    void download_current_frame(Framebuffer&);
    std::vector<RealtimeDiagnosticPixel> download_diagnostics();
    CudaStreamHandle stream_handle() const;
    const CudaPathStatistics& statistics() const;
    void refresh_statistics();
    void set_presentation_state(bool interop, bool fallback);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace renderer
