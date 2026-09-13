#pragma once

#include "render/pathtracer/cuda_pathtracer.h"

namespace renderer {

// Optional GPU-only first-hit provider. A zero surface selects CUDA traversal.
// RGBA32UI, bottom-left origin: instance+1, asset-local triangle+1, float bits
// of perspective-correct barycentric u/v. Zero instance denotes background.
class RealtimePrimaryVisibility {
public:
    virtual ~RealtimePrimaryVisibility() = default;
    virtual CudaSurfaceHandle begin_frame(const RenderSceneSnapshot&, const Camera&,
        int width, int height, float jitter_x, float jitter_y, CudaStreamHandle) = 0;
    virtual void end_frame(CudaStreamHandle) = 0;
    virtual float gpu_milliseconds() const = 0;
    virtual std::uint64_t resident_bytes() const = 0;
    virtual const std::string& reason() const = 0;
};

} // namespace renderer
