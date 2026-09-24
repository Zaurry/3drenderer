#pragma once

#include "render/pathtracer/cuda_pathtracer.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace renderer {

// Packed, top-left-origin float images on the renderer's CUDA device.
// Normal: world XYZ. Flow: previous-to-current displacement in pixels.
struct OptixDenoiserGuides {
    float* albedo = nullptr;
    float* normal = nullptr;
    float* flow = nullptr;
    float* trustworthiness = nullptr;
};

// Independent of the RT pipeline: usable with CUDA traversal and analytic spheres.
class OptixRealtimeDenoiser {
public:
    explicit OptixRealtimeDenoiser(CudaDeviceContext);
    ~OptixRealtimeDenoiser();
    bool prepare(int width, int height, bool temporal, CudaStreamHandle, bool lighting_aovs = false);
    bool invoke(const void* linear_rgb, bool use_history, CudaStreamHandle);
    void release(CudaStreamHandle);
    OptixDenoiserGuides guides() const;
    const void* output() const;
    // Four contiguous RGB planes: direct diffuse, indirect diffuse, reflection,
    // transmission. Available only when prepare(..., lighting_aovs=true).
    void* aov_input() const;
    const void* aov_output() const;
    std::uint64_t resident_bytes() const;
    std::uint64_t allocation_generation() const;
    const std::string& reason() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace renderer
