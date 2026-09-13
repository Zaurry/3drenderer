#pragma once
#include "render/pathtracer/cuda_pathtracer.h"

namespace renderer {
enum class OptixRealtimePass { Primary, Lighting, NativeOptics };
// Owns driver-backed RT acceleration and programs. The CUDA renderer owns all
// shading/history buffers; launch parameters carry their device addresses.
class OptixRealtimeBackend {
public:
    explicit OptixRealtimeBackend(CudaDeviceContext);
    ~OptixRealtimeBackend();
    bool sync(const RenderSceneSnapshot&,CudaStreamHandle);
    void launch(OptixRealtimePass,const void* parameters,std::size_t bytes,int width,int height,CudaStreamHandle);
    std::uint64_t traversable() const;
    std::uint64_t resident_bytes() const;
    const std::string& reason() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}
