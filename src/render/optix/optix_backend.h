#pragma once
#include "render/pathtracer/cuda_pathtracer.h"

namespace renderer {
enum class OptixRealtimePass { Primary, Lighting, NativeOptics };
// Owns driver-backed RT acceleration and programs. The realtime renderer owns all
// shading/history buffers; launch parameters carry their device addresses.
class OptixRealtimeBackend {
public:
    explicit OptixRealtimeBackend(CudaDeviceContext);
    ~OptixRealtimeBackend();
    bool sync(const RenderSceneSnapshot&,CudaStreamHandle);
    void launch(OptixRealtimePass,const void* parameters,std::size_t bytes,int width,int height,CudaStreamHandle);
    std::uint64_t traversable() const;
    std::uint64_t resident_bytes() const;
    unsigned rt_core_version() const;
    bool ser_supported() const;
    std::uint64_t gas_builds() const;
    std::uint64_t ias_builds() const;
    std::uint64_t ias_updates() const;
    float acceleration_ms() const;
    const std::string& reason() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}
