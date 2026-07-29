#pragma once

#include "render/pathtracer/cuda_pathtracer.h"

#include <memory>
#include <string>

namespace renderer {

enum class CudaOpenGlInteropState {
    Unavailable,
    Ready,
    Active,
    Fallback,
};

class CudaOpenGlInteropTexture {
public:
    CudaOpenGlInteropTexture();
    ~CudaOpenGlInteropTexture();

    CudaOpenGlInteropTexture(const CudaOpenGlInteropTexture&) = delete;
    CudaOpenGlInteropTexture& operator=(const CudaOpenGlInteropTexture&) = delete;

    bool initialize();
    bool begin_frame(
        int width,
        int height,
        CudaStreamHandle stream,
        CudaSurfaceHandle& surface);
    bool end_frame(CudaStreamHandle stream);
    void cancel_frame() noexcept;
    void disable(std::string reason) noexcept;
    void release_texture() noexcept;

    CudaOpenGlInteropState state() const;
    const std::string& reason() const;
    unsigned int texture() const;
    int width() const;
    int height() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace renderer
