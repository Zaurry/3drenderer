#pragma once

#include "render/interactive/render_frame_output.h"
#include "render/pathtracer/cuda_pathtracer.h"

#include <memory>
#include <optional>
#include <string>

namespace renderer {

enum class CudaOpenGlInteropState {
    Unavailable,
    Ready,
    Active,
    Fallback,
};

std::optional<CudaDeviceContext>
select_cuda_device_for_current_opengl_context(
    int requested_device,
    std::string* reason = nullptr);

class CudaOpenGlInteropTexture : public TextureLifetimeOwner {
public:
    CudaOpenGlInteropTexture();
    ~CudaOpenGlInteropTexture();

    CudaOpenGlInteropTexture(const CudaOpenGlInteropTexture&) = delete;
    CudaOpenGlInteropTexture& operator=(const CudaOpenGlInteropTexture&) = delete;

    bool initialize(const CudaDeviceContext& device_context);
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
    int device_id() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace renderer
