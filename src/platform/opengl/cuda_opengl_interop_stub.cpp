#include "platform/opengl/cuda_opengl_interop.h"

#include <utility>

namespace renderer {

class CudaOpenGlInteropTexture::Impl {
public:
    bool initialize() {
        return false;
    }

    bool begin_frame(int, int, CudaSurfaceHandle& surface) {
        surface = 0;
        return false;
    }

    bool end_frame() {
        return false;
    }

    void cancel_frame() noexcept {}

    void disable(std::string reason) noexcept {
        state_ = CudaOpenGlInteropState::Fallback;
        reason_ = std::move(reason);
    }

    void release_texture() noexcept {}

    CudaOpenGlInteropState state() const {
        return state_;
    }

    const std::string& reason() const {
        return reason_;
    }

    unsigned int texture() const {
        return 0;
    }

    int width() const {
        return 0;
    }

    int height() const {
        return 0;
    }

private:
    CudaOpenGlInteropState state_ = CudaOpenGlInteropState::Unavailable;
    std::string reason_ = "renderer was built without CUDA/OpenGL interop support";
};

CudaOpenGlInteropTexture::CudaOpenGlInteropTexture()
    : impl_(std::make_unique<Impl>()) {}

CudaOpenGlInteropTexture::~CudaOpenGlInteropTexture() = default;

bool CudaOpenGlInteropTexture::initialize() {
    return impl_->initialize();
}

bool CudaOpenGlInteropTexture::begin_frame(
    int width,
    int height,
    CudaSurfaceHandle& surface) {
    return impl_->begin_frame(width, height, surface);
}

bool CudaOpenGlInteropTexture::end_frame() {
    return impl_->end_frame();
}

void CudaOpenGlInteropTexture::cancel_frame() noexcept {
    impl_->cancel_frame();
}

void CudaOpenGlInteropTexture::disable(std::string reason) noexcept {
    impl_->disable(std::move(reason));
}

void CudaOpenGlInteropTexture::release_texture() noexcept {
    impl_->release_texture();
}

CudaOpenGlInteropState CudaOpenGlInteropTexture::state() const {
    return impl_->state();
}

const std::string& CudaOpenGlInteropTexture::reason() const {
    return impl_->reason();
}

unsigned int CudaOpenGlInteropTexture::texture() const {
    return impl_->texture();
}

int CudaOpenGlInteropTexture::width() const {
    return impl_->width();
}

int CudaOpenGlInteropTexture::height() const {
    return impl_->height();
}

}  // namespace renderer
