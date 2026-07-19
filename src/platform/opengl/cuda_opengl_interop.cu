#define EIGEN_NO_CUDA
#include "platform/opengl/cuda_opengl_interop.h"

#include <glad/gl.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

namespace renderer {

namespace {

std::string cuda_error_message(const char* operation, cudaError_t result) {
    std::string message = operation;
    message += ": ";
    message += cudaGetErrorString(result);
    return message;
}

std::string gl_error_message(const char* operation, GLenum error) {
    std::ostringstream message;
    message << operation << ": OpenGL error 0x"
            << std::hex << std::uppercase << static_cast<unsigned int>(error);
    return message.str();
}

void clear_gl_errors() {
    while (glGetError() != GL_NO_ERROR) {
    }
}

}  // namespace

class CudaOpenGlInteropTexture::Impl {
public:
    ~Impl() {
        release_texture();
    }

    bool initialize() {
        release_texture();
        state_ = CudaOpenGlInteropState::Unavailable;
        reason_.clear();

        int cuda_device_count = 0;
        cudaError_t result = cudaGetDeviceCount(&cuda_device_count);
        if (result != cudaSuccess) {
            return fail(cuda_error_message("cudaGetDeviceCount failed", result));
        }
        if (cuda_device_count <= 0) {
            return fail("no CUDA-capable device is available");
        }

        std::vector<int> gl_devices(static_cast<std::size_t>(cuda_device_count));
        unsigned int gl_device_count = 0;
        result = cudaGLGetDevices(
            &gl_device_count,
            gl_devices.data(),
            static_cast<unsigned int>(gl_devices.size()),
            cudaGLDeviceListAll);
        if (result != cudaSuccess) {
            return fail(cuda_error_message("cudaGLGetDevices failed", result));
        }
        if (gl_device_count == 0) {
            return fail("the current OpenGL context has no compatible CUDA device");
        }
        gl_devices.resize(gl_device_count);

        int current_device = 0;
        result = cudaGetDevice(&current_device);
        if (result != cudaSuccess) {
            return fail(cuda_error_message("cudaGetDevice failed", result));
        }
        if (std::find(gl_devices.begin(), gl_devices.end(), current_device) == gl_devices.end()) {
            current_device = gl_devices.front();
            result = cudaSetDevice(current_device);
            if (result != cudaSuccess) {
                return fail(cuda_error_message("cudaSetDevice failed", result));
            }
        }

        state_ = CudaOpenGlInteropState::Ready;
        return true;
    }

    bool begin_frame(int width, int height, CudaSurfaceHandle& output_surface) {
        output_surface = 0;
        if (state_ != CudaOpenGlInteropState::Ready &&
            state_ != CudaOpenGlInteropState::Active) {
            return false;
        }
        if (width <= 0 || height <= 0) {
            return fail("CUDA/OpenGL texture dimensions must be positive");
        }
        if (!ensure_texture(width, height)) {
            return false;
        }

        cudaError_t result = cudaGraphicsMapResources(1, &resource_, nullptr);
        if (result != cudaSuccess) {
            return fail(cuda_error_message("cudaGraphicsMapResources failed", result));
        }
        mapped_ = true;

        cudaArray_t mapped_array = nullptr;
        result = cudaGraphicsSubResourceGetMappedArray(&mapped_array, resource_, 0, 0);
        if (result != cudaSuccess) {
            return fail(cuda_error_message(
                "cudaGraphicsSubResourceGetMappedArray failed",
                result));
        }

        cudaResourceDesc description{};
        description.resType = cudaResourceTypeArray;
        description.res.array.array = mapped_array;
        result = cudaCreateSurfaceObject(&surface_, &description);
        if (result != cudaSuccess) {
            return fail(cuda_error_message("cudaCreateSurfaceObject failed", result));
        }
        output_surface = static_cast<CudaSurfaceHandle>(surface_);
        return true;
    }

    bool end_frame() {
        if (!mapped_ || surface_ == 0) {
            return fail("CUDA/OpenGL frame was not mapped");
        }

        cudaError_t destroy_result = cudaDestroySurfaceObject(surface_);
        surface_ = 0;
        cudaError_t unmap_result = cudaGraphicsUnmapResources(1, &resource_, nullptr);
        if (unmap_result == cudaSuccess) {
            mapped_ = false;
        }
        if (destroy_result != cudaSuccess) {
            return fail(cuda_error_message("cudaDestroySurfaceObject failed", destroy_result));
        }
        if (unmap_result != cudaSuccess) {
            return fail(cuda_error_message("cudaGraphicsUnmapResources failed", unmap_result));
        }

        state_ = CudaOpenGlInteropState::Active;
        return true;
    }

    void cancel_frame() noexcept {
        if (surface_ != 0) {
            cudaDestroySurfaceObject(surface_);
            surface_ = 0;
        }
        if (mapped_ && resource_) {
            cudaGraphicsUnmapResources(1, &resource_, nullptr);
            mapped_ = false;
        }
    }

    void disable(std::string reason) noexcept {
        release_texture_resources();
        state_ = CudaOpenGlInteropState::Fallback;
        reason_ = std::move(reason);
    }

    void release_texture() noexcept {
        release_texture_resources();
        if (state_ == CudaOpenGlInteropState::Active) {
            state_ = CudaOpenGlInteropState::Ready;
        }
    }

    CudaOpenGlInteropState state() const {
        return state_;
    }

    const std::string& reason() const {
        return reason_;
    }

    unsigned int texture() const {
        return texture_;
    }

    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

private:
    CudaOpenGlInteropState state_ = CudaOpenGlInteropState::Unavailable;
    std::string reason_ = "CUDA/OpenGL interop has not been initialized";
    GLuint texture_ = 0;
    cudaGraphicsResource_t resource_ = nullptr;
    cudaSurfaceObject_t surface_ = 0;
    bool mapped_ = false;
    int width_ = 0;
    int height_ = 0;

    bool ensure_texture(int width, int height) {
        if (resource_ && texture_ != 0 && width == width_ && height == height_) {
            return true;
        }
        release_texture_resources();

        clear_gl_errors();
        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA32F,
            width,
            height,
            0,
            GL_RGBA,
            GL_FLOAT,
            nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        const GLenum gl_error = glGetError();
        if (texture_ == 0 || gl_error != GL_NO_ERROR) {
            const std::string message = texture_ == 0
                ? "OpenGL failed to create the CUDA interop texture"
                : gl_error_message("CUDA interop texture creation failed", gl_error);
            release_texture_resources();
            return fail(message);
        }

        const unsigned int flags =
            cudaGraphicsRegisterFlagsWriteDiscard |
            cudaGraphicsRegisterFlagsSurfaceLoadStore;
        const cudaError_t result = cudaGraphicsGLRegisterImage(
            &resource_,
            texture_,
            GL_TEXTURE_2D,
            flags);
        if (result != cudaSuccess) {
            const std::string message = cuda_error_message(
                "cudaGraphicsGLRegisterImage failed",
                result);
            release_texture_resources();
            return fail(message);
        }

        width_ = width;
        height_ = height;
        return true;
    }

    bool fail(std::string reason) {
        disable(std::move(reason));
        return false;
    }

    void release_texture_resources() noexcept {
        cancel_frame();
        if (resource_) {
            cudaGraphicsUnregisterResource(resource_);
            resource_ = nullptr;
        }
        if (texture_ != 0) {
            glDeleteTextures(1, &texture_);
            texture_ = 0;
        }
        width_ = 0;
        height_ = 0;
    }
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
