#include "render/pathtracer/cuda_pathtracer.h"

#include <stdexcept>
#include <utility>

namespace renderer {

bool cuda_path_backend_compiled() {
    return false;
}

bool cuda_path_backend_available(std::string* reason) {
    if (reason) {
        *reason = "renderer was built without CUDA support";
    }
    return false;
}

RenderResult render_cuda_path(const Scene&, const Camera&, const RenderSettings&) {
    throw std::runtime_error("CUDA path backend is unavailable: renderer was built without CUDA support");
}

class CudaPathInteractiveRenderer::Impl {};

CudaPathInteractiveRenderer::CudaPathInteractiveRenderer()
    : impl_(std::make_unique<Impl>()) {}
CudaPathInteractiveRenderer::~CudaPathInteractiveRenderer() = default;
CudaPathInteractiveRenderer::CudaPathInteractiveRenderer(CudaPathInteractiveRenderer&&) noexcept = default;
CudaPathInteractiveRenderer& CudaPathInteractiveRenderer::operator=(CudaPathInteractiveRenderer&&) noexcept = default;

void CudaPathInteractiveRenderer::reset(const Scene&, const RenderSettings&) {
    throw std::runtime_error("CUDA path backend is unavailable: renderer was built without CUDA support");
}

void CudaPathInteractiveRenderer::render_next_frame(
    const Scene&,
    const Camera&,
    const RenderSettings&,
    const InteractiveFrameState&,
    Framebuffer&) {
    throw std::runtime_error("CUDA path backend is unavailable: renderer was built without CUDA support");
}

int CudaPathInteractiveRenderer::accumulated_samples() const {
    return 0;
}

}  // namespace renderer
