#include "render/pathtracer/cuda_pathtracer.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace renderer {

namespace {
[[noreturn]] void throw_unavailable(const char* operation) {
    throw std::runtime_error(
        std::string("CUDA path backend is unavailable: renderer was built "
                    "without CUDA support (") + operation + ")");
}
}  // namespace

bool cuda_path_backend_compiled() {
    return false;
}

bool cuda_path_backend_available(std::string* reason) {
    return cuda_path_backend_available(0, reason);
}

bool cuda_path_backend_available(int, std::string* reason) {
    if (reason) {
        *reason = "renderer was built without CUDA support";
    }
    return false;
}

RenderResult render_cuda_path(
    const RenderSceneSnapshot&,
    const Camera&,
    const RenderSettings&) {
    throw_unavailable("render_cuda_path");
}

class CudaPathInteractiveRenderer::Impl {};

CudaPathInteractiveRenderer::CudaPathInteractiveRenderer()
    : impl_(std::make_unique<Impl>()) {}
CudaPathInteractiveRenderer::CudaPathInteractiveRenderer(CudaDeviceContext)
    : impl_(std::make_unique<Impl>()) {}
CudaPathInteractiveRenderer::~CudaPathInteractiveRenderer() = default;
CudaPathInteractiveRenderer::CudaPathInteractiveRenderer(CudaPathInteractiveRenderer&&) noexcept = default;
CudaPathInteractiveRenderer& CudaPathInteractiveRenderer::operator=(CudaPathInteractiveRenderer&&) noexcept = default;

void CudaPathInteractiveRenderer::reset(
    const RenderSceneSnapshot&,
    const RenderSettings&) {
    throw_unavailable("reset");
}

void CudaPathInteractiveRenderer::render_next_frame(
    const RenderSceneSnapshot&,
    const Camera&,
    const RenderSettings&,
    const InteractiveFrameState&,
    Framebuffer&) {
    throw_unavailable("render_next_frame");
}

void CudaPathInteractiveRenderer::render_next_frame_to_surface(
    const RenderSceneSnapshot&,
    const Camera&,
    const RenderSettings&,
    const InteractiveFrameState&,
    CudaSurfaceHandle) {
    throw_unavailable("render_next_frame_to_surface");
}

void CudaPathInteractiveRenderer::download_current_frame(Framebuffer&) {
    throw_unavailable("download_current_frame");
}

int CudaPathInteractiveRenderer::accumulated_samples() const {
    throw_unavailable("accumulated_samples");
}

CudaStreamHandle CudaPathInteractiveRenderer::stream_handle() const {
    throw_unavailable("stream_handle");
}

int CudaPathInteractiveRenderer::device_id() const {
    throw_unavailable("device_id");
}

const CudaPathStatistics& CudaPathInteractiveRenderer::statistics() const {
    throw_unavailable("statistics");
}

void CudaPathInteractiveRenderer::refresh_statistics() {
    throw_unavailable("refresh_statistics");
}

CudaPathDiagnosticProfile
CudaPathInteractiveRenderer::download_diagnostic_profile() {
    throw_unavailable("download_diagnostic_profile");
}

void CudaPathInteractiveRenderer::set_presentation_state(bool, bool) {
    throw_unavailable("set_presentation_state");
}

}  // namespace renderer
