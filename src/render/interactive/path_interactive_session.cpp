#include "render/interactive/path_interactive_session.h"

#include <stdexcept>

namespace renderer {

void PathInteractiveSession::reset(
    const RenderSceneSnapshot& snapshot,
    const RenderSettings& settings) {
    std::string reason;
    if (!cuda_path_backend_available(settings.path.cuda_device, &reason)) {
        throw std::runtime_error("CUDA Path is unavailable: " + reason);
    }
    if (!cuda_renderer_ || cuda_device_ != settings.path.cuda_device) {
        CudaDeviceContext context =
            CudaDeviceContext::create(settings.path.cuda_device);
        cuda_renderer_ = std::make_unique<CudaPathInteractiveRenderer>(context);
        cuda_device_ = context.device_id();
    }
    cuda_renderer_->reset(snapshot, settings);
    accumulated_samples_ = 0;
    width_ = settings.width;
    height_ = settings.height;
}

void PathInteractiveSession::render_next_frame(
    const RenderSceneSnapshot& snapshot,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    Framebuffer& target) {
    if (!cuda_renderer_) {
        reset(snapshot, settings);
    }
    cuda_renderer_->render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        target);
    update_cuda_frame_state(settings);
}

void PathInteractiveSession::render_next_frame_to_cuda_surface(
    const RenderSceneSnapshot& snapshot,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    CudaSurfaceHandle surface) {
    if (!cuda_renderer_) {
        reset(snapshot, settings);
    }
    cuda_renderer_->render_next_frame_to_surface(
        snapshot,
        camera,
        settings,
        frame_state,
        surface);
    update_cuda_frame_state(settings);
}

void PathInteractiveSession::download_current_cuda_frame(Framebuffer& target) {
    if (!cuda_renderer_) {
        throw std::logic_error("CUDA frame download requires the active CUDA path backend");
    }
    cuda_renderer_->download_current_frame(target);
}

int PathInteractiveSession::accumulated_samples() const {
    return accumulated_samples_;
}

CudaStreamHandle PathInteractiveSession::cuda_stream_handle() const {
    return cuda_renderer_ ? cuda_renderer_->stream_handle() : 0;
}

const CudaPathStatistics* PathInteractiveSession::cuda_statistics() const {
    return cuda_renderer_ ? &cuda_renderer_->statistics() : nullptr;
}

void PathInteractiveSession::set_cuda_presentation_state(
    bool interop_active,
    bool fallback_active) {
    if (cuda_renderer_) {
        cuda_renderer_->set_presentation_state(interop_active, fallback_active);
    }
}

void PathInteractiveSession::update_cuda_frame_state(const RenderSettings& settings) {
    accumulated_samples_ = cuda_renderer_->accumulated_samples();
    width_ = settings.width;
    height_ = settings.height;
}

}  // namespace renderer
