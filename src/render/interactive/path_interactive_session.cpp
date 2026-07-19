#include "render/interactive/path_interactive_session.h"

#include "render/pathtracer/path_backend.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace renderer {

void PathInteractiveSession::reset(const Scene& scene, const RenderSettings& settings) {
    requested_backend_ = settings.path_backend;
    active_backend_ = resolve_path_backend(settings.path_backend);
    if (active_backend_ == ExecutionBackend::Cuda) {
        if (!cuda_renderer_) {
            cuda_renderer_ = std::make_unique<CudaPathInteractiveRenderer>();
        }
        cuda_renderer_->reset(scene, settings);
        accumulated_samples_ = 0;
        width_ = settings.width;
        height_ = settings.height;
        return;
    }
    cuda_renderer_.reset();
    reset_accumulation(settings.width, settings.height);
}

void PathInteractiveSession::render_next_frame(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    Framebuffer& target) {
    if (settings.path_backend != requested_backend_) {
        reset(scene, settings);
    }
    if (active_backend_ == ExecutionBackend::Cuda) {
        cuda_renderer_->render_next_frame(scene, camera, settings, frame_state, target);
        update_cuda_frame_state(settings);
        return;
    }

    const bool dimensions_changed = settings.width != width_ || settings.height != height_;
    if (dimensions_changed ||
        frame_state.camera_changed ||
        frame_state.scene_changed ||
        frame_state.lighting_changed ||
        frame_state.framebuffer_resized ||
        frame_state.reset_requested) {
        reset_accumulation(settings.width, settings.height);
    }

    RenderSettings one_sample_settings = settings;
    one_sample_settings.samples_per_pixel = 1;
    one_sample_settings.sample_seed_offset =
        settings.sample_seed_offset + static_cast<std::uint64_t>(accumulated_samples_) + 1ULL;
    const Image sample = renderer_.render(scene, camera, one_sample_settings).image;
    if (target.width() != sample.width() || target.height() != sample.height()) {
        target.resize(sample.width(), sample.height());
    }

    for (int y = 0; y < sample.height(); ++y) {
        for (int x = 0; x < sample.width(); ++x) {
            const int index = y * sample.width() + x;
            accumulated_[static_cast<std::size_t>(index)] += sample.pixel(x, y);
            target.set_pixel(
                x,
                y,
                accumulated_[static_cast<std::size_t>(index)] /
                    static_cast<float>(accumulated_samples_ + 1));
        }
    }
    ++accumulated_samples_;
}

void PathInteractiveSession::render_next_frame_to_cuda_surface(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    CudaSurfaceHandle surface) {
    if (settings.path_backend != requested_backend_) {
        reset(scene, settings);
    }
    if (active_backend_ != ExecutionBackend::Cuda || !cuda_renderer_) {
        throw std::logic_error("CUDA surface output requires the active CUDA path backend");
    }
    cuda_renderer_->render_next_frame_to_surface(
        scene,
        camera,
        settings,
        frame_state,
        surface);
    update_cuda_frame_state(settings);
}

void PathInteractiveSession::download_current_cuda_frame(Framebuffer& target) {
    if (active_backend_ != ExecutionBackend::Cuda || !cuda_renderer_) {
        throw std::logic_error("CUDA frame download requires the active CUDA path backend");
    }
    cuda_renderer_->download_current_frame(target);
}

int PathInteractiveSession::accumulated_samples() const {
    return accumulated_samples_;
}

ExecutionBackend PathInteractiveSession::active_backend() const {
    return active_backend_;
}

void PathInteractiveSession::reset_accumulation(int width, int height) {
    width_ = std::max(1, width);
    height_ = std::max(1, height);
    accumulated_.assign(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_), Color::Zero());
    accumulated_samples_ = 0;
}

void PathInteractiveSession::update_cuda_frame_state(const RenderSettings& settings) {
    accumulated_samples_ = cuda_renderer_->accumulated_samples();
    width_ = settings.width;
    height_ = settings.height;
}

}  // namespace renderer
