#include "render/interactive/path_interactive_session.h"

#include <algorithm>

namespace renderer {

void PathInteractiveSession::reset(const Scene& scene, const RenderSettings& settings) {
    (void)scene;
    reset_accumulation(settings.width, settings.height);
}

void PathInteractiveSession::render_next_frame(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    Framebuffer& target) {
    const bool dimensions_changed = settings.width != width_ || settings.height != height_;
    if (dimensions_changed ||
        frame_state.camera_changed ||
        frame_state.scene_changed ||
        frame_state.framebuffer_resized ||
        frame_state.reset_requested) {
        reset_accumulation(settings.width, settings.height);
    }

    RenderSettings one_sample_settings = settings;
    one_sample_settings.samples_per_pixel = 1;
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
                    static_cast<double>(accumulated_samples_ + 1));
        }
    }
    ++accumulated_samples_;
}

int PathInteractiveSession::accumulated_samples() const {
    return accumulated_samples_;
}

void PathInteractiveSession::reset_accumulation(int width, int height) {
    width_ = std::max(1, width);
    height_ = std::max(1, height);
    accumulated_.assign(static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_), Color());
    accumulated_samples_ = 0;
}

}  // namespace renderer
