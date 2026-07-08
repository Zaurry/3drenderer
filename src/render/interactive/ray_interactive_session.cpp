#include "render/interactive/ray_interactive_session.h"

namespace renderer {

namespace {

void copy_image_to_framebuffer(const Image& image, Framebuffer& target) {
    if (target.width() != image.width() || target.height() != image.height()) {
        target.resize(image.width(), image.height());
    }
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            target.set_pixel(x, y, image.pixel(x, y));
        }
    }
}

}  // namespace

void RayInteractiveSession::reset(const Scene& scene, const RenderSettings& settings) {
    (void)scene;
    (void)settings;
}

void RayInteractiveSession::render_next_frame(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    Framebuffer& target) {
    (void)frame_state;
    copy_image_to_framebuffer(renderer_.render(scene, camera, settings).image, target);
}

}  // namespace renderer
