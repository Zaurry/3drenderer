#pragma once

#include "render/framebuffer.h"

#include <memory>
#include <variant>

namespace renderer {

struct HostFrameHandle {
    std::shared_ptr<const Framebuffer> framebuffer;
};

struct OpenGlTextureHandle {
    unsigned int texture = 0;
    int width = 0;
    int height = 0;
    bool flip_y = false;
    // Keeps the backend resource owner alive until presentation completes.
    std::shared_ptr<const void> lifetime;
};

using RenderFrameOutput = std::variant<
    std::monostate,
    HostFrameHandle,
    OpenGlTextureHandle>;

}  // namespace renderer
